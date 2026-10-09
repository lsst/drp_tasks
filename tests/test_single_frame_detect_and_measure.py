# This file is part of drp_tasks.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import unittest
from unittest import mock

import astropy.units as u
import numpy as np

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image
import lsst.daf.butler
import lsst.geom
import lsst.meas.algorithms
import lsst.meas.base.tests
import lsst.pipe.base.testUtils
import lsst.utils.tests
from lsst.drp.tasks.single_frame_detect_and_measure import (
    SingleFrameDetectAndMeasureConfig,
    SingleFrameDetectAndMeasureTask,
)
from lsst.images import VisitImage
from lsst.images.fields import SumField, field_from_legacy_background
from lsst.images.tests import get_dp2_exposure_record


def make_config(image_type):
    """Return a config with ``image_type`` set to the given value."""
    config = SingleFrameDetectAndMeasureConfig()
    config.image_type = image_type
    return config


def make_exposure_and_background():
    """Build a synthetic calibrated exposure with a few point sources.

    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        Background-subtracted exposure ready for detection and measurement,
        with no DETECTED pixels.
    background : `lsst.afw.math.BackgroundList`
        Background subtracted from ``exposure``.
    """
    bbox = lsst.geom.Box2I(lsst.geom.Point2I(5, 4), lsst.geom.Point2I(205, 184))
    dataset = lsst.meas.base.tests.TestDataset(
        bbox,
        crval=lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees),
        calibration=12.3,
        detector=42,
        visitId=98765,
    )
    psf_scale = np.sqrt(4 * np.pi * (dataset.psfShape.getDeterminantRadius()) ** 2)
    noise = 10.0
    for flux, centroid in [
        (45 * noise * psf_scale, (40, 70)),
        (150 * noise * psf_scale, (50, 120)),
        (400 * noise * psf_scale, (92, 35)),
        (1000 * noise * psf_scale, (175, 154)),
    ]:
        dataset.addSource(instFlux=flux, centroid=lsst.geom.Point2D(*centroid))
    exposure, _ = dataset.realize(noise=noise, schema=dataset.makeMinimalSchema())
    exposure.mask.clearMaskPlane(exposure.mask.getMaskPlane("DETECTED"))
    exposure.info.setApCorrMap(lsst.afw.image.ApCorrMap())

    background_config = lsst.meas.algorithms.SubtractBackgroundTask.ConfigClass()
    background_config.approxOrderX = 1
    background_task = lsst.meas.algorithms.SubtractBackgroundTask(config=background_config)
    background = background_task.run(exposure).background
    return exposure, background


class ConnectionsTestCase(lsst.utils.tests.TestCase):
    """Test how image_type wires up the connections."""

    def make_connections(self, image_type):
        config = make_config(image_type)
        return config.connections.ConnectionsClass(config=config)

    def test_lint_connections(self):
        for image_type in ("legacy", "future"):
            with self.subTest(image_type=image_type):
                config = make_config(image_type)
                lsst.pipe.base.testUtils.lintConnections(config.connections.ConnectionsClass)

    def test_legacy(self):
        connections = self.make_connections("legacy")
        self.assertEqual(connections.exposure.storageClass, "Exposure")
        self.assertEqual(connections.output_exposure.storageClass, "Exposure")
        self.assertIn("input_background", connections.inputs)
        self.assertIn("background", connections.outputs)

    def test_future(self):
        """The backgrounds are part of the images in future mode, so both
        background connections are dropped.
        """
        connections = self.make_connections("future")
        self.assertEqual(connections.exposure.storageClass, "VisitImage")
        self.assertEqual(connections.output_exposure.storageClass, "VisitImage")
        self.assertNotIn("input_background", connections.inputs)
        self.assertNotIn("background", connections.outputs)


class RunQuantumTestCase(lsst.utils.tests.TestCase):
    """Test what `runQuantum` hands to `run` for each input image type."""

    def setUp(self):
        self.dataId = lsst.daf.butler.DataCoordinate.standardize(
            instrument="I",
            visit=42,
            detector=12,
            universe=lsst.daf.butler.DimensionUniverse(),
        )

    def run_quantum(self, image_type, **inputs):
        """Run a quantum with `run` mocked out.

        Parameters
        ----------
        image_type : `str`
            Value to set ``config.image_type`` to.
        **inputs
            The inputs the quantum provides, as if read from the butler.

        Returns
        -------
        kwargs : `dict`
            The keyword arguments `runQuantum` called `run` with.
        butlerQC : `_RecordingQuantumContext`
            The context that was used, holding what was put.
        """
        config = make_config(image_type)
        # The default dimension packer reads the instrument record, which a
        # data ID standardized outside a registry does not carry.
        config.id_generator.packer.name = "observation"
        config.id_generator.packer["observation"].n_observations = 10000
        config.id_generator.packer["observation"].n_detectors = 99
        task = SingleFrameDetectAndMeasureTask(config=config)
        butlerQC = _RecordingQuantumContext(self.dataId)
        with (
            mock.patch.object(task, "run") as mock_run,
            mock.patch.object(task, "_make_future_output") as mock_make_future_output,
        ):
            task.runQuantum(butlerQC, _FakeRefs(**inputs), _FakeRefs())
        mock_run.assert_called_once()
        self.make_future_output_calls = mock_make_future_output.call_args_list
        return mock_run.call_args.kwargs, butlerQC

    def test_run_quantum_legacy(self):
        """The exposure and the separate background input both reach `run`."""
        exposure = mock.sentinel.exposure
        background = mock.sentinel.background

        kwargs, butlerQC = self.run_quantum("legacy", exposure=exposure, input_background=background)

        self.assertIs(kwargs["exposure"], exposure)
        self.assertIs(kwargs["input_background"], background)
        self.assertIsNotNone(butlerQC.put_values)
        self.assertEqual(self.make_future_output_calls, [])

    def test_run_quantum_future(self):
        """The image is converted to a legacy Exposure before `run` sees it,
        `run` fits its own background, having been given none, and the
        input image becomes the output.
        """
        image = _FakeVisitImage(mock.sentinel.legacy_exposure)

        kwargs, butlerQC = self.run_quantum("future", exposure=image)

        self.assertIs(kwargs["exposure"], mock.sentinel.legacy_exposure)
        self.assertEqual(image.to_legacy_calls, 1)
        self.assertIsNone(kwargs["input_background"])
        self.assertIsNotNone(butlerQC.put_values)
        self.assertEqual(len(self.make_future_output_calls), 1)
        self.assertIs(self.make_future_output_calls[0].args[0], image)


class RunImageOutputTestCase(lsst.utils.tests.TestCase):
    """Test the output exposure on a synthetic image."""

    def setUp(self):
        self.exposure, self.background = make_exposure_and_background()
        self.input_exposure = self.exposure.clone()

    def make_task(self, image_type):
        config = make_config(image_type)
        # Adjust the config for a small test image.
        config.detection.background.approxOrderX = 1
        config.sky_sources.nSources = 2
        return SingleFrameDetectAndMeasureTask(config=config)

    def check_mask(self, output_mask, sources):
        """Check that only DETECTED changed, and that it covers the
        footprints of ``sources``.
        """
        detected = self.input_exposure.mask.getPlaneBitMask("DETECTED")
        np.testing.assert_array_equal(output_mask & ~detected, self.input_exposure.mask.array & ~detected)
        is_detected = (output_mask & detected) > 0
        self.assertTrue(np.any(is_detected))
        x0, y0 = self.exposure.getXY0()
        for source in sources:
            if source["sky_source"]:
                continue
            ys, xs = source.getFootprint().spans.indices()
            self.assertTrue(np.all(is_detected[ys - y0, xs - x0]))

    def test_run(self):
        """The output exposure is the input exposure, with only its DETECTED
        mask plane replaced.
        """
        task = self.make_task("legacy")
        n_input_background = len(self.background)

        result = task.run(exposure=self.exposure, input_background=self.background)

        self.assertIs(result.output_exposure, self.exposure)
        np.testing.assert_array_equal(result.output_exposure.image.array, self.input_exposure.image.array)
        np.testing.assert_array_equal(
            result.output_exposure.variance.array, self.input_exposure.variance.array
        )
        self.check_mask(result.output_exposure.mask.array, result.sources_footprints)
        # The background fit here is added to the input background.
        self.assertGreater(len(result.background), n_input_background)

    def test_make_future_output(self):
        """The future output keeps the input's subtracted background and adds
        the total background, not subtracted.
        """
        task = self.make_task("future")
        # The TestDataset detector has no amplifier geometry, which
        # `VisitImage` requires.
        self.exposure.setDetector(list(lsst.afw.cameraGeom.testUtils.CameraWrapper().camera)[0])
        self.exposure.setFilter(lsst.afw.image.FilterLabel(band="r", physical="r_57"))
        record = get_dp2_exposure_record(lsst.daf.butler.DimensionUniverse())
        record = type(record)(**(record.toDict() | {"id": self.exposure.visitInfo.id}))
        image = VisitImage.from_legacy(self.exposure, unit=u.nJy, exposure_record=record)
        subtracted = field_from_legacy_background(self.background, bounds=image.bbox, unit=image.unit)
        image.backgrounds.add("subtracted", subtracted, is_subtracted=True)
        input_image = image.copy()

        legacy = image.to_legacy()
        # An empty aperture correction map is not carried by `VisitImage`.
        legacy.info.setApCorrMap(lsst.afw.image.ApCorrMap())
        result = task.run(exposure=legacy, input_background=None)
        output = task._make_future_output(image, result)

        self.assertIs(output, image)
        np.testing.assert_array_equal(output.image.array, input_image.image.array)
        np.testing.assert_array_equal(output.variance.array, input_image.variance.array)
        self.assertEqual(output.mask.compare(input_image.mask).keys(), {"DETECTED"})
        np.testing.assert_array_equal(
            output.mask.get("DETECTED"),
            result.output_exposure.mask.array & result.output_exposure.mask.getPlaneBitMask("DETECTED") > 0,
        )
        self.assertEqual(output.backgrounds.subtracted.name, "subtracted")
        self.assertIs(output.backgrounds.subtracted.field, subtracted)
        source_detection = output.backgrounds["sourceDetection"].field
        self.assertIsInstance(source_detection, SumField)
        self.assertIs(source_detection.operands[0], subtracted)
        self.assertEqual(source_detection.unit, image.unit)


class _FakeRefs:
    """Stand-in for an input or output ``QuantizedConnection``."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RecordingQuantumContext:
    """Minimal `~lsst.pipe.base.QuantumContext` that records what was put.

    Parameters
    ----------
    dataId : `lsst.daf.butler.DataCoordinate`
        Data ID to report as the quantum data ID.
    """

    def __init__(self, dataId):
        self.quantum = _FakeRefs(dataId=dataId)
        self.put_values = None

    def get(self, refs):
        return dict(refs.__dict__)

    def put(self, values, refs):
        self.put_values = values


class _FakeVisitImage:
    """Stand-in for an `lsst.images.VisitImage` input.

    ``runQuantum`` only calls `to_legacy` on the image it is given before
    calling `run`, so this records that call instead.

    Parameters
    ----------
    legacy : `object`
        The value `to_legacy` returns.
    """

    def __init__(self, legacy):
        self.legacy = legacy
        self.to_legacy_calls = 0

    def to_legacy(self):
        self.to_legacy_calls += 1
        return self.legacy


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
