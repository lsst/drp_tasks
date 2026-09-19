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

import lsst.daf.butler
import lsst.pipe.base.testUtils
import lsst.utils.tests
from lsst.drp.tasks.single_frame_detect_and_measure import (
    SingleFrameDetectAndMeasureConfig,
    SingleFrameDetectAndMeasureTask,
)


def make_config(input_image_type):
    """Return a config with ``input_image_type`` set to the given value."""
    config = SingleFrameDetectAndMeasureConfig()
    config.input_image_type = input_image_type
    return config


class ConnectionsTestCase(lsst.utils.tests.TestCase):
    """Test how input_image_type wires up the connections."""

    def make_connections(self, input_image_type):
        config = make_config(input_image_type)
        return config.connections.ConnectionsClass(config=config)

    def test_lint_connections(self):
        for input_image_type in ("legacy", "future"):
            with self.subTest(input_image_type=input_image_type):
                config = make_config(input_image_type)
                lsst.pipe.base.testUtils.lintConnections(config.connections.ConnectionsClass)

    def test_legacy(self):
        connections = self.make_connections("legacy")
        self.assertEqual(connections.exposure.storageClass, "Exposure")
        self.assertIn("input_background", connections.inputs)
        self.assertIn("background", connections.outputs)

    def test_future(self):
        """The background is part of the image in future mode, and this task
        writes no image, so both background connections are dropped.
        """
        connections = self.make_connections("future")
        self.assertEqual(connections.exposure.storageClass, "VisitImage")
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

    def run_quantum(self, input_image_type, **inputs):
        """Run a quantum with `run` mocked out.

        Parameters
        ----------
        input_image_type : `str`
            Value to set ``config.input_image_type`` to.
        **inputs
            The inputs the quantum provides, as if read from the butler.

        Returns
        -------
        kwargs : `dict`
            The keyword arguments `runQuantum` called `run` with.
        butlerQC : `_RecordingQuantumContext`
            The context that was used, holding what was put.
        """
        config = make_config(input_image_type)
        # The default dimension packer reads the instrument record, which a
        # data ID standardized outside a registry does not carry.
        config.id_generator.packer.name = "observation"
        config.id_generator.packer["observation"].n_observations = 10000
        config.id_generator.packer["observation"].n_detectors = 99
        task = SingleFrameDetectAndMeasureTask(config=config)
        butlerQC = _RecordingQuantumContext(self.dataId)
        with mock.patch.object(task, "run") as mock_run:
            task.runQuantum(butlerQC, _FakeRefs(**inputs), _FakeRefs())
        mock_run.assert_called_once()
        return mock_run.call_args.kwargs, butlerQC

    def test_run_quantum_legacy(self):
        """The exposure and the separate background input both reach `run`."""
        exposure = mock.sentinel.exposure
        background = mock.sentinel.background

        kwargs, butlerQC = self.run_quantum("legacy", exposure=exposure, input_background=background)

        self.assertIs(kwargs["exposure"], exposure)
        self.assertIs(kwargs["input_background"], background)
        self.assertIsNotNone(butlerQC.put_values)

    def test_run_quantum_future(self):
        """The image is converted to a legacy Exposure before `run` sees it,
        and `run` fits its own background, having been given none.
        """
        image = _FakeVisitImage(mock.sentinel.legacy_exposure)

        kwargs, butlerQC = self.run_quantum("future", exposure=image)

        self.assertIs(kwargs["exposure"], mock.sentinel.legacy_exposure)
        self.assertEqual(image.to_legacy_calls, 1)
        self.assertIsNone(kwargs["input_background"])
        self.assertIsNotNone(butlerQC.put_values)


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

    ``runQuantum`` only calls `to_legacy` on the image it is given, and
    drp_tasks does not depend on lsst.images, so this records that call
    instead.

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
