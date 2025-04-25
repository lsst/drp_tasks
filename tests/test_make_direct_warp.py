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

from __future__ import annotations

import copy
import unittest
from typing import Self, Type

import numpy as np

import lsst.afw.cameraGeom.testUtils
import lsst.afw.image
import lsst.afw.math as afwMath
import lsst.skymap as skyMap
import lsst.utils.tests
from lsst.afw.detection import GaussianPsf
from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.drp.tasks.make_direct_warp import MakeDirectWarpConfig, MakeDirectWarpTask, WarpDetectorInputs
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.coaddBase import makeSkyInfo


class MakeWarpTestCase(lsst.utils.tests.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(12345)

        rng = np.random.Generator(np.random.MT19937(12345))

        cls.config = MakeDirectWarpConfig()
        cls.config.useVisitSummaryPsf = False
        cls.config.doSelectPreWarp = False
        cls.config.doWarpMaskedFraction = True
        cls.config.numberOfNoiseRealizations = 1

        cls.exposurePhotoCalib = lsst.afw.image.PhotoCalib(1.0)
        # An external photoCalib calibration to return
        cls.externalPhotoCalib = lsst.afw.image.PhotoCalib(1e-6, 1e-8)

        crpix = lsst.geom.Point2D(0, 0)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0 * lsst.geom.arcseconds)
        cls.skyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)
        externalCdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.9 * lsst.geom.arcseconds)
        # An external skyWcs to return
        cls.externalSkyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, externalCdMatrix)

        cls.exposure = lsst.afw.image.ExposureF(100, 150)
        cls.exposure.maskedImage.image.array = rng.random((150, 100)).astype(np.float32) * 1000
        cls.exposure.maskedImage.variance.array = rng.random((150, 100)).astype(np.float32)
        # mask at least one pixel
        cls.exposure.maskedImage.mask[5, 5] = 3
        # set the PhotoCalib and Wcs objects of this exposure.
        cls.exposure.setPhotoCalib(lsst.afw.image.PhotoCalib(1.0))
        cls.exposure.setWcs(cls.skyWcs)
        cls.exposure.setPsf(GaussianPsf(5, 5, 2.5))
        cls.exposure.setFilter(lsst.afw.image.FilterLabel(physical="fakeFilter", band="fake"))

        cls.backgroundToPhotometricRatio = lsst.afw.image.ImageF(100, 150)
        cls.backgroundToPhotometricRatio.array[:, :] = 1.1

        cls.visit = 100
        cls.detector = 5
        detectorName = f"detector {cls.detector}"
        detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(name=detectorName, id=cls.detector).detector
        cls.exposure.setDetector(detector)

        dataId_dict = {"detector_id": cls.detector, "visit_id": 1248, "band": "i"}
        cls.dataId = cls.generate_data_id(**dataId_dict)
        simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
        simpleMapConfig.raList = [crval.getRa().asDegrees()]
        simpleMapConfig.decList = [crval.getDec().asDegrees()]
        simpleMapConfig.radiusList = [0.1]

        cls.simpleMap = skyMap.DiscreteSkyMap(simpleMapConfig)
        cls.tractId = 0
        cls.patchId = cls.simpleMap[0].findPatch(crval).sequential_index
        cls.skyInfo = makeSkyInfo(cls.simpleMap, cls.tractId, cls.patchId)

    @classmethod
    def generate_data_id(
        cls: Type[Self],
        *,
        tract: int = 9813,
        patch: int = 42,
        band: str = "r",
        detector_id: int = 9,
        visit_id: int = 1234,
        detector_max: int = 109,
        visit_max: int = 10000,
    ) -> DataCoordinate:
        """Generate a DataCoordinate instance to use as data_id.

        Parameters
        ----------
        tract : `int`, optional
            Tract ID for the data_id
        patch : `int`, optional
            Patch ID for the data_id
        band : `str`, optional
            Band for the data_id
        detector_id : `int`, optional
            Detector ID for the data_id
        visit_id : `int`, optional
            Visit ID for the data_id
        detector_max : `int`, optional
            Maximum detector ID for the data_id
        visit_max : `int`, optional
            Maximum visit ID for the data_id

        Returns
        -------
        data_id : `lsst.daf.butler.DataCoordinate`
            An expanded data_id instance.
        """
        universe = DimensionUniverse()

        instrument = universe["instrument"]
        instrument_record = instrument.RecordClass(
            name="DummyCam",
            class_name="lsst.obs.base.instrument_tests.DummyCam",
            detector_max=detector_max,
            visit_max=visit_max,
        )

        skymap = universe["skymap"]
        skymap_record = skymap.RecordClass(name="test_skymap")

        band_element = universe["band"]
        band_record = band_element.RecordClass(name=band)

        visit = universe["visit"]
        visit_record = visit.RecordClass(id=visit_id, instrument="test")

        detector = universe["detector"]
        detector_record = detector.RecordClass(id=detector_id, instrument="test")

        physical_filter = universe["physical_filter"]
        physical_filter_record = physical_filter.RecordClass(name=band, instrument="test", band=band)

        patch_element = universe["patch"]
        patch_record = patch_element.RecordClass(
            skymap="test_skymap",
            tract=tract,
            patch=patch,
        )

        if "day_obs" in universe:
            day_obs_element = universe["day_obs"]
            day_obs_record = day_obs_element.RecordClass(id=20240201, instrument="test")
        else:
            day_obs_record = None

        # A dictionary with all the relevant records.
        record = {
            "instrument": instrument_record,
            "visit": visit_record,
            "detector": detector_record,
            "patch": patch_record,
            "tract": 9813,
            "band": band_record.name,
            "skymap": skymap_record.name,
            "physical_filter": physical_filter_record,
        }

        if day_obs_record:
            record["day_obs"] = day_obs_record

        # A dictionary with all the relevant recordIds.
        record_id = record.copy()
        for key in ("visit", "detector"):
            record_id[key] = record_id[key].id

        # TODO: Catching mypy failures on Github Actions should be made easier,
        # perhaps in DM-36873. Igroring these for now.
        data_id = DataCoordinate.standardize(record_id, universe=universe)
        return data_id.expanded(record)

    def test_makeWarp(self):
        """Test basic MakeDirectWarpTask."""
        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        makeWarp = MakeDirectWarpTask(config=config)
        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
            )
        }
        result = makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        warp = result.warp
        mfrac = result.masked_fraction_warp
        noise = result.noise_warp0

        # Ensure we got an exposure out
        self.assertIsInstance(warp, lsst.afw.image.ExposureF)
        # Ensure that masked fraction is an ImageF object.
        self.assertIsInstance(mfrac, lsst.afw.image.ImageF)
        # Ensure that the noise image is a MaskedImageF object.
        self.assertIsInstance(noise, lsst.afw.image.MaskedImageF)
        # Check that the noise image is not accidentally the same as the image.
        with self.assertRaises(AssertionError):
            self.assertImagesAlmostEqual(noise.image, warp.image)
        # Ensure the warp has valid pixels
        self.assertGreater(np.isfinite(warp.image.array.ravel()).sum(), 0)
        # Ensure the warp has the correct WCS
        self.assertEqual(warp.getWcs(), self.skyInfo.wcs)
        # Ensure that mfrac has pixels between 0 and 1
        self.assertTrue(np.nanmax(mfrac.array) <= 1)
        self.assertTrue(np.nanmin(mfrac.array) >= 0)

    def make_backgroundList(self):
        """Obtain a BackgroundList for the image."""
        bgCtrl = afwMath.BackgroundControl(10, 10)
        interpStyle = afwMath.Interpolate.AKIMA_SPLINE
        undersampleStyle = afwMath.REDUCE_INTERP_ORDER
        approxStyle = afwMath.ApproximateControl.UNKNOWN
        approxOrderX = 0
        approxOrderY = 0
        approxWeighting = False

        backgroundList = afwMath.BackgroundList()

        bkgd = afwMath.makeBackground(self.exposure.image, bgCtrl)

        backgroundList.append(
            (bkgd, interpStyle, undersampleStyle, approxStyle, approxOrderX, approxOrderY, approxWeighting)
        )

        return backgroundList

    @lsst.utils.tests.methodParametersProduct(
        doApplyNewBackground=[True, False], doRevertOldBackground=[True, False]
    )
    def test_backgrounds(self, doApplyNewBackground, doRevertOldBackground):
        """Test that applying and reverting backgrounds runs without errors,
        especially on noise images.
        """
        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        backgroundList = self.make_backgroundList()

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=backgroundList if doApplyNewBackground else None,
                background_revert=backgroundList if doRevertOldBackground else None,
            )
        }

        config.numberOfNoiseRealizations = 1
        config.doApplyNewBackground = doApplyNewBackground
        config.doRevertOldBackground = doRevertOldBackground

        makeWarp = MakeDirectWarpTask(config=config)
        makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

    def test_flat_background_ratio(self):
        """Test that using the flat background ratio works."""
        backgroundList = self.make_backgroundList()

        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        warp_detector_inputs_basic = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
            )
        }

        makeWarpBasic = MakeDirectWarpTask(config=config)
        resultBasic = makeWarpBasic.run(
            warp_detector_inputs_basic,
            sky_info=copy.deepcopy(self.skyInfo),
            visit_summary=None,
        )

        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        backgroundRatioDataRef = InMemoryDatasetHandle(
            self.backgroundToPhotometricRatio.clone(),
            dataId=self.dataId,
        )

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=backgroundList,
                background_revert=backgroundList,
                background_ratio_or_handle=backgroundRatioDataRef,
            )
        }

        config.numberOfNoiseRealizations = 1
        config.doApplyNewBackground = True
        config.doRevertOldBackground = True
        config.doApplyFlatBackgroundRatio = True

        makeWarp = MakeDirectWarpTask(config=config)
        result = makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        finite = np.isfinite(result.warp.image.array)
        delta = result.warp.image.array[finite] - resultBasic.warp.image.array[finite]
        self.assertFloatsAlmostEqual(np.median(delta), 0.0, atol=1e-6)

    def test_background_errors(self):
        """Test that MakeDirectWarpTask raises errors when backgrounds are not
        set correctly.
        """
        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        backgroundList = self.make_backgroundList()
        config.numberOfNoiseRealizations = 1

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=backgroundList,
            )
        }
        makeWarp = MakeDirectWarpTask(config=config)
        with self.assertRaises(RuntimeError, msg="doApplyNewBackground is False, but"):
            makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=None,
            )
        }
        config.doApplyNewBackground = True
        makeWarp = MakeDirectWarpTask(config=config)
        with self.assertRaises(RuntimeError, msg="No background to apply"):
            makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=backgroundList,
                background_revert=backgroundList,
            )
        }
        makeWarp = MakeDirectWarpTask(config=config)
        with self.assertRaises(RuntimeError, msg="doRevertOldBackground is False, but"):
            makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
                background_apply=backgroundList,
                background_revert=None,
            )
        }
        config.doRevertOldBackground = True
        makeWarp = MakeDirectWarpTask(config=config)
        with self.assertRaises(RuntimeError, msg="No background to revert"):
            makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

    def test_long_data_ids(self):
        """Test MakeDirectWarpTask fails gracefully with no good pixels.

        It should return an empty exposure, with no PSF.
        """
        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
            )
        }

        config.border = 0  # Repeated calls will expand it otherwise.
        makeWarp_original = MakeDirectWarpTask(config=config)
        makeWarp_short = MakeDirectWarpTask(config=config)
        makeWarp_short.get_seed_from_data_id = (
            lambda data_id: 2**32 - 1 + makeWarp_original.get_seed_from_data_id(data_id)
        )
        makeWarp_long = MakeDirectWarpTask(config=config)
        makeWarp_long.get_seed_from_data_id = lambda data_id: 2**32 + makeWarp_original.get_seed_from_data_id(
            data_id
        )

        result_long = makeWarp_long.run(
            warp_detector_inputs,
            sky_info=copy.deepcopy(self.skyInfo),
            visit_summary=None,
        )
        result_short = makeWarp_short.run(
            warp_detector_inputs,
            sky_info=copy.deepcopy(self.skyInfo),
            visit_summary=None,
        )
        result_original = makeWarp_original.run(
            warp_detector_inputs,
            sky_info=copy.deepcopy(self.skyInfo),
            visit_summary=None,
        )

        self.assertMaskedImagesAlmostEqual(result_long.noise_warp0, result_original.noise_warp0, atol=6e-8)
        with self.assertRaises(AssertionError):
            self.assertMaskedImagesEqual(result_short.noise_warp0, result_original.noise_warp0)


class MakeWarpNoGoodPixelsTestCase(MakeWarpTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.exposure.mask.array |= cls.exposure.mask.getPlaneBitMask("NO_DATA")

    def test_makeWarp(self):
        """Test MakeDirectWarpTask fails gracefully with no good pixels.

        It should return an empty exposure, with no PSF.
        """
        dataRef = InMemoryDatasetHandle(self.exposure.clone(), dataId=self.dataId)
        config = copy.copy(self.config)

        makeWarp = MakeDirectWarpTask(config=config)
        warp_detector_inputs = {
            dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=dataRef,
                data_id=dataRef.dataId,
            )
        }
        result = makeWarp.run(warp_detector_inputs, sky_info=copy.deepcopy(self.skyInfo), visit_summary=None)

        # Ensure we got None
        self.assertIsNone(result.warp)
        self.assertIsNone(result.masked_fraction_warp)
        self.assertIsNone(result.noise_warp0)

    def test_compare_warps(self):
        """This test is not applicable when there are no good pixels."""

    def test_long_data_ids(self):
        """This test is not applicable when there are no good pixels."""

    def test_flat_background_ratio(self):
        """This test is not applicable when there are no good pixels."""


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
