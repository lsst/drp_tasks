# This file is part of pipe_tasks.
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

from typing import Self, Type

import unittest

import numpy as np

import lsst.utils.tests

import lsst.afw.image
import lsst.afw.math as afwMath
from lsst.daf.butler import DataCoordinate, DimensionUniverse
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.make_direct_warp import (MakeDirectWarpConfig, MakeDirectWarpTask, WarpDetectorInputs)
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.skymap as skyMap
from lsst.afw.detection import GaussianPsf
import lsst.afw.cameraGeom.testUtils


class MakeWarpTestCase(lsst.utils.tests.TestCase):
    def setUp(self):
        np.random.seed(12345)

        self.config = MakeDirectWarpConfig()
        self.config.useVisitSummaryPsf = False
        self.config.doSelectPreWarp = False
        self.config.doWarpMaskedFraction = True
        self.config.numberOfNoiseRealizations = 1

        meanCalibration = 1e-4
        calibrationErr = 1e-5
        self.exposurePhotoCalib = lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr)
        # An external photoCalib calibration to return
        self.externalPhotoCalib = lsst.afw.image.PhotoCalib(1e-6, 1e-8)

        crpix = lsst.geom.Point2D(0, 0)
        crval = lsst.geom.SpherePoint(0, 45, lsst.geom.degrees)
        cdMatrix = lsst.afw.geom.makeCdMatrix(scale=1.0*lsst.geom.arcseconds)
        self.skyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, cdMatrix)
        externalCdMatrix = lsst.afw.geom.makeCdMatrix(scale=0.9*lsst.geom.arcseconds)
        # An external skyWcs to return
        self.externalSkyWcs = lsst.afw.geom.makeSkyWcs(crpix, crval, externalCdMatrix)

        self.exposure = lsst.afw.image.ExposureF(100, 150)
        self.exposure.maskedImage.image.array = np.random.random((150, 100)).astype(np.float32) * 1000
        self.exposure.maskedImage.variance.array = np.random.random((150, 100)).astype(np.float32)
        # mask at least one pixel
        self.exposure.maskedImage.mask[5, 5] = 3
        # set the PhotoCalib and Wcs objects of this exposure.
        self.exposure.setPhotoCalib(lsst.afw.image.PhotoCalib(meanCalibration, calibrationErr))
        self.exposure.setWcs(self.skyWcs)
        self.exposure.setPsf(GaussianPsf(5, 5, 2.5))
        self.exposure.setFilter(lsst.afw.image.FilterLabel(physical="fakeFilter", band="fake"))

        self.visit = 100
        self.detector = 5
        detectorName = f"detector {self.detector}"
        detector = lsst.afw.cameraGeom.testUtils.DetectorWrapper(name=detectorName, id=self.detector).detector
        self.exposure.setDetector(detector)

        dataId_dict = {"detector_id": self.detector, "visit_id": 1248, "band": "i"}
        dataId = self.generate_data_id(**dataId_dict)
        self.dataRef = InMemoryDatasetHandle(self.exposure, dataId=dataId)
        simpleMapConfig = skyMap.discreteSkyMap.DiscreteSkyMapConfig()
        simpleMapConfig.raList = [crval.getRa().asDegrees()]
        simpleMapConfig.decList = [crval.getDec().asDegrees()]
        simpleMapConfig.radiusList = [0.1]

        self.simpleMap = skyMap.DiscreteSkyMap(simpleMapConfig)
        self.tractId = 0
        self.patchId = self.simpleMap[0].findPatch(crval).sequential_index
        self.skyInfo = makeSkyInfo(self.simpleMap, self.tractId, self.patchId)

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
        visit_max: int = 10000
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
            skymap="test_skymap", tract=tract, patch=patch,
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
        makeWarp = MakeDirectWarpTask(config=self.config)
        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
            )
        }
        result = makeWarp.run(
            warp_detector_inputs,
            sky_info=self.skyInfo,
            visit_summary=None
        )

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
        backgroundList = self.make_backgroundList()

        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
                background_apply=backgroundList if doApplyNewBackground else None,
                background_revert=backgroundList if doRevertOldBackground else None,

            )
        }

        self.config.numberOfNoiseRealizations = 1
        self.config.doApplyNewBackground = doApplyNewBackground
        self.config.doRevertOldBackground = doRevertOldBackground

        makeWarp = MakeDirectWarpTask(config=self.config)
        makeWarp.run(
            warp_detector_inputs,
            sky_info=self.skyInfo,
            visit_summary=None
        )

    def test_background_errors(self):
        """Test that MakeDirectWarpTask raises errors when backgrounds are not
        set correctly.
        """
        backgroundList = self.make_backgroundList()
        self.config.numberOfNoiseRealizations = 1

        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
                background_apply=backgroundList,
            )
        }
        makeWarp = MakeDirectWarpTask(config=self.config)
        with self.assertRaises(RuntimeError, msg="doApplyNewBackground is False, but"):
            makeWarp.run(
                warp_detector_inputs,
                sky_info=self.skyInfo,
                visit_summary=None
            )

        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
                background_apply=None,
            )
        }
        self.config.doApplyNewBackground = True
        makeWarp = MakeDirectWarpTask(config=self.config)
        with self.assertRaises(RuntimeError, msg="No background to apply"):
            makeWarp.run(
                warp_detector_inputs,
                sky_info=self.skyInfo,
                visit_summary=None
            )

        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
                background_apply=backgroundList,
                background_revert=backgroundList,
            )
        }
        makeWarp = MakeDirectWarpTask(config=self.config)
        with self.assertRaises(RuntimeError, msg="doRevertOldBackground is False, but"):
            makeWarp.run(
                warp_detector_inputs,
                sky_info=self.skyInfo,
                visit_summary=None
            )

        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef,
                data_id=self.dataRef.dataId,
                background_apply=backgroundList,
                background_revert=None,
            )
        }
        self.config.doRevertOldBackground = True
        makeWarp = MakeDirectWarpTask(config=self.config)
        with self.assertRaises(RuntimeError, msg="No background to revert"):
            makeWarp.run(
                warp_detector_inputs,
                sky_info=self.skyInfo,
                visit_summary=None
            )


class MakeWarpNoGoodPixelsTestCase(MakeWarpTestCase):
    def setUp(self):
        super().setUp()
        self.exposure.mask.array |= self.exposure.mask.getPlaneBitMask("NO_DATA")
        self.dataRef = InMemoryDatasetHandle(self.exposure, dataId=self.dataRef.dataId)

    def test_makeWarp(self):
        """Test MakeDirectWarpTask fails gracefully with no good pixels.

        It should return an empty exposure, with no PSF.
        """
        makeWarp = MakeDirectWarpTask(config=self.config)
        warp_detector_inputs = {
            self.dataRef.dataId.detector.id: WarpDetectorInputs(
                exposure_or_handle=self.dataRef, data_id=self.dataRef.dataId
            )
        }
        result = makeWarp.run(
            warp_detector_inputs,
            sky_info=self.skyInfo,
            visit_summary=None
        )

        # Ensure we got None
        self.assertIsNone(result.warp)
        self.assertIsNone(result.masked_fraction_warp)
        self.assertIsNone(result.noise_warp0)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
