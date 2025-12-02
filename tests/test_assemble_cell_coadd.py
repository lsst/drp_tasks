# This file is part of drp_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

from __future__ import annotations

import unittest
import warnings
from typing import TYPE_CHECKING, Iterable

import numpy as np
from assemble_coadd_test_utils import MockCoaddTestData, makeMockSkyInfo

import lsst.afw.image as afwImage
import lsst.pipe.base as pipeBase
import lsst.utils.tests
from lsst.cell_coadds import CommonComponents
from lsst.drp.tasks.assemble_cell_coadd import (
    AssembleCellCoaddConfig,
    AssembleCellCoaddTask,
    WarpInputs,
)

if TYPE_CHECKING:
    from lsst.cell_coadds import ObservationIdentifiers

__all__ = (
    "MockAssembleCellCoaddConfig",
    "MockAssembleCellCoaddTask",
)


class MockAssembleCellCoaddConfig(AssembleCellCoaddConfig):
    pass


class MockAssembleCellCoaddTask(AssembleCellCoaddTask):
    """Lightly modified version of `AssembleCellCoaddTask` for unit tests.

    The modifications bypass the usual middleware for loading data and setting
    up the Task, and instead supply in-memory mock data references to the `run`
    method so that the coaddition algorithms can be tested without a Butler.
    """

    ConfigClass = MockAssembleCellCoaddConfig

    def runQuantum(
        self,
        mockSkyInfo,
        warpRefList,
        maskedFractionRefList,
        noise0RefList,
        visitSummaryList=None,
    ):
        """Modified interface for testing coaddition algorithms without a
        Butler.

        Parameters
        ----------
        mockSkyInfo : `lsst.pipe.base.Struct`
            A simple container that supplies a bounding box and WCS in the
            same format as the output of
            `lsst.pipe.tasks.CoaddBaseTask.getSkyInfo`
        warpRefList : `list` of `lsst.pipe.tasks.MockExposureReference`
            Data references to the test exposures that will be coadded,
            using the Gen 3 API.

        Returns
        -------
        retStruct : `lsst.pipe.base.Struct`
            The coadded exposure and associated metadata.
        """

        self.common = CommonComponents(
            units=None,
            wcs=mockSkyInfo.wcs,
            band="i",
            identifiers=pipeBase.Struct(skymap=None, tract=0, patch=42, band="i"),
        )

        inputs = {}
        for warpInput, maskedFractionInput, noise0Input in zip(
            warpRefList,
            maskedFractionRefList,
            noise0RefList,
        ):
            inputs[warpInput.dataId] = WarpInputs(
                warp=warpInput,
                masked_fraction=maskedFractionInput,
                noise_warps=[noise0Input],
            )

        retStruct = self.run(
            inputs=inputs,
            skyInfo=mockSkyInfo,
            visitSummaryList=visitSummaryList,
        )

        return retStruct


class AssembleCellCoaddTestCase(lsst.utils.tests.TestCase):
    """Tests of AssembleCellCoaddTask.

    These tests bypass the middleware used for accessing data and managing Task
    execution.
    """

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(42)
        rng = np.random.Generator(np.random.MT19937(42))
        patch = 42
        tract = 0
        testData = MockCoaddTestData(fluxRange=1e4)
        exposures = {}
        matchedExposures = {}
        masked_fraction_images = {}
        noise0_masked_images = {}
        for expId in range(100, 110):
            exposures[expId], matchedExposures[expId] = testData.makeTestImage(expId)
            masked_fraction_images[expId] = afwImage.ImageF(bbox=exposures[expId].getBBox())
            masked_fraction_images[expId].array[:, :] = rng.random(masked_fraction_images[expId].array.shape)
            noise0_masked_images[expId] = afwImage.MaskedImageF(bbox=exposures[expId].getBBox())
            noise0_masked_images[expId].image.array[:, :] = rng.normal(
                0, 1, noise0_masked_images[expId].image.array.shape
            ) * (exposures[expId].variance.array**0.5)

        cls.handleList = testData.makeDataRefList(exposures, patch=patch, tract=tract)
        cls.maskedFractionRefList = testData.makeDataRefList(masked_fraction_images, patch=patch, tract=tract)
        cls.noise0RefList = testData.makeDataRefList(noise0_masked_images, patch=patch, tract=tract)
        cls.visitSummaryList = [
            testData.makeVisitSummaryTableHandle(warpHandle) for warpHandle in cls.handleList
        ]
        cls.skyInfo = makeMockSkyInfo(testData.bbox, testData.wcs, patch=patch)

    def runTask(
        self,
        config=None,
        warpRefList=None,
        maskedFractionRefList=None,
        noise0RefList=None,
        visitSummaryList=None,
    ) -> None:
        if config is None:
            config = MockAssembleCellCoaddConfig()
        assembleTask = MockAssembleCellCoaddTask(config=config)
        if warpRefList is None:
            warpRefList = self.handleList
        if maskedFractionRefList is None:
            maskedFractionRefList = self.maskedFractionRefList
        if noise0RefList is None:
            noise0RefList = self.noise0RefList
        if visitSummaryList is None:
            visitSummaryList = self.visitSummaryList

        self.result = assembleTask.runQuantum(
            self.skyInfo,
            warpRefList=warpRefList,
            maskedFractionRefList=maskedFractionRefList,
            noise0RefList=noise0RefList,
            visitSummaryList=visitSummaryList,
        )

    def checkSortOrder(self, inputs: Iterable[ObservationIdentifiers]) -> None:
        """Check that the inputs are sorted.

        The inputs must be sorted first by visit, and within the same visit,
        by detector.

        Parameters
        ----------
        inputs : `Iterable` [`ObservationIdentifiers`]
            The inputs to be checked.
        """
        visit, detector = -np.inf, -np.inf  # Previous visit, detector IDs.
        for _, obsId in enumerate(inputs):
            with self.subTest(input_number=repr(obsId)):
                self.assertGreaterEqual(obsId.visit, visit)
            if visit == obsId.visit:
                with self.subTest(detector_number=repr(obsId.detector)):
                    self.assertGreaterEqual(obsId.detector, detector)

            visit, detector = obsId.visit, obsId.detector

    def checkRun(self, assembleTask):
        """Check that the task runs successfully."""
        result = assembleTask.runQuantum(self.skyInfo, self.handleList)

        # Check that we produced an exposure.
        self.assertTrue(result.multipleCellCoadd is not None)
        # Check that the visit_count method returns a number less than or equal
        # to the total number of input exposures available.
        max_visit_count = len(self.handleList)
        for cellId, singleCellCoadd in result.multipleCellCoadd.cells.items():
            with self.subTest(x=repr(cellId.x), y=repr(cellId.y)):
                self.assertLessEqual(singleCellCoadd.visit_count, max_visit_count)
            # Check that the aperture correction maps are not None.
            with self.subTest(x=repr(cellId.x), y=repr(cellId.y)):
                self.assertTrue(singleCellCoadd.aperture_correction_map is not None)
            # Check that the inputs are sorted.
            with self.subTest(x=repr(cellId.x), y=repr(cellId.y)):
                self.checkSortOrder(singleCellCoadd.inputs)

    def test_assemble_basic(self):
        """Test that AssembleCellCoaddTask runs successfully without errors.

        This test does not check the correctness of the coaddition algorithms.
        This is intended to prevent the code from bit rotting.
        """
        self.runTask()
        # Check that we produced an exposure.
        self.assertTrue(self.result.multipleCellCoadd is not None)

    def test_assemble_empty(self):
        """Test that AssembleCellCoaddTask runs successfully without errors
        when no input exposures are provided."""
        self.result = None  # so tearDown has something.
        with self.assertRaises(pipeBase.NoWorkFound, msg="No cells could be populated for the cell coadd."):
            self.runTask(warpRefList=[], maskedFractionRefList=[], noise0RefList=[], visitSummaryList=[])

    def test_assemble_without_visitSummary(self):
        """Test that AssembleCellCoaddTask calculates detector weights and
        runs successfully without errors when no visit summaries are provided.
        """
        self.runTask(visitSummaryList=[])
        # Check that we produced an exposure.
        self.assertTrue(self.result.multipleCellCoadd is not None)

    # TODO: Remove this test in DM-49401
    @lsst.utils.tests.methodParameters(do_scale_zero_point=[False, True])
    def test_do_scale_zero_point(self, do_scale_zero_point):
        config = MockAssembleCellCoaddConfig()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            config.do_scale_zero_point = do_scale_zero_point
            self.runTask(config)
        # Check that we produced an exposure.
        self.assertTrue(self.result.multipleCellCoadd is not None)

    @lsst.utils.tests.methodParameters(do_calculate_weight_from_warp=[False, True])
    def test_do_calculate_weight_from_warp(self, do_calculate_weight_from_warp):
        config = MockAssembleCellCoaddConfig()
        config.do_calculate_weight_from_warp = do_calculate_weight_from_warp
        self.runTask(config)
        # Check that we produced an exposure.
        self.assertTrue(self.result.multipleCellCoadd is not None)

    def test_visit_count(self):
        """Check that the visit_count method returns a number less than or
        equal to the total number of input exposures available.
        """
        self.runTask()
        max_visit_count = len(self.handleList)
        for cellId, singleCellCoadd in self.result.multipleCellCoadd.cells.items():
            with self.subTest(x=repr(cellId.x), y=repr(cellId.y)):
                self.assertLessEqual(singleCellCoadd.visit_count, max_visit_count)

    def test_inputs_sorted(self):
        """Check that the inputs are sorted.

        The ordering is that inputs are sorted first by visit, and within the
        same visit, they are ordered by detector.
        """
        self.runTask()
        for _, singleCellCoadd in self.result.multipleCellCoadd.cells.items():
            self.checkSortOrder(singleCellCoadd.inputs)

    def test_psf_normalization(self):
        """Check that the sum of PSF images is close to 1."""
        self.runTask()
        for cellId, singleCellCoadd in self.result.multipleCellCoadd.cells.items():
            with self.subTest(x=repr(cellId.x), y=repr(cellId.y)):
                self.assertFloatsAlmostEqual(singleCellCoadd.psf_image.array.sum(), 1.0, rtol=None, atol=1e-7)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
