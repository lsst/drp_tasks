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
from typing import TYPE_CHECKING, Iterable

import numpy as np
from assemble_coadd_test_utils import MockCoaddTestData, makeMockSkyInfo

import lsst.pipe.base as pipeBase
import lsst.utils.tests
from lsst.drp.tasks.assemble_cell_coadd import AssembleCellCoaddConfig, AssembleCellCoaddTask

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

    def runQuantum(self, mockSkyInfo, warpRefList, visitSummaryList=None):
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

        self.common = pipeBase.Struct(
            units=None,
            wcs=mockSkyInfo.wcs,
            band="i",
            identifiers=pipeBase.Struct(skymap=None, tract=0, patch=42, band="i"),
        )

        retStruct = self.run(
            warpRefList,
            mockSkyInfo,
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
        patch = 42
        tract = 0
        testData = MockCoaddTestData(fluxRange=1e4)
        exposures = {}
        matchedExposures = {}
        for expId in range(100, 110):
            exposures[expId], matchedExposures[expId] = testData.makeTestImage(expId)
        cls.handleList = testData.makeDataRefList(
            exposures, matchedExposures, "direct", patch=patch, tract=tract
        )
        cls.visitSummaryList = [
            testData.makeVisitSummaryTableHandle(warpHandle) for warpHandle in cls.handleList
        ]
        cls.skyInfo = makeMockSkyInfo(testData.bbox, testData.wcs, patch=patch)

    def tearDown(self) -> None:
        del self.result

    def runTask(self, config=None) -> None:
        if config is None:
            config = MockAssembleCellCoaddConfig()
        assembleTask = MockAssembleCellCoaddTask(config=config)
        self.result = assembleTask.runQuantum(self.skyInfo, self.handleList, self.visitSummaryList)

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
            with self.subTest(input_number=obsId):
                self.assertGreaterEqual(obsId.visit, visit)
            if visit == obsId.visit:
                with self.subTest(detector_number=obsId.detector):
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
            with self.subTest(x=cellId.x, y=cellId.y):
                self.assertLessEqual(singleCellCoadd.visit_count, max_visit_count)
            # Check that the inputs are sorted.
            self.checkSortOrder(singleCellCoadd.inputs)

    def test_assemble_basic(self):
        """Test that AssembleCellCoaddTask runs successfully without errors.

        This test does not check the correctness of the coaddition algorithms.
        This is intended to prevent the code from bit rotting.
        """
        self.runTask()
        # Check that we produced an exposure.
        self.assertTrue(self.result.multipleCellCoadd is not None)

    def test_visit_count(self):
        """Check that the visit_count method returns a number less than or
        equal to the total number of input exposures available.
        """
        self.runTask()
        max_visit_count = len(self.handleList)
        for cellId, singleCellCoadd in self.result.multipleCellCoadd.cells.items():
            with self.subTest(x=cellId.x, y=cellId.y):
                self.assertLessEqual(singleCellCoadd.visit_count, max_visit_count)

    def test_inputs_sorted(self):
        """Check that the inputs are sorted.

        The ordering is that inputs are sorted first by visit, and within the
        same visit, they are ordered by detector.
        """
        self.runTask()
        for _, singleCellCoadd in self.result.multipleCellCoadd.cells.items():
            self.checkSortOrder(singleCellCoadd.inputs)


class MyMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
