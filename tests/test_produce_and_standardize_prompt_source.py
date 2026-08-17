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

"""End-to-end test of the prompt_source production chain.

This exercises the sequence of tasks that ApPipe.yaml runs to turn a
calibrated visit image into the ``prompt_source`` table:

1. ``SingleFrameDetectAndMeasureTask`` detects and measures
   sources on a synthetic exposure, producing the
   ``single_visit_star_reprocessed_unstandardized`` catalog.
2. ``TransformSourceTableTask`` applies the
    ``prompt_source.yaml`` functors from ``pipe_tasks``.
3. ``ConsolidateSourceTableTask`` concatenates the per-detector tables.
4. ``SplitPrimaryTask`` keeps the primary rows and drops sky sources.
"""

import os
import unittest

import numpy as np

import lsst.afw.geom
import lsst.afw.image
import lsst.geom
import lsst.meas.algorithms
import lsst.meas.base.tests
import lsst.utils.tests
from lsst.drp.tasks.single_frame_detect_and_measure import (
    SingleFrameDetectAndMeasureConfig,
    SingleFrameDetectAndMeasureTask,
)
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.postprocess import (
    TableVStack,
    TransformSourceTableConfig,
    TransformSourceTableTask,
)
from lsst.pipe.tasks.schemaUtils import (
    checkSdmSchemaColumns,
    column_dtype,
    readSdmSchemaFile,
)
from lsst.pipe.tasks.split_primary import SplitPrimaryTask
from lsst.utils import getPackageDir

FUNCTOR_FILE = os.path.join(getPackageDir("pipe_tasks"), "schemas", "prompt_source.yaml")
SCHEMA_FILE = os.path.join("${SDM_SCHEMAS_DIR}", "yml", "ap_extra.yaml")
TABLE_NAME = "PromptSource"
SPLIT_DISCARD_PRIMARY_COLUMNS = ["sky_source"]

VISIT = 98765
DETECTOR = 42


def make_exposure_and_background():
    """Build a synthetic calibrated exposure and its background model.

    Modeled on ``test_reprocess_visit_image.py``: a `TestDataset` provides the
    PSF, WCS and PhotoCalib, a handful of point sources are added, and a
    background is fit to the realized image.

    Returns
    -------
    exposure : `lsst.afw.image.ExposureF`
        Calibrated exposure ready for detection and measurement.
    background : `lsst.afw.math.BackgroundList`
        Background model for the exposure.
    """
    bbox = lsst.geom.Box2I(lsst.geom.Point2I(5, 4), lsst.geom.Point2I(205, 184))
    dataset = lsst.meas.base.tests.TestDataset(
        bbox,
        crval=lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees),
        calibration=12.3,
        detector=DETECTOR,
        visitId=VISIT,
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
    truth_exposure, _ = dataset.realize(noise=noise, schema=dataset.makeMinimalSchema())

    # Build an input exposure carrying the calibrations the task reads.
    exposure = lsst.afw.image.ExposureF(truth_exposure.maskedImage.clone())
    exposure.mask.clearMaskPlane(exposure.mask.getMaskPlane("DETECTED"))
    exposure.setPsf(truth_exposure.psf)
    exposure.setWcs(truth_exposure.wcs)
    exposure.setPhotoCalib(truth_exposure.photoCalib)
    exposure.info.setApCorrMap(lsst.afw.image.ApCorrMap())
    exposure.info.setVisitInfo(truth_exposure.visitInfo)
    exposure.info.setDetector(truth_exposure.getDetector())

    bkgConfig = lsst.meas.algorithms.SubtractBackgroundTask.ConfigClass()
    # Small test image; fit a simple background model.
    bkgConfig.approxOrderX = 1
    bkgTask = lsst.meas.algorithms.SubtractBackgroundTask(config=bkgConfig)
    background = bkgTask.run(truth_exposure).background

    return exposure, background


class PromptSourceEndToEndTestCase(lsst.utils.tests.TestCase):
    """Run SingleFrameDetectAndMeasure output through the prompt_source steps
    and validate the final schema.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        exposure, background = make_exposure_and_background()

        # 1. Detect and measure direct sources.
        sfdmConfig = SingleFrameDetectAndMeasureConfig()
        sfdmConfig.detection.background.approxOrderX = 1
        sfdmConfig.sky_sources.nSources = 2
        sfdmTask = SingleFrameDetectAndMeasureTask(config=sfdmConfig)
        cls.sources = sfdmTask.run(exposure=exposure, input_background=background).sources

        schemaFile = os.path.expandvars(SCHEMA_FILE)
        cls.schema = readSdmSchemaFile(schemaFile)

        # 2. Run standardization
        transformConfig = TransformSourceTableConfig()
        transformConfig.functorFile = FUNCTOR_FILE
        transformTask = TransformSourceTableTask(config=transformConfig)
        detectorTable = transformTask.run(
            handle=InMemoryDatasetHandle(cls.sources.to_pandas(), storageClass="DataFrame"),
            funcs=transformTask.funcs,
            dataId={"visit": VISIT, "detector": DETECTOR, "band": "r", "physical_filter": "r_03"},
        ).outputCatalog

        # 3. Mimic the stacking done by the consolidate task
        consolidated = TableVStack.vstack_handles(
            [InMemoryDatasetHandle(detectorTable, storageClass="ArrowAstropy")]
        )

        # 4. Drop non-primary rows from blends, and sky sources
        splitConfig = SplitPrimaryTask.ConfigClass()
        splitConfig.discard_primary_columns = SPLIT_DISCARD_PRIMARY_COLUMNS
        splitTask = SplitPrimaryTask(config=splitConfig)
        cls.promptSource = splitTask.run(full=consolidated).primary

    def testProducesPrimarySources(self):
        """prompt_source holds the primary rows minus the dropped columns."""
        nPrimary = int(np.sum(self.sources["detect_isPrimary"]))
        self.assertGreater(nPrimary, 0)
        self.assertEqual(len(self.promptSource), nPrimary)
        self.assertNotIn("detect_isPrimary", self.promptSource.colnames)
        self.assertNotIn("sky_source", self.promptSource.colnames)

    def testCalibratedFluxesComputed(self):
        """The calibrated flux functors ran and produced finite values."""
        for column in ("psfFlux", "calibFlux"):
            self.assertIn(column, self.promptSource.colnames)
            self.assertTrue(np.isfinite(self.promptSource[column]).any())

    def testSchema(self):
        """Check that prompt_source column names and dtypes match the
        PromptSource schema.
        """
        dataframe = self.promptSource.to_pandas()
        # There should be no columns beyond those defined for PromptSource.
        extra = checkSdmSchemaColumns(self.schema, list(dataframe.columns), TABLE_NAME)
        self.assertEqual(extra, [], f"prompt_source has columns absent from {TABLE_NAME}: {extra}")

        # Every schema column is present with the expected dtype.
        mismatches = {}
        for columnDef in self.schema[TABLE_NAME].columns:
            # Extract the expected datatype for each column from the schema
            expected = column_dtype(columnDef.datatype)
            if columnDef.name not in dataframe.columns:
                mismatches[columnDef.name] = f"missing (expected {expected})"
            elif str(dataframe[columnDef.name].dtype) != expected:
                mismatches[columnDef.name] = f"{dataframe[columnDef.name].dtype} != {expected}"
        self.assertEqual(mismatches, {}, f"prompt_source dtypes do not match {TABLE_NAME}: {mismatches}")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
