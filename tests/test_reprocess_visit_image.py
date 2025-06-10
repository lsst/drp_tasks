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

import tempfile
import unittest

import astropy.table
import numpy as np

import lsst.afw.image
import lsst.afw.math
import lsst.afw.table
import lsst.daf.butler.tests as butlerTests
import lsst.geom
import lsst.meas.algorithms
import lsst.meas.base.tests
import lsst.pipe.base.testUtils
import lsst.utils.tests
from lsst.drp.tasks.reprocess_visit_image import ReprocessVisitImageTask


def make_visit_summary(summary=None, psf=None, wcs=None, photo_calib=None, detector=42):
    """Return a visit summary table with an entry for the given detector."""
    if summary is None:
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        lsst.afw.image.ExposureSummaryStats.update_schema(schema)
        summary = lsst.afw.table.ExposureCatalog(schema)
    if summary.find(detector) is not None:
        raise RuntimeError(f"Detector {detector} already exists in visit summary table, can't re-add it.")

    record = summary.addNew()
    record.setId(detector)
    record.setApCorrMap(lsst.afw.image.ApCorrMap()),

    record.setPsf(psf)
    record.setPhotoCalib(photo_calib)

    if wcs is not None:
        record.setWcs(wcs)
    else:
        crpix = lsst.geom.Box2D(
            lsst.geom.Box2D(lsst.geom.Point2D(0, 0), lsst.geom.Point2D(100, 100))
        ).getCenter()
        crval = lsst.geom.SpherePoint(45.0, 45.0, lsst.geom.degrees)
        cdelt = 0.2 * lsst.geom.arcseconds
        wcs = lsst.afw.geom.makeSkyWcs(
            crpix=crpix, crval=crval, cdMatrix=lsst.afw.geom.makeCdMatrix(scale=cdelt)
        )
        record.setWcs(wcs)

    lsst.afw.image.ExposureSummaryStats().update_record(record),

    return summary


class ReprocessVisitImageTaskTests(lsst.utils.tests.TestCase):
    def setUp(self):
        # Different x/y dimensions so they're easy to distinguish in a plot,
        # and non-zero minimum, to help catch xy0 errors.
        detector = 42
        bbox = lsst.geom.Box2I(lsst.geom.Point2I(5, 4), lsst.geom.Point2I(205, 184))
        self.sky_center = lsst.geom.SpherePoint(245.0, -45.0, lsst.geom.degrees)
        self.photo_calib = 12.3
        dataset = lsst.meas.base.tests.TestDataset(
            bbox,
            crval=self.sky_center,
            calibration=self.photo_calib,
            detector=detector,
            # Force a large visitId, to test DM-49138.
            visitId=2**33,
        )
        # sqrt of area of a normalized 2d gaussian
        psf_scale = np.sqrt(4 * np.pi * (dataset.psfShape.getDeterminantRadius()) ** 2)
        noise = 10.0  # stddev of noise per pixel
        # Sources ordered from faintest to brightest.
        self.fluxes = np.array(
            (
                6 * noise * psf_scale,
                12 * noise * psf_scale,
                45 * noise * psf_scale,
                150 * noise * psf_scale,
                400 * noise * psf_scale,
                1000 * noise * psf_scale,
            )
        )
        self.centroids = np.array(
            ((162, 25), (40, 70), (100, 160), (50, 120), (92, 35), (175, 154)), dtype=np.float32
        )
        for flux, centroid in zip(self.fluxes, self.centroids):
            dataset.addSource(instFlux=flux, centroid=lsst.geom.Point2D(centroid[0], centroid[1]))

        # Bright extended source in the center of the image: should not appear
        # in any of the output catalogs.
        center = lsst.geom.Point2D(100, 100)
        shape = lsst.afw.geom.Quadrupole(8, 9, 3)
        dataset.addSource(instFlux=500 * noise * psf_scale, centroid=center, shape=shape)

        schema = dataset.makeMinimalSchema()
        self.truth_exposure, self.truth_cat = dataset.realize(noise=noise, schema=schema)
        # Add a cosmic ray-like two hot pixels, to check CR removal.
        self.truth_exposure.image[60, 60] = 10000
        self.truth_exposure.image[60, 61] = 10000

        # Make an exposure that looks like a PostISRCCD, to serve as the input.
        self.exposure = lsst.afw.image.ExposureF()
        self.exposure.maskedImage = self.truth_exposure.maskedImage.clone()
        self.exposure.mask.clearMaskPlane(self.exposure.mask.getMaskPlane("DETECTED"))
        self.exposure.mask.clearMaskPlane(self.exposure.mask.getMaskPlane("DETECTED_NEGATIVE"))
        # PostISRCCD will have a VisitInfo and Detector attached.
        self.exposure.info.setVisitInfo(self.truth_exposure.visitInfo)
        self.exposure.info.setDetector(self.truth_exposure.getDetector())

        # Subtract a background from the truth exposure, so that the input
        # exposure still has the background in it.
        config = lsst.meas.algorithms.SubtractBackgroundTask.ConfigClass()
        # Don't really have a background, so have to fit simpler models.
        config.approxOrderX = 1
        task = lsst.meas.algorithms.SubtractBackgroundTask(config=config)
        self.background = task.run(self.truth_exposure).background
        self.visit_summary = make_visit_summary(
            psf=self.truth_exposure.psf,
            wcs=self.truth_exposure.wcs,
            photo_calib=self.truth_exposure.photoCalib,
        )

        # A catalog that looks like the output of finalizeCharacterization,
        # with a value set that we can test on the output.
        self.visit_catalog = self.truth_cat.asAstropy()
        self.visit_catalog.add_column(
            astropy.table.Column(data=np.zeros(len(self.visit_catalog)), name="calib_psf_used", dtype=bool)
        )
        self.visit_catalog.add_column(
            astropy.table.Column(
                data=np.zeros(len(self.visit_catalog)), name="calib_psf_reserved", dtype=bool
            )
        )
        self.visit_catalog.add_column(
            astropy.table.Column(
                data=np.zeros(len(self.visit_catalog)), name="calib_psf_candidate", dtype=bool
            )
        )
        self.visit_catalog.add_column(
            astropy.table.Column(data=[detector] * len(self.visit_catalog), name="detector")
        )
        # Marking faintest source, so it's easy to identify later.
        self.visit_catalog["calib_psf_used"][0] = True

        # Test-specific configuration:
        self.config = ReprocessVisitImageTask.ConfigClass()
        # Don't really have a background, so have to fit simpler models.
        self.config.detection.background.approxOrderX = 1
        # Only insert 2 sky sources, for simplicity.
        self.config.sky_sources.nSources = 2

        # Make a realistic id generator so that output catalog ids are useful.
        # NOTE: The id generator is used to seed the noise replacer during
        # measurement, so changes to values here can have subtle effects on
        # the centroids and fluxes measured on the image, which might cause
        # tests to fail.
        data_id = lsst.daf.butler.DataCoordinate.standardize(
            instrument="I",
            visit=self.truth_exposure.visitInfo.id,
            detector=12,
            universe=lsst.daf.butler.DimensionUniverse(),
        )
        self.config.id_generator.packer.name = "observation"
        # Without the LSSTCam-specific visitId handler, we have to use a large
        # n_observation to fit visitId=2^33.
        self.config.id_generator.packer["observation"].n_observations = 2**35
        self.config.id_generator.packer["observation"].n_detectors = 99
        self.config.id_generator.n_releases = 8
        self.config.id_generator.release_id = 2
        self.id_generator = self.config.id_generator.apply(data_id)

    def test_run(self):

        background_to_photometric_ratio_value = 1.1

        for do_apply_flat_background_ratio in [False, True]:
            config = self.config
            config.do_apply_flat_background_ratio = do_apply_flat_background_ratio

            if do_apply_flat_background_ratio:
                background_to_photometric_ratio = self.exposure.image.clone()
                background_to_photometric_ratio.array[:, :] = background_to_photometric_ratio_value
            else:
                background_to_photometric_ratio = None

            task = ReprocessVisitImageTask(config=config)
            result = task.run(
                exposures=self.exposure.clone(),
                initial_photo_calib=self.truth_exposure.photoCalib,
                psf=self.truth_exposure.psf,
                background=self.background,
                ap_corr=lsst.afw.image.ApCorrMap(),
                photo_calib=self.truth_exposure.photoCalib,
                wcs=self.truth_exposure.wcs,
                calib_sources=self.visit_catalog,
                id_generator=self.id_generator,
                background_to_photometric_ratio=background_to_photometric_ratio,
            )

            calibrated = result.exposure.photoCalib.calibrateImage(result.exposure.maskedImage)
            self.assertImagesAlmostEqual(result.exposure.image, calibrated.image)
            self.assertImagesAlmostEqual(result.exposure.variance, calibrated.variance)
            self.assertEqual(result.exposure.psf, self.truth_exposure.psf)
            self.assertEqual(result.exposure.wcs, self.truth_exposure.wcs)
            # A calibrated exposure has PhotoCalib==1.
            self.assertNotEqual(result.exposure.photoCalib, self.truth_exposure.photoCalib)
            self.assertFloatsAlmostEqual(result.exposure.photoCalib.getCalibrationMean(), 1)

            # All sources (plus sky sources) should have been detected.
            self.assertEqual(len(result.sources), len(self.truth_cat) + self.config.sky_sources.nSources)
            # Faintest non-sky source should be marked as used.
            flux_sorted = result.sources[result.sources.argsort("slot_CalibFlux_instFlux")]
            self.assertTrue(flux_sorted[~flux_sorted["sky_source"]]["calib_psf_used"][0])
            # Test that the schema init-output agrees with the catalog output.
            self.assertEqual(task.sources_schema.schema, result.sources_footprints.schema)
            # The flux/instFlux ratio should be the LocalPhotoCalib value.
            for record in result.sources_footprints:
                self.assertAlmostEqual(
                    record["base_PsfFlux_flux"] / record["base_PsfFlux_instFlux"],
                    record["base_LocalPhotoCalib"],
                )

            if not do_apply_flat_background_ratio:
                result_false = result

        self.assertFloatsAlmostEqual(
            result_false.sources_footprints["base_PsfFlux_instFlux"]
            / result.sources_footprints["base_PsfFlux_instFlux"],
            background_to_photometric_ratio_value,
            rtol=1e-6,
        )

        self.assertFloatsAlmostEqual(
            result_false.sources_footprints["base_PsfFlux_flux"]
            / result.sources_footprints["base_PsfFlux_flux"],
            background_to_photometric_ratio_value,
            rtol=1e-6,
        )

        # Prior to DM-49138, LSSTCam-style >32-bit visitIds were silently
        # down-cast to `0`.
        self.assertTrue((result.sources["visit"] == 2**33).all())

        self.assertEqual(result.exposure.metadata["BUNIT"], "nJy")


class ReprocessVisitImageTaskRunQuantumTests(lsst.utils.tests.TestCase):
    """Tests of ``ReprocessVisitImageTask.runQuantum``, which need a test
    butler, but do not need real data.
    """

    def setUp(self):
        instrument = "testCam"
        exposure0 = 101
        exposure1 = 102
        visit = 100101
        detector = 42

        # Create a and populate a test butler for runQuantum tests.
        self.repo_path = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = butlerTests.makeTestRepo(self.repo_path.name)

        # A complete instrument record is necessary for the id generator.
        instrumentRecord = self.repo.dimensions["instrument"].RecordClass(
            name=instrument,
            visit_max=1e6,
            exposure_max=1e6,
            detector_max=128,
            class_name="lsst.obs.base.instrument_tests.DummyCam",
        )
        self.repo.registry.syncDimensionData("instrument", instrumentRecord)

        # dataIds for fake data
        butlerTests.addDataIdValue(self.repo, "detector", detector)
        butlerTests.addDataIdValue(self.repo, "detector", detector + 1)
        butlerTests.addDataIdValue(self.repo, "detector", detector + 2)
        butlerTests.addDataIdValue(self.repo, "exposure", exposure0)
        butlerTests.addDataIdValue(self.repo, "exposure", exposure1)
        butlerTests.addDataIdValue(self.repo, "visit", visit)

        # inputs
        butlerTests.addDatasetType(
            self.repo, "postISRCCD", {"instrument", "exposure", "detector"}, "ExposureF"
        )
        butlerTests.addDatasetType(self.repo, "finalVisitSummary", {"instrument", "visit"}, "ExposureCatalog")
        butlerTests.addDatasetType(
            self.repo, "initial_pvi_background", {"instrument", "visit", "detector"}, "Background"
        )
        butlerTests.addDatasetType(
            self.repo, "initial_photoCalib_detector", {"instrument", "visit", "detector"}, "PhotoCalib"
        )
        butlerTests.addDatasetType(self.repo, "skyCorr", {"instrument", "visit", "detector"}, "Background")
        butlerTests.addDatasetType(self.repo, "finalized_src_table", {"instrument", "visit"}, "DataFrame")
        butlerTests.addDatasetType(
            self.repo, "background_to_photometric_ratio", {"instrument", "visit", "detector"}, "Image"
        )

        # outputs
        butlerTests.addDatasetType(
            self.repo, "source_schema", {"instrument", "visit", "detector"}, "SourceCatalog"
        )
        butlerTests.addDatasetType(self.repo, "pvi", {"instrument", "visit", "detector"}, "ExposureF")
        butlerTests.addDatasetType(
            self.repo, "sources_footprints_detector", {"instrument", "visit", "detector"}, "SourceCatalog"
        )
        butlerTests.addDatasetType(
            self.repo, "sources_detector", {"instrument", "visit", "detector"}, "ArrowAstropy"
        )
        butlerTests.addDatasetType(
            self.repo, "pvi_background", {"instrument", "visit", "detector"}, "Background"
        )

        # dataIds
        self.exposure0_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure0, "detector": detector}
        )
        self.exposure1_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "exposure": exposure1, "detector": detector}
        )
        self.visit_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": visit, "detector": detector}
        )
        # Second id for testing on a detector that is not in visitSummary.
        self.visit1_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": visit, "detector": detector + 1}
        )
        # Third id for testing on a detector that is in visitSummary but
        # has missing calibs.
        self.visit2_id = self.repo.registry.expandDataId(
            {"instrument": instrument, "visit": visit, "detector": detector + 2}
        )
        self.visit_only_id = self.repo.registry.expandDataId({"instrument": instrument, "visit": visit})

        # put empty data
        self.butler = butlerTests.makeTestCollection(self.repo)
        self.butler.put(lsst.afw.image.ExposureF(), "postISRCCD", self.exposure0_id)
        self.butler.put(lsst.afw.image.ExposureF(), "postISRCCD", self.exposure1_id)
        control = lsst.afw.math.BackgroundControl(10, 10)
        background = lsst.afw.math.makeBackground(lsst.afw.image.ImageF(100, 100), control)
        self.butler.put(lsst.afw.image.PhotoCalib(10), "initial_photoCalib_detector", self.visit_id)
        self.butler.put(lsst.afw.image.PhotoCalib(10), "initial_photoCalib_detector", self.visit1_id)
        self.butler.put(lsst.afw.image.PhotoCalib(10), "initial_photoCalib_detector", self.visit2_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "initial_pvi_background", self.visit_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "initial_pvi_background", self.visit1_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "initial_pvi_background", self.visit2_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "skyCorr", self.visit_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "skyCorr", self.visit1_id)
        self.butler.put(lsst.afw.math.BackgroundList(background), "skyCorr", self.visit2_id)
        self.butler.put(lsst.afw.table.SourceCatalog().asAstropy(), "finalized_src_table", self.visit_only_id)
        self.butler.put(lsst.afw.image.ImageF(), "background_to_photometric_ratio", self.visit_id)
        self.butler.put(lsst.afw.image.ImageF(), "background_to_photometric_ratio", self.visit1_id)
        self.butler.put(lsst.afw.image.ImageF(), "background_to_photometric_ratio", self.visit2_id)
        # Make a simple single gaussian psf so that psf is not None in
        # finalVisitSummary table which would result in
        # UpstreamFailureNoWorkFound being raised in ReprocessVisitImageTask,
        # which will be independently tested with self.visit2_id.
        simple_psf = lsst.meas.algorithms.SingleGaussianPsf(11, 11, 2.0)
        photo_calib = lsst.afw.image.PhotoCalib(10, 0.5)
        visit_summary = make_visit_summary(psf=simple_psf, photo_calib=photo_calib, detector=detector)
        # Add a detector with an entry, but some required calibs set to None.
        visit_summary = make_visit_summary(
            summary=visit_summary, psf=None, photo_calib=None, detector=detector + 2
        )
        self.butler.put(visit_summary, "finalVisitSummary", self.visit_only_id)

    def tearDown(self):
        self.repo_path.cleanup()

    def test_lintConnections(self):
        """Check that the connections are self-consistent."""
        Connections = ReprocessVisitImageTask.ConfigClass.ConnectionsClass
        lsst.pipe.base.testUtils.lintConnections(Connections)

    def test_runQuantum(self):
        task = ReprocessVisitImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.visit_id,
            {
                "exposures": [self.exposure0_id],
                "visit_summary": self.visit_only_id,
                "initial_photo_calib": self.visit_id,
                "background_1": self.visit_id,
                "background_2": self.visit_id,
                "calib_sources": self.visit_only_id,
                "background_to_photometric_ratio": self.visit_id,
                # outputs
                "exposure": self.visit_id,
                "sources": self.visit_id,
                "sources_footprints": self.visit_id,
                "background": self.visit_id,
            },
        )
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)
        # Check that the proper kwargs are passed to run().
        self.assertEqual(
            mock_run.call_args.kwargs.keys(),
            {
                "exposures",
                "initial_photo_calib",
                "psf",
                "background",
                "ap_corr",
                "photo_calib",
                "wcs",
                "calib_sources",
                "id_generator",
                "background_to_photometric_ratio",
                "result",
            },
        )

    def test_runQuantum_no_detector_in_visit_summary(self):
        """Test how the task handles the detector not being in the input visit
        summary.
        """
        task = ReprocessVisitImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.visit1_id,
            {
                "exposures": [self.exposure0_id],
                "visit_summary": self.visit_only_id,
                "initial_photo_calib": self.visit1_id,
                "background_1": self.visit1_id,
                "background_2": self.visit1_id,
                "calib_sources": self.visit_only_id,
                "background_to_photometric_ratio": self.visit_id,
                # outputs
                "exposure": self.visit1_id,
                "sources": self.visit1_id,
                "sources_footprints": self.visit1_id,
                "background": self.visit1_id,
            },
        )
        msg = "  > no entry for the detector was found in the visit summary table"
        with self.assertRaisesRegex(
            lsst.pipe.base.UpstreamFailureNoWorkFound, f"Skipping reprocessing of detector 43 because:\n{msg}"
        ):
            lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)

    def test_runQuantum_missing_calibs_for_detector_in_visit_summary(self):
        """Test how the task handles the detector not being in the input visit
        summary.
        """
        task = ReprocessVisitImageTask()
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.visit2_id,
            {
                "exposures": [self.exposure0_id],
                "visit_summary": self.visit_only_id,
                "initial_photo_calib": self.visit2_id,
                "background_1": self.visit2_id,
                "background_2": self.visit2_id,
                "calib_sources": self.visit_only_id,
                "background_to_photometric_ratio": self.visit_id,
                # outputs
                "exposure": self.visit2_id,
                "sources": self.visit2_id,
                "sources_footprints": self.visit2_id,
                "background": self.visit2_id,
            },
        )
        lines = [
            "  > the PSF model for the detector is None",
            "  > the photometric calibration model for the detector is None",
        ]
        msg = "\n".join(lines)
        with self.assertRaisesRegex(
            lsst.pipe.base.UpstreamFailureNoWorkFound, f"Skipping reprocessing of detector 44 because:\n{msg}"
        ):
            lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)

    def test_runQuantum_no_sky_corr(self):
        """Test that the task will run if using the sky_corr input is
        diabled.
        """
        config = ReprocessVisitImageTask.ConfigClass()
        config.do_use_sky_corr = False
        task = ReprocessVisitImageTask(config=config)
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.visit_id,
            {
                "exposures": [self.exposure0_id],
                "visit_summary": self.visit_only_id,
                "initial_photo_calib": self.visit_id,
                "background_1": self.visit_id,
                "calib_sources": self.visit_only_id,
                "background_to_photometric_ratio": self.visit_id,
                # outputs
                "exposure": self.visit_id,
                "sources": self.visit_id,
                "sources_footprints": self.visit_id,
                "background": self.visit_id,
            },
        )
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)
        # Check that the proper kwargs are passed to run().
        self.assertEqual(
            mock_run.call_args.kwargs.keys(),
            {
                "exposures",
                "initial_photo_calib",
                "psf",
                "background",
                "ap_corr",
                "photo_calib",
                "wcs",
                "calib_sources",
                "id_generator",
                "background_to_photometric_ratio",
                "result",
            },
        )

    def test_runQuantum_illumination_correction(self):
        """Test the task with illumination correction enabled."""
        config = ReprocessVisitImageTask.ConfigClass()
        config.do_apply_flat_background_ratio = True
        task = ReprocessVisitImageTask(config=config)
        lsst.pipe.base.testUtils.assertValidInitOutput(task)

        quantum = lsst.pipe.base.testUtils.makeQuantum(
            task,
            self.butler,
            self.visit_id,
            {
                "exposures": [self.exposure0_id],
                "visit_summary": self.visit_only_id,
                "initial_photo_calib": self.visit_id,
                "background_1": self.visit_id,
                "background_2": self.visit_id,
                "calib_sources": self.visit_only_id,
                "background_to_photometric_ratio": self.visit_id,
                # outputs
                "exposure": self.visit_id,
                "sources": self.visit_id,
                "sources_footprints": self.visit_id,
                "background": self.visit_id,
            },
        )
        mock_run = lsst.pipe.base.testUtils.runTestQuantum(task, self.butler, quantum)
        # Check that the proper kwargs are passed to run().
        self.assertEqual(
            mock_run.call_args.kwargs.keys(),
            {
                "exposures",
                "initial_photo_calib",
                "psf",
                "background",
                "ap_corr",
                "photo_calib",
                "wcs",
                "calib_sources",
                "id_generator",
                "background_to_photometric_ratio",
                "result",
            },
        )


def setup_module(module):
    lsst.utils.tests.init()


class MemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
