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

__all__ = ["ReprocessVisitImageTask", "ReprocessVisitImageConfig", "combine_backgrounds"]

import numpy as np
import smatch

import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.algorithms
import lsst.meas.deblender
import lsst.meas.extensions.photometryKron
import lsst.meas.extensions.shapeHSM
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks import computeExposureSummaryStats, repair, snapCombine


class ReprocessVisitImageConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
):
    exposures = connectionTypes.Input(
        doc="Exposure (or two snaps) to be calibrated, and detected and measured on.",
        name="postISRCCD",
        storageClass="Exposure",
        multiple=True,  # to handle 1 exposure or 2 snaps
        dimensions=["instrument", "exposure", "detector"],
    )
    visit_summary = connectionTypes.Input(
        doc="Visit-level catalog summarizing all image characterizations and calibrations.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=["instrument", "visit"],
    )
    initial_photo_calib = connectionTypes.Input(
        doc="Photometric calibration that was applied to exposure during the measurement of background_1."
        " Used to uncalibrate the background before subtracting it from the input exposure.",
        name="initial_photoCalib_detector",
        storageClass="PhotoCalib",
        dimensions=("instrument", "visit", "detector"),
    )
    background_1 = connectionTypes.Input(
        doc="Background models estimated during calibration.",
        name="initial_pvi_background",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )
    background_2 = connectionTypes.Input(
        doc="Background that was fit on top of background_1.",
        name="skyCorr",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
    )
    calib_sources = connectionTypes.Input(
        doc="Per-visit catalog of measurements to get 'calib_*' flags from.",
        name="finalized_src_table",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit"],
    )
    background_to_photometric_ratio = connectionTypes.Input(
        doc=(
            "Ratio of a background-flattened image to a photometric-flattened image. "
            "Only used if do_apply_flat_background_ratio is True."
        ),
        name="background_to_photometric_ratio",
        storageClass="Image",
        dimensions=("instrument", "visit", "detector"),
    )
    # TODO DM-46947: pull in the STREAK mask from CompareWarp.

    # outputs
    sources_schema = connectionTypes.InitOutput(
        doc="Schema of the output sources catalog.",
        name="sources_schema",
        storageClass="SourceCatalog",
    )

    exposure = connectionTypes.Output(
        doc="Photometrically calibrated exposure with attached calibrations and summary statistics.",
        name="pvi",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    sources = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure.",
        name="sources_detector",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector"],
    )
    sources_footprints = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure; includes source footprints.",
        name="sources_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    background = connectionTypes.Output(
        doc=(
            "Total background model including new detections in this task. "
            "Note that the background model has units of ADU, while the corresponding "
            "image has units of nJy - the image must be 'uncalibrated' before the background "
            "can be restored."
        ),
        name="pvi_background",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
    )

    def __init__(self, *, config=None):
        if not config.do_use_sky_corr:
            del self.background_2
        if not config.remove_initial_photo_calib:
            del self.initial_photo_calib
        if not config.do_apply_flat_background_ratio:
            del self.background_to_photometric_ratio


class ReprocessVisitImageConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=ReprocessVisitImageConnections
):
    # To generate catalog ids consistently across subtasks.
    id_generator = lsst.meas.base.DetectorVisitIdGeneratorConfig.make_field()

    do_use_sky_corr = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Include the skyCorr input for background subtraction?",
    )
    remove_initial_photo_calib = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Remove an already-applied photometric calibration from the backgrounds?",
    )
    snap_combine = pexConfig.ConfigurableField(
        target=snapCombine.SnapCombineTask,
        doc="Task to combine two snaps to make one exposure.",
    )
    repair = pexConfig.ConfigurableField(
        target=repair.RepairTask,
        doc="Task to repair cosmic rays on the exposure before PSF determination.",
    )
    detection = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SourceDetectionTask,
        doc="Task to detect sources to return in the output catalog.",
    )
    sky_sources = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SkyObjectsTask,
        doc="Task to generate sky sources ('empty' regions where there are no detections).",
    )
    deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask, doc="Split blended sources into their components."
    )
    measurement = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to measure sources to return in the output catalog.",
    )
    normalized_calibration_flux = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.NormalizedCalibrationFluxTask,
        doc="Task to normalize the calibration flux (e.g. compensated tophats).",
    )
    apply_aperture_correction = pexConfig.ConfigurableField(
        target=lsst.meas.base.ApplyApCorrTask,
        doc="Task to apply aperture corrections to the measured sources.",
    )
    set_primary_flags = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.setPrimaryFlags.SetPrimaryFlagsTask,
        doc="Task to add isPrimary to the catalog.",
    )
    catalog_calculation = pexConfig.ConfigurableField(
        target=lsst.meas.base.CatalogCalculationTask,
        doc="Task to compute catalog values using only the catalog entries.",
    )
    post_calculations = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to compute catalog values after all other calculations have been done.",
    )
    compute_summary_stats = pexConfig.ConfigurableField(
        target=computeExposureSummaryStats.ComputeExposureSummaryStatsTask,
        doc="Task to to compute summary statistics on the calibrated exposure.",
    )
    calib_match_radius = pexConfig.Field(
        dtype=float,
        default=0.2,
        doc="Radius in arcseconds to cross-match calib_sources to the output catalog.",
    )
    do_apply_flat_background_ratio = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="This should be True if processing was done with an illumination correction.",
    )

    def setDefaults(self):
        super().setDefaults()

        # No need to redo background: we have the global background model.
        self.detection.reEstimateBackground = False
        self.detection.doTempLocalBackground = False

        # NOTE: these apertures were selected for HSC, and may not be
        # what we want for LSSTCam.
        self.measurement.plugins["base_CircularApertureFlux"].radii = [
            3.0,
            4.5,
            6.0,
            9.0,
            12.0,
            17.0,
            25.0,
            35.0,
            50.0,
            70.0,
        ]
        lsst.meas.extensions.shapeHSM.configure_hsm(self.measurement)
        self.measurement.plugins.names |= ["base_Jacobian", "base_FPPosition", "ext_photometryKron_KronFlux"]
        self.measurement.plugins["base_Jacobian"].pixelScale = 0.2

        # TODO DM-46306: should make this the ApertureFlux default!
        # Use a large aperture to be independent of seeing in calibration
        self.measurement.plugins["base_CircularApertureFlux"].maxSincRadius = 12.0

        # Only apply calibration fluxes, do not measure them.
        self.normalized_calibration_flux.do_measure_ap_corr = False

        self.post_calculations.plugins.names = ["base_LocalPhotoCalib", "base_LocalWcs"]
        self.post_calculations.doReplaceWithNoise = False
        for key in self.post_calculations.slots:
            setattr(self.post_calculations.slots, key, None)

    def validate(self):
        super().validate()

        if self.do_apply_flat_background_ratio:
            if self.detection.reEstimateBackground:
                if not self.detection.doApplyFlatBackgroundRatio:
                    raise pexConfig.FieldValidationError(
                        ReprocessVisitImageConfig.detection,
                        self,
                        "ReprocessVisitImageConfig.detection background must be configured with "
                        "doApplyFlatBackgroundRatio if do_apply_flat_background_ratio is True.",
                    )


class ReprocessVisitImageTask(pipeBase.PipelineTask):
    """Use the visit-level calibrations to perform detection and measurement
    on the single frame exposures and produce a "final" exposure and catalog.
    """

    ConfigClass = ReprocessVisitImageConfig
    _DefaultName = "reprocessVisitImage"

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("snap_combine")
        self.makeSubtask("repair")
        self.makeSubtask("detection", schema=schema)
        self.makeSubtask("sky_sources", schema=schema)
        self.makeSubtask("deblend", schema=schema)
        self.makeSubtask("measurement", schema=schema)
        self.makeSubtask("normalized_calibration_flux", schema=schema)
        self.makeSubtask("apply_aperture_correction", schema=schema)
        self.makeSubtask("catalog_calculation", schema=schema)
        self.makeSubtask("set_primary_flags", schema=schema, isSingleFrame=True)
        self.makeSubtask("post_calculations", schema=schema)
        self.makeSubtask("compute_summary_stats")

        schema.addField(
            "visit",
            type="L",
            doc="Visit this source appeared on.",
        )
        schema.addField(
            "detector",
            type="U",
            doc="Detector this source appeared on.",
        )

        # These fields will be propagated from finalizeCharacterization.
        # It might be better to get them from the finalized catalog instead
        # (if it output a schema), so the docstrings exactly match.
        schema.addField(
            "calib_psf_candidate",
            type="Flag",
            doc="Set if the source was a candidate for PSF determination, "
            "as determined from FinalizeCharacterizationTask.",
        )
        schema.addField(
            "calib_psf_reserved",
            type="Flag",
            doc="set if source was reserved from PSF determination by FinalizeCharacterizationTask.",
        )
        schema.addField(
            "calib_psf_used",
            type="Flag",
            doc="Set if source was used in the PSF determination by FinalizeCharacterizationTask.",
        )
        self.psf_fields = ("calib_psf_candidate", "calib_psf_used", "calib_psf_reserved")

        # TODO (DM-46971):
        # These fields are only here to satisfy the SDM schema, and will
        # be removed from there as they are misleading (because we don't
        # propagate this information from gbdes/fgcmcal).
        schema.addField(
            "calib_photometry_used",
            type="Flag",
            doc="Unused; placeholder for SDM schemas.",
        )
        schema.addField(
            "calib_photometry_reserved",
            type="Flag",
            doc="Unused; placeholder for SDM schemas.",
        )
        schema.addField(
            "calib_astrometry_used",
            type="Flag",
            doc="Unused; placeholder for SDM schemas.",
        )
        schema.addField(
            "calib_astrometry_reserved",
            type="Flag",
            doc="Unused; placeholder for SDM schemas.",
        )
        # This pre-calibration schema is the one that most methods should use.
        self.schema = schema
        # The final catalog will have calibrated flux columns, which we add to
        # the init-output schema by calibrating our zero-length catalog with an
        # arbitrary dummy PhotoCalib.
        dummy_photo_calib = afwImage.PhotoCalib(1.0, 0, bbox=lsst.geom.Box2I())
        self.sources_schema = dummy_photo_calib.calibrateCatalog(afwTable.SourceCatalog(schema))

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        id_generator = self.config.id_generator.apply(butlerQC.quantum.dataId)

        detector = outputRefs.exposure.dataId["detector"]
        exposures = inputs.pop("exposures")
        visit_summary = inputs.pop("visit_summary")
        calib_sources = inputs.pop("calib_sources")
        if self.config.remove_initial_photo_calib:
            initial_photo_calib = inputs.pop("initial_photo_calib")
        else:
            initial_photo_calib = None
        background_1 = inputs.pop("background_1")
        if self.config.do_use_sky_corr:
            background_2 = inputs.pop("background_2")
            background = combine_backgrounds(background_1, background_2)
        else:
            background = background_1
        if self.config.do_apply_flat_background_ratio:
            background_to_photometric_ratio = inputs.pop("background_to_photometric_ratio")
        else:
            background_to_photometric_ratio = None

        # This should not happen with a properly configured execution context.
        assert not inputs, "runQuantum got more inputs than expected"

        detector_summary = visit_summary.find(detector)
        lines = []
        if detector_summary is None:
            lines.append("  > no entry for the detector was found in the visit summary table")
        else:
            if detector_summary.psf is None:
                lines.append("  > the PSF model for the detector is None")
            if detector_summary.wcs is None:
                lines.append("  > the WCS model for the detector is None")
            if detector_summary.apCorrMap is None:
                lines.append("  > the aperture correction model map for the detector is None")
            if detector_summary.photoCalib is None:
                lines.append("  > the photometric calibration model for the detector is None")

        if lines:
            msg = "\n".join(lines)
            raise pipeBase.UpstreamFailureNoWorkFound(
                f"Skipping reprocessing of detector {detector} because:\n{msg}"
            )

        # Specify the fields that `annotate` needs below, to ensure they
        # exist, even as None.
        result = pipeBase.Struct(
            exposure=None,
            sources_footprints=None,
        )
        try:
            self.run(
                exposures=exposures,
                initial_photo_calib=initial_photo_calib,
                psf=detector_summary.psf,
                background=background,
                ap_corr=detector_summary.apCorrMap,
                photo_calib=detector_summary.photoCalib,
                wcs=detector_summary.wcs,
                calib_sources=calib_sources,
                result=result,
                id_generator=id_generator,
                background_to_photometric_ratio=background_to_photometric_ratio,
            )
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(
                e, self, result.exposure, result.sources_footprints, log=self.log
            )
            butlerQC.put(result, outputRefs)
            raise error from e

        butlerQC.put(result, outputRefs)

    def run(
        self,
        *,
        exposures,
        initial_photo_calib,
        psf,
        background,
        ap_corr,
        photo_calib,
        wcs,
        calib_sources,
        id_generator=None,
        background_to_photometric_ratio=None,
        result=None,
    ):
        """Detect and measure sources on the exposure(s) (snap combined as
        necessary), and make a "final" Processed Visit Image using all of the
        supplied metadata, plus a catalog measured on it.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure` or
                    `list` [`lsst.afw.image.Exposure`]
            Post-ISR exposure(s), with an initial WCS, VisitInfo, and Filter.
            Modified in-place during processing if only one is passed.
            If two exposures are passed, treat them as snaps and combine
            before doing further processing.
        initial_photo_calib : `lsst.afw.image.PhotoCalib` or `None`
            Photometric calibration that was applied to exposure during the
            measurement of the background.  Should be `None` if and only if
            ``config.remove_initial_photo_calib` is false.
        psf : `lsst.afw.detection.Psf`
            PSF model for this exposure.
        background : `lsst.afw.math.BackgroundList`
            Total background that had been fit to the exposure so far;
            modified in place to include background fit when detecting sources.
        ap_corr : `lsst.afw.image.ApCorrMap`
            Aperture Correction model for this exposure.
        photo_calib : `lsst.afw.image.PhotoCalib`
            Photometric calibration model for this exposure.
        wcs : `lsst.afw.geom.SkyWcs`
            World Coordinate System model for this exposure.
        calib_sources : `astropy.table.Table`
            Per-visit catalog of measurements to get 'calib_*' flags from.
        id_generator : `lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides random seeds.
        background_to_photometric_ratio : `lsst.afw.image.ImageF`, optional
            Background to photometric ratio image, to convert between
            photometric flattened and background flattened image.
        result : `lsst.pipe.base.Struct`, optional
            Result struct that is modified to allow saving of partial outputs
            for some failure conditions. If the task completes successfully,
            this is also returned.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``exposure``
                Calibrated exposure, with pixels in nJy units.
                (`lsst.afw.image.Exposure`)
            ``sources``
                Sources that were measured on the exposure, with calibrated
                fluxes and magnitudes. (`astropy.table.Table`)
            ``sources_footprints``
                Footprints of sources that were measured on the exposure.
                (`lsst.afw.table.SourceCatalog`)
            ``background``
                Total background that was fit to, and subtracted from the
                exposure when detecting ``sources``, in the same nJy units as
                ``exposure``. (`lsst.afw.math.BackgroundList`)
        """
        if result is None:
            result = pipeBase.Struct()
        if id_generator is None:
            id_generator = lsst.meas.base.IdGenerator()

        result.exposure = self.snap_combine.run(exposures).exposure

        # Apply the illumination correction if required.
        # This assumes the input images have had a background-flat applied.
        if self.config.do_apply_flat_background_ratio:
            result.exposure.maskedImage /= background_to_photometric_ratio

        if self.config.remove_initial_photo_calib:
            # Calibrate the image, so it's on the same units as the background.
            result.exposure.maskedImage = initial_photo_calib.calibrateImage(result.exposure.maskedImage)

        with lsst.meas.algorithms.backgroundFlatContext(
            result.exposure.maskedImage,
            self.config.do_apply_flat_background_ratio,
            backgroundToPhotometricRatio=background_to_photometric_ratio,
        ):
            result.exposure.maskedImage -= background.getImage()

        if self.config.remove_initial_photo_calib:
            # Uncalibrate so that we do the measurements in instFlux, because
            # we don't have a way to identify measurements as being in nJy.
            result.exposure.maskedImage /= initial_photo_calib.getCalibrationMean()

        result.exposure.setPsf(psf)
        result.exposure.setApCorrMap(ap_corr)
        result.exposure.setWcs(wcs)
        result.exposure.setPhotoCalib(photo_calib)

        result.sources_footprints = self._find_sources(
            result.exposure,
            background,
            calib_sources,
            id_generator,
            background_to_photometric_ratio=background_to_photometric_ratio,
        )
        result.background = background
        # TODO (DM-46971):
        # Now that we're running them before we apply the PhotoCalib to the
        # image pixels, there's no need for post_calibrations to exist as
        # a separate measurement instance from the main one (which is invoked
        # in _find_sources), but it's better to save removal (which may need
        # to involve a deprecation) until after we've got everything running.
        self.post_calculations.run(result.sources_footprints, result.exposure)
        result.exposure.info.setSummaryStats(
            self.compute_summary_stats.run(result.exposure, result.sources_footprints, background)
        )
        result.sources_footprints = self._apply_photo_calib(
            result.exposure,
            result.sources_footprints,
            photo_calib,
        )
        result.sources = result.sources_footprints.asAstropy()

        return result

    def _find_sources(
        self,
        exposure,
        background,
        calib_sources,
        id_generator,
        background_to_photometric_ratio=None,
    ):
        """Detect and measure sources on the exposure.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure to detect and measure sources on; must have a valid PSF.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection;
            modified in-place during subsequent detection.
        calib_sources : `astropy.table.Table`
            Per-visit catalog of measurements to get 'calib_*' flags from.
        id_generator : `lsst.meas.base.IdGenerator`
            Object that generates source IDs and provides random seeds.
        background_to_photometric_ratio : `lsst.afw.image.Image`, optional
            Image to convert photometric-flattened image to
            background-flattened image.

        Returns
        -------
        sources
            Catalog that was detected and measured on the exposure.
        """
        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        self.repair.run(exposure=exposure)
        detections = self.detection.run(
            table=table,
            exposure=exposure,
            background=background,
            backgroundToPhotometricRatio=background_to_photometric_ratio,
        )
        sources = detections.sources

        self.sky_sources.run(exposure.mask, id_generator.catalog_id, sources)

        self.deblend.run(exposure=exposure, sources=sources)
        # The deblender may not produce a contiguous catalog; ensure
        # contiguity for subsequent tasks.
        if not sources.isContiguous():
            sources = sources.copy(deep=True)

        self.measurement.run(sources, exposure)
        self.normalized_calibration_flux.run(exposure=exposure, catalog=sources)
        self.apply_aperture_correction.run(sources, exposure.apCorrMap)
        self.catalog_calculation.run(sources)
        self.set_primary_flags.run(sources)

        sources["visit"] = exposure.visitInfo.id
        sources["detector"] = exposure.info.getDetector().getId()

        self._match_calib_sources(sources, calib_sources, exposure.info.getDetector().getId())

        return sources

    def _match_calib_sources(self, sources, calib_sources, detector):
        """Match with calib_sources to set `calib_*` flags in the output
        catalog.

        Parameters
        ----------
        sources : `lsst.afw.table.SourceCatalog`
            Catalog that was detected and measured on the exposure. Modified
            in place to set the psf_fields.
        calib_sources : `astropy.table.Table`
            Per-visit catalog of measurements to get 'calib_*' flags from.
        detector : `int`
            Id of detector for this exposure, to get the correct sources from
            calib_sources for cross-matching.
        """
        # NOTE: we don't remove the sky sources here, but they should be very
        # far from any actual source, and so should not match with anything.
        use = calib_sources["detector"] == detector
        with smatch.Matcher(sources["coord_ra"], sources["coord_dec"]) as matcher:
            _, i1, i2, _ = matcher.query_knn(
                calib_sources[use]["coord_ra"],
                calib_sources[use]["coord_dec"],
                1,
                self.config.calib_match_radius / 3600.0,
                return_indices=True,
            )

        for field in self.psf_fields:
            # NOTE: Have to fill a full-sized array first, then set with it.
            result = np.zeros(len(sources), dtype=bool)
            result[i1] = calib_sources[use][i2][field]
            sources[field] = result

    def _apply_photo_calib(self, exposure, sources_footprints, photo_calib):
        """Photometrically calibrate the exposure and catalog with the
        supplied PhotoCalib, and set the exposure's PhotoCalib to 1.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure`
            Exposure to calibrate and set PhotoCalib on; Modified in place.
        sources_footprints : `lsst.afw.table.SourceCatalog`
            Catalog to calibrate.
        photo_calib : `lsst.afw.image.PhotoCalib`
            Photometric calibration to apply.
        calibrated_stars : `lsst.afw.table.SourceCatalog`
            Star catalog with flux/magnitude columns computed from the
            supplied PhotoCalib.
        """
        calibrated_sources_footprints = photo_calib.calibrateCatalog(sources_footprints)
        exposure.maskedImage = photo_calib.calibrateImage(exposure.maskedImage)
        identity = afwImage.PhotoCalib(1.0, photo_calib.getCalibrationErr(), bbox=exposure.getBBox())
        exposure.setPhotoCalib(identity)
        exposure.metadata["BUNIT"] = "nJy"
        return calibrated_sources_footprints


def combine_backgrounds(initial_pvi_background, sky_corr):
    """Return the total background that was applied to the original
    processing.
    """
    background = lsst.afw.math.BackgroundList()
    for item in initial_pvi_background:
        background.append(item)
    for item in sky_corr:
        background.append(item)
    return background
