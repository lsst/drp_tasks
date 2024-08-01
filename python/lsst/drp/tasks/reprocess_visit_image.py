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

__all__ = ["ReprocessVisitImageTask", "ReprocessVisitImageConfig"]

import lsst.afw.table as afwTable
import lsst.afw.image as afwImage
import lsst.meas.deblender
import lsst.meas.algorithms
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.pipe.tasks import snapCombine

import lsst.meas.extensions.photometryKron
import lsst.meas.extensions.shapeHSM


class ReprocessVisitImageConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "visit", "detector")):
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

    # TODO: eventually we'll want to pull in the STREAK mask from the diffim.
    # TODO: do we need a background input or output here?
    # TODO: where to get the "best" PSF model from?

    # outputs
    source_schema = connectionTypes.InitOutput(
        doc="Schema of the output source catalog.",
        name="source_schema",
        storageClass="SourceCatalog",
    )

    exposure = connectionTypes.Output(
        doc="Photometrically calibrated exposure with fitted calibrations and summary statistics.",
        name="pvi",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
    )
    source = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure.",
        name="source_detector",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector"],
    )
    source_footprints = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure; "
            "includes source footprints.",
        name="source_footprints_detector",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
    )
    background = connectionTypes.Output(
        doc="Total background model adjusted for new detections in this task.",
        name="pvi_background",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
    )


class ReprocessVisitImageConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=ReprocessVisitImageConnections):

    # To generate catalog ids consistently across subtasks.
    id_generator = lsst.meas.base.DetectorVisitIdGeneratorConfig.make_field()

    snap_combine = pexConfig.ConfigurableField(
        target=snapCombine.SnapCombineTask,
        doc="Task to combine two snaps to make one exposure.",
    )
    detection = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SourceDetectionTask,
        doc="Task to detect sources to return in the output catalog."
    )
    sky_sources = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SkyObjectsTask,
        doc="Task to generate sky sources ('empty' regions where there are no detections).",
    )
    deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask,
        doc="Split blended sources into their components."
    )
    measurement = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to measure sources to return in the output catalog."
    )
    apply_aperture_correction = pexConfig.ConfigurableField(
        target=lsst.meas.base.ApplyApCorrTask,
        doc="Task to apply aperture corrections to the measured sources."
    )
    set_primary_flags = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.setPrimaryFlags.SetPrimaryFlagsTask,
        doc="Task to add isPrimary to the catalog."
    )
    catalog_calculation = pexConfig.ConfigurableField(
        target=lsst.meas.base.CatalogCalculationTask,
        doc="Task to compute catalog values using only the catalog entries."
    )
    # TODO: hopefully we can get rid of this, once Jim sorts out SDM details.
    post_calculations = pexConfig.ConfigurableField(
        target=lsst.meas.base.SingleFrameMeasurementTask,
        doc="Task to compute catalog values after all other calculations have been done."
    )

    def setDefaults(self):
        super().setDefaults()

        # TODO: ???
        # No need to estimate background: we have the global background model.
        # self.detection.reEstimateBackground = False
        self.detection.doTempLocalBackground = False

        # TODO: Note: these apertures were selected for HSC, and may not be
        # what we want for LSSTCam.
        self.measurement.plugins["base_CircularApertureFlux"].radii = [3.0,
                                                                       4.5,
                                                                       6.0,
                                                                       9.0,
                                                                       12.0,
                                                                       17.0,
                                                                       25.0,
                                                                       35.0,
                                                                       50.0,
                                                                       70.0
                                                                       ]
        lsst.meas.extensions.shapeHSM.configure_hsm(self.measurement)
        self.measurement.plugins.names |= ["base_Jacobian",
                                           "base_FPPosition",
                                           "ext_photometryKron_KronFlux"
                                           ]
        self.measurement.plugins["base_Jacobian"].pixelScale = 0.2

        # TODO: hopefully we can get rid of this, once Jim sorts out SDM.
        self.post_calculations.plugins.names = ["base_LocalPhotoCalib", "base_LocalWcs"]
        self.post_calculations.doReplaceWithNoise = False
        for key in self.post_calculations.slots:
            setattr(self.post_calculations.slots, key, None)


class ReprocessVisitImageTask(pipeBase.PipelineTask):
    """Use the visit-level calibrations to perform detection and measurement
    on the single frame exposures.
    """

    ConfigClass = ReprocessVisitImageConfig
    _DefaultName = "reprocessVisitImage"

    def __init__(self, source_schema=None, **kwargs):
        super().__init__(**kwargs)

        if source_schema is None:
            source_schema = afwTable.SourceTable.makeMinimalSchema()

        afwTable.CoordKey.addErrorFields(source_schema)
        self.makeSubtask("snap_combine")
        self.makeSubtask("detection", schema=source_schema)
        self.makeSubtask("sky_sources", schema=source_schema)
        self.makeSubtask("deblend", schema=source_schema)
        self.makeSubtask("measurement", schema=source_schema)
        self.makeSubtask("apply_aperture_correction", schema=source_schema)
        self.makeSubtask("catalog_calculation", schema=source_schema)
        self.makeSubtask("set_primary_flags", schema=source_schema, isSingleFrame=True)
        self.makeSubtask("post_calculations", schema=source_schema)
        self.source_schema = afwTable.SourceCatalog(source_schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        detector = outputRefs.exposure.dataId["detector"]
        exposures = inputs.pop("exposures")
        visit_summary = inputs.pop("visit_summary")
        background_1 = inputs.pop("background_1")
        background_2 = inputs.pop("background_2")

        id_generator = self.config.id_generator.apply(butlerQC.quantum.dataId)

        # This should not happen with a properly configured execution context.
        assert not inputs, "runQuantum got more inputs than expected"

        background = _combine_backgrounds(background_1, background_2)
        detector_summary = visit_summary.find(detector)
        # Specify the fields that `annotate` needs below, to ensure they
        # exist, even as None.
        result = pipeBase.Struct(exposure=None,
                                 source_footprints=None,
                                 )
        try:
            self.run(exposures=exposures,
                     psf=detector_summary.psf,
                     background=background,
                     ap_corr=detector_summary.apCorrMap,
                     photo_calib=detector_summary.photoCalib,
                     wcs=detector_summary.wcs,
                     summary_stats=detector_summary,
                     result=result,
                     id_generator=id_generator)
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(
                e,
                self,
                result.exposure,
                result.source_footprints,
                log=self.log
            )
            butlerQC.put(result, outputRefs)
            raise error from e

        butlerQC.put(result, outputRefs)

    def run(self, *, exposures, psf, background, ap_corr, photo_calib, wcs,
            summary_stats, id_generator=None, result=None):
        """Detect and measure sources on the exposure (snap combined as
        necessary), and make a "final" Processed Visit Image using all of the
        supplied metadata.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure` or
                    `list` [`lsst.afw.image.Exposure`]
            Post-ISR exposure(s), with an initial WCS, VisitInfo, and Filter.
            Modified in-place during processing if only one is passed.
            If two exposures are passed, treat them as snaps and combine
            before doing further processing.
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
        summary_stats : `lsst.afw.image.ExposureSummaryStats`
            Summary statistics measured on this exposure.
        id_generator : `lsst.meas.base.IdGenerator`, optional
            Object that generates source IDs and provides random seeds.
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
            ``source``
                Sources that were measured on the exposure calibrated fluxes
                and magnitudes.
                (`astropy.table.Table`)
            ``source_footprints``
                Footprints of sources that were measured on the exposure.
                (`lsst.afw.table.SourceCatalog`)
        """
        if result is None:
            result = pipeBase.Struct()
        if id_generator is None:
            id_generator = lsst.meas.base.IdGenerator()

        result.exposure = self.snap_combine.run(exposures).exposure

        result.exposure.maskedImage -= background.getImage()

        result.exposure.setPsf(psf)
        result.exposure.setApCorrMap(ap_corr)
        result.exposure.setWcs(wcs)
        result.exposure.info.setSummaryStats(summary_stats)
        # Note: we don't set photoCalib here, because we have to use it below
        # to calibrate the image and catalog, and thus the image will have a
        # photoCalib of exactly 1.

        result.source_footprints = self._find_sources(result.exposure, background, id_generator)
        result.background = background
        self._apply_photo_calib(result.exposure, result.source_footprints, photo_calib)

        # TODO: do we need this, or can we drop LocalWcs/LocalPhotoCalib?
        self.post_calculations.run(result.source_footprints, result.exposure)
        result.source = result.source_footprints.asAstropy()

        return result

    def _find_sources(self, exposure, background, id_generator):
        """Detect and measure sources on the exposure.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure`
            Exposure to detect and measure sources on; must have a valid PSF.
        background : `lsst.afw.math.BackgroundList`
            Background that was fit to the exposure during detection;
            modified in-place during subsequent detection.
        id_generator : `lsst.meas.base.IdGenerator`
            Object that generates source IDs and provides random seeds.

        Returns
        -------
        source
            Catalog that was detected and measured on the exposure.
        """
        table = afwTable.SourceTable.make(self.source_schema.schema,
                                          id_generator.make_table_id_factory())

        detections = self.detection.run(table=table, exposure=exposure, background=background)
        source = detections.sources

        self.sky_sources.run(exposure.mask, id_generator.catalog_id, source)

        self.deblend.run(exposure=exposure, sources=source)
        # The deblender may not produce a contiguous catalog; ensure
        # contiguity for subsequent tasks.
        if not source.isContiguous():
            source = source.copy(deep=True)

        self.measurement.run(source, exposure)
        self.apply_aperture_correction.run(source, exposure.apCorrMap)
        self.catalog_calculation.run(source)
        self.set_primary_flags.run(source)

        return source

    def _apply_photo_calib(self, exposure, source_footprints, photo_calib):
        """Photometrically calibrate the exposure and catalog with the
        supplied PhotoCalib, and set the exposure's PhotoCalib to 1.

        Parameters
        ----------
        exposures : `lsst.afw.image.Exposure`
            Exposure to calibrate and set PhotoCalib on; Modified in place.
        source_footprints : `lsst.afw.table.SourceCatalog`
            Catalog to calibrate.
        photo_calib : `lsst.afw.image.PhotoCalib`
            Photometric calibration to apply.
        calibrated_stars : `lsst.afw.table.SourceCatalog`
            Star catalog with flux/magnitude columns computed from the
            supplied PhotoCalib.
        """
        source_footprints = photo_calib.calibrateCatalog(source_footprints)
        exposure.maskedImage = photo_calib.calibrateImage(exposure.maskedImage)
        identity = afwImage.PhotoCalib(1.0,
                                       photo_calib.getCalibrationErr(),
                                       bbox=exposure.getBBox())
        exposure.setPhotoCalib(identity)
        return source_footprints


def _combine_backgrounds(initial_pvi_background, sky_corr):
    """Return the total background that was applied to the original processing.
    """
    background = lsst.afw.math.BackgroundList()
    for item in initial_pvi_background:
        background.append(item)
    for item in sky_corr:
        background.append(item)
    return background
