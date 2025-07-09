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

__all__ = ["SingleFrameDetectAndMeasureTask", "SingleFrameDetectAndMeasureConfig"]

import lsst.afw.table as afwTable
import lsst.geom
import lsst.meas.algorithms
import lsst.meas.deblender
import lsst.meas.extensions.photometryKron
import lsst.meas.extensions.shapeHSM
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes


class SingleFrameDetectAndMeasureConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "detector")
):
    # inputs
    exposure = connectionTypes.Input(
        doc="Exposure to be calibrated, and detected and measured on.",
        name="preliminary_visit_image",
        storageClass="Exposure",
        dimensions=["instrument", "visit", "detector"],
    )
    input_background = connectionTypes.Input(
        doc="Background models estimated during calibration task; calibrated to be in nJy units.",
        name="preliminary_visit_image_background",
        storageClass="Background",
        dimensions=("instrument", "visit", "detector"),
    )

    # outputs
    sources = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure.",
        name="single_visit_star_reprocessed_unstandardized",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit", "detector"],
    )
    sources_footprints = connectionTypes.Output(
        doc="Catalog of measured sources detected on the calibrated exposure; includes source footprints.",
        name="single_visit_star_reprocessed_footprints",
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
        name="preliminary_visit_image_reprocessed_background",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
    )


class SingleFrameDetectAndMeasureConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=SingleFrameDetectAndMeasureConnections
):
    # To generate catalog ids consistently across subtasks.
    id_generator = lsst.meas.base.DetectorVisitIdGeneratorConfig.make_field()

    detection = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SourceDetectionTask,
        doc="Task to detect sources to return in the output catalog.",
    )
    sky_sources = pexConfig.ConfigurableField(
        target=lsst.meas.algorithms.SkyObjectsTask,
        doc="Task to generate sky sources ('empty' regions where there are no detections).",
    )
    deblend = pexConfig.ConfigurableField(
        target=lsst.meas.deblender.SourceDeblendTask,
        doc="Task to split blended sources into their components.",
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
    do_add_sky_sources = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Generate sky sources?",
    )

    def setDefaults(self):
        super().setDefaults()

        # Re-estimate the background
        self.detection.reEstimateBackground = True
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


class SingleFrameDetectAndMeasureTask(pipeBase.PipelineTask):
    """Use the visit-level calibrations to perform detection and measurement
    on the single frame exposures and produce a "final" exposure and catalog.
    """

    ConfigClass = SingleFrameDetectAndMeasureConfig
    _DefaultName = "singleFrameDetectAndMeasure"

    def __init__(self, schema=None, **kwargs):
        super().__init__(**kwargs)

        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()

        self.makeSubtask("detection", schema=schema)
        self.makeSubtask("sky_sources", schema=schema)
        self.makeSubtask("deblend", schema=schema)
        self.makeSubtask("measurement", schema=schema)
        self.makeSubtask("normalized_calibration_flux", schema=schema)
        self.makeSubtask("apply_aperture_correction", schema=schema)
        self.makeSubtask("catalog_calculation", schema=schema)
        self.makeSubtask("set_primary_flags", schema=schema, isSingleFrame=True)

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

        self.schema = schema

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        id_generator = self.config.id_generator.apply(butlerQC.quantum.dataId)

        exposure = inputs.pop("exposure")
        input_background = inputs.pop("input_background")

        # This should not happen with a properly configured execution context.
        assert not inputs, "runQuantum got more inputs than expected"

        # Specify the fields that `annotate` needs below, to ensure they
        # exist, even as None.
        result = pipeBase.Struct(
            sources=None,
            sources_footprints=None,
        )
        try:
            self.run(
                exposure=exposure,
                input_background=input_background,
                id_generator=id_generator,
                result=result,
            )
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(
                e, self, result.sources_footprints, log=self.log
            )
            butlerQC.put(result, outputRefs)
            raise error from e

        butlerQC.put(result, outputRefs)

    def run(
        self,
        exposure,
        input_background,
        id_generator=None,
        result=None,
    ):
        """Detect and measure sources on the exposure(s) (snap combined as
        necessary), and make a "final" Processed Visit Image using all of the
        supplied metadata, plus a catalog measured on it.
        Stripped-down version of `ReprocessVisitImageTask`.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Initial calibrated exposure.
            The DETECTED mask plane will be modified in place.
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

        table = afwTable.SourceTable.make(self.schema, id_generator.make_table_id_factory())

        detections = self.detection.run(
            table=table,
            exposure=exposure,
            background=input_background,
        )
        sources = detections.sources
        result.background = detections.background

        if self.config.do_add_sky_sources:
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
        result.sources_footprints = sources
        result.sources = sources.asAstropy()

        return result
