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

import numpy as np
from scipy.optimize import least_squares
from scipy.signal.windows import hann

import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.geom as geom
from lsst.ip.diffim.dcrModel import calculateDcr, wavelengthGenerator, fitThroughput
import lsst.meas.base as measBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.utils as utils
from lsst.skymap import BaseSkyMap


__all__ = ("CalculateDcrCorrectionConfig",
           "CalculateDcrCorrectionTask",
           )


class CalculateDcrCorrectionConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={
        "inputWarpName": "deep",
        "inputCoaddName": "deep",
        "warpType": "direct",
        "warpTypeSuffix": "",
        "fakesType": "",
    },
):
    inputWarps = pipeBase.connectionTypes.Input(
        doc="Input list of warps to be assembled i.e. stacked.",
        name="{inputWarpName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    templateCoadd = pipeBase.connectionTypes.Input(
        doc="Input coadded exposure, produced by previous call to AssembleCoadd",
        name="{fakesType}{inputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    objectCatalog = pipeBase.connectionTypes.Input(
        doc="Input coadded exposure, produced by previous call to AssembleCoadd",
        name="{fakesType}object_unforced_measurement",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded " "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    throughput = pipeBase.connectionTypes.Input(
        doc="Bandpass of the filter used for the observation.",
        name="standard_passband",
        storageClass="ArrowAstropy",
        dimensions=("band", "instrument"),
    )
    dcrCorrectionCatalog = pipeBase.connectionTypes.Output(
        doc="Output catalog of sub-band fluxes and footprints",
        name="{fakesType}dcr_correction_catalog",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    dcrResidual = pipeBase.connectionTypes.Output(
        doc="The template with DCR sources removed, so they can be added back"
        " using the DCR model.",
        name="{fakesType}dcr_residual_coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.doWriteDcrResidual:
            self.outputs.remove("dcrResidual")


class CalculateDcrCorrectionConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=CalculateDcrCorrectionConnections):
    dcrNumSubfilters = pexConfig.Field(
        dtype=int,
        doc="Number of sub-filters to forward model chromatic effects to fit the supplied exposures.",
        default=3,
    )
    minimumSNR = pexConfig.Field(
        doc="Minimum signal to noise of sources in the reference catalog to model.",
        dtype=float,
        default=30,
    )
    maximumSNR = pexConfig.Field(
        doc="Maximum signal to noise of sources in the reference catalog to model.",
        dtype=float,
        default=10000,
    )
    minimumModelFraction = pexConfig.Field(
        doc="Minimum fraction of the total flux to allow for the fit to each subfilter.",
        dtype=float,
        default=0.15,
    )
    maximumModelFraction = pexConfig.Field(
        doc="Minimum fraction of the total flux to allow for the fit to each subfilter.",
        dtype=float,
        default=0.7,
    )
    footprintSize = pexConfig.Field(
        dtype=int,
        doc="Size of the footprints to calculate the DCR correctionin around objects.",
        default=35,
    )
    bad_psf_threshold = pexConfig.Field(
        dtype=float,
        doc="Maximum relative difference between the PSF and a gaussian approximation.",
        default=0.2,
    )
    doTaperFootprint = pexConfig.Field(
        dtype=bool,
        doc="Weight the PSF model by a hanning window function to reduce edge artifacts?",
        default=True,
    )
    minNVisits = pexConfig.Field(
        dtype=int,
        doc="Minimum number of times a source must be observed to be included.",
        default=3,
    )
    doWriteDcrResidual = pexConfig.Field(
        dtype=bool,
        doc="Write the residual coadd exposure after removing the DCR modeled sources?",
        default=True,
    )


class CalculateDcrCorrectionTask(pipeBase.PipelineTask):
    """
    """
    ConfigClass = CalculateDcrCorrectionConfig
    _DefaultName = "calculateDcrCorrection"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema.addField("modelFlux", "F", doc="Fit PSF flux.", units="nJy")
        self.schema.addField("numSubfilters", "F", doc="Number of DCR subfilters.", units="count")
        self.schema.addField("isCoaddModel", "Flag",
                             doc="If set, the footprint of this record is the DCR-smeared model of the"
                                 " source as it appears in the coadd, which is to be subtracted. If not"
                                 " set, the footprint is the un-shifted model, which is to be shifted"
                                 " by the DCR of the science visit and added back. Each source has one"
                                 " record of each kind, sharing the same id.")
        for subfilter in range(self.config.dcrNumSubfilters):
            self.schema.addField(f"subfilterWeight_{subfilter}", "F",
                                 doc="Fraction of the full band flux attributed to this subfilter.",
                                 units="count")
            self.schema.addField(f"subfilterWavelength_{subfilter}", "F",
                                 doc="Central wavelength of this subfilter.",
                                 units="nm")
        # The following sets the necessary columns and mappings to the schema
        self.centroidName = "base_SdssCentroid"
        control = measBase.SdssCentroidControl()
        self.schema.getAliasMap().set("slot_Centroid", self.centroidName)
        self.centroider = measBase.SdssCentroidAlgorithm(control, self.centroidName, self.schema)

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring to be formatted with info from PipelineTask.runQuantum
        """
        Notes
        -----
        Assemble a coadd from a set of Warps.
        """
        inputData = butlerQC.get(inputRefs)
        # Construct skyInfo expected by run
        skyMap = inputData.pop("skyMap")
        throughput = fitThroughput(inputData.pop("throughput"))
        outputDataId = butlerQC.quantum.dataId

        skyInfo = makeSkyInfo(
            skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"]
        )

        # Construct list of input Deferred Datasets
        warpRefList = self.prepareInputs(inputData.pop("inputWarps"))
        self.log.info("Found %d input warps", len(warpRefList))
        if len(warpRefList) == 0:
            raise pipeBase.NoWorkFound("No coadd temporary exposures found")

        templateCoadd = inputData.pop("templateCoadd")
        objectCatalog = inputData.pop("objectCatalog")
        retStruct = self.run(
            warpRefList=warpRefList,
            templateCoadd=templateCoadd,
            objectCatalog=objectCatalog,
            effectiveWavelength=throughput.effectiveWavelength,
            bandwidth=throughput.bandwidth,
            bbox=skyInfo.bbox,
        )

        butlerQC.put(retStruct, outputRefs)
        return retStruct

    def prepareInputs(self, refList):
        """Check that the input warps are calibrated in the units this task
        expects.

        Only the metadata of each warp is read, so that the pixels are read
        exactly once, in `run`. Warps that are empty of data are skipped there
        rather than here, since that test does need the pixels.

        Parameters
        ----------
        refList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Dataset handles of the warps to use.

        Returns
        -------
        warpRefList : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            The same dataset handles, once they have all been validated.

        Raises
        ------
        ValueError
            If any warp does not record that it is calibrated in nJy.
        """
        for warpRef in refList:
            metadata = warpRef.get(component="metadata")
            if "BUNIT" not in metadata:
                raise ValueError(f"Warp {warpRef.dataId} has no BUNIT metadata")
            if metadata["BUNIT"] != "nJy":
                raise ValueError(
                    f"Warp {warpRef.dataId} has BUNIT {metadata['BUNIT']}, expected nJy"
                )
        return list(refList)

    def run(self, warpRefList, templateCoadd, objectCatalog, effectiveWavelength, bandwidth, bbox):
        self.metadata['effectiveWavelength'] = effectiveWavelength
        self.metadata['bandwidth'] = bandwidth
        self.effectiveWavelength = effectiveWavelength
        self.bandwidth = bandwidth
        self.log.info("Dividing %fnm bandwidth into %d subfilters with %fnm effective wavelength",
                      bandwidth, self.config.dcrNumSubfilters, effectiveWavelength)
        refCat = self.filter_object_catalog(objectCatalog)
        dcrFpLookupTable = {}
        cutoutLookupTable = {}
        unshiftedLookupTable = {}
        recordVisitCount = {}
        for record in refCat:
            recId = record.getId()
            dcrFpLookupTable[recId] = {}
            cutoutLookupTable[recId] = {}
            unshiftedLookupTable[recId] = {}
            recordVisitCount[recId] = 0
        nVisitsUsed = 0
        for warpRef in warpRefList:
            visit = warpRef.dataId['visit']
            # This is the only place the pixels of a warp are read.
            warp = warpRef.get(parameters={"bbox": bbox})
            if np.isnan(warp.image.array).all():
                self.log.info("Skipping visit %d because the warp is empty of data", visit)
                continue
            psf_metric, psf_gaussian = self.check_psf(warp)
            if psf_metric > self.config.bad_psf_threshold:
                self.log.info("Skipping visit %d due to bad PSF fit (metric %f > %f threshold)",
                              visit, psf_metric, self.config.bad_psf_threshold)
                continue
            else:
                self.log.info("Using visit %d with PSF fit metric %f", visit, psf_metric)
            nVisitsUsed += 1

            # Generate a lookup table with the shifted PSF models for each
            # subfilter, and the image cutouts for each object in the catalog
            lookupTableSingle = self.make_warp_footprints(refCat, warp, psf_gaussian)
            # Reformat the per-visit lookup table into new tables with a
            # different ordering, all indexed by source record first and
            # having an inner lookup table over visit.
            # That way we can solve for the best fit to the subfilters across
            # all visits at once for a single source.
            for record in refCat:
                recId = record.getId()
                if lookupTableSingle[recId] is not None:
                    dcrFpLookupTable[recId][visit] = lookupTableSingle[recId]['subfilterPsf']
                    cutoutLookupTable[recId][visit] = lookupTableSingle[recId]['cutout']
                    unshiftedLookupTable[recId][visit] = lookupTableSingle[recId]['unshiftedPsf']
                    recordVisitCount[recId] += 1
        if nVisitsUsed == 0:
            raise pipeBase.NoWorkFound("No input warps had usable data and an acceptable PSF fit.")
        self.log.info("Modeling DCR from %d of %d input warps", nVisitsUsed, len(warpRefList))
        # Drop any records that were removed from too many visits
        badRecords = np.array([recordVisitCount[record.getId()] < self.config.minNVisits
                              for record in refCat])
        if np.any(badRecords):
            for badRec in refCat[badRecords]:
                recId = badRec.getId()
                dcrFpLookupTable.pop(recId)
                cutoutLookupTable.pop(recId)
                unshiftedLookupTable.pop(recId)
                recordVisitCount.pop(recId)
            refCat = refCat[~badRecords].copy(deep=True)
        self.log.info("Calculating DCR correction for %d surviving sources, and dropping %d sources",
                      len(refCat), np.sum(badRecords))
        # Calculate one model per source
        results = self.calculateTemplateResidual(templateCoadd, dcrFpLookupTable, cutoutLookupTable,
                                                 unshiftedLookupTable)

        # Convert the lookup table to a source catalog with heavy footprints
        # containing the unshifted PSF model of the coadd at the source
        # location, and columns containing the overall flux and fractional flux
        # per subfilter
        dcrCorrectionCatalog = self.make_dcr_catalog(refCat, dcrFpLookupTable, results.fluxLookupTable,
                                                     results.templateFootprints,
                                                     results.coaddFootprints)
        return pipeBase.Struct(dcrResidual=results.residual,
                               dcrCorrectionCatalog=dcrCorrectionCatalog)

    def filter_object_catalog(self, objectCat):
        """Select sources to model from an input catalog.

        Only include moderately bright objects.
        Faint objects won't have enough signal to fit DCR, and bright objects
        will saturate the model in the bounding box and create unwanted
        artifacts.

        Parameters
        ----------
        objectCat : `lsst.afw.table.SourceCatalog`
            Description

        Returns
        -------
        filteredCatalog : `lsst.afw.table.SourceCatalog`
            Description
        """
        snr = objectCat['slot_CalibFlux_instFlux']/objectCat['slot_CalibFlux_instFluxErr']
        goodSnr = (snr > self.config.minimumSNR) & (snr < self.config.maximumSNR)
        # Exclude flagged objects that probably won't compute
        goodCentroid = ~objectCat['base_SdssCentroid_flag']
        goodShape = ~objectCat['base_SdssShape_flag']
        # Exclude extended objects
        goodExtendedness = objectCat['base_ClassificationSizeExtendedness_value'] < 0.5
        # Use isolated sources and deblended children, but not the blended
        # parents that those children were deblended from. This is the same
        # condition as `detect_isDeblendedSource`.
        # Note that `detect_isPrimary` is deliberately not used here, since it
        # also restricts to the inner region of the patch. The correction
        # catalog must cover the full outer bbox of the patch, so that a source
        # in the overlap between two patches is corrected in both of them.
        notParent = objectCat['deblend_nChild'] == 0
        # The source needs to fit in the defined footprint.
        # If it's larger, it's either trailed, extended, or just very bright
        # None of those cases will be fit well by the DCR model
        # Currently allow a slightly larger footprint, since the wings of the
        # footprint will extend beyond the core of the source we are fitting.
        maxFootprintArea = 2*self.config.footprintSize**2
        goodArea = objectCat['base_FootprintArea_value'] < maxFootprintArea
        srcUse = goodSnr & goodCentroid & goodShape & goodExtendedness & notParent & goodArea
        return objectCat[srcUse].copy(deep=True)

    def check_psf(self, warp):
        psf = warp.psf
        psf_pos = geom.Point2I(psf.getAveragePosition())

        psf_major, psf_minor = getPsfMajorMinorAxes(psf, useFwhm=False)
        psf_gaussian = afwDet.GaussianPsf(self.config.footprintSize, self.config.footprintSize, psf_minor)
        # xc, yc = self.config.footprintSize//2

        dcrShift = calculateDcr(warp.visitInfo, warp.getWcs(),
                                self.effectiveWavelength,
                                self.bandwidth,
                                self.config.dcrNumSubfilters,
                                )
        boxSize = geom.Extent2I(self.config.footprintSize, self.config.footprintSize)
        psf_bbox = geom.Box2I.makeCenteredBox(center=psf.getAveragePosition(), size=boxSize)
        psf_img = self.create_psf_image_in_bbox(psf, psf_bbox, psf_pos[0], psf_pos[1]).array
        fit_img = np.zeros_like(psf_img)
        for subfilter, shift in enumerate(dcrShift):
            xc, yc = psf_pos
            # shift format is numpy (y,x)
            xc += shift[1]
            yc += shift[0]
            subfilter_img = self.create_psf_image_in_bbox(psf_gaussian, psf_bbox, xc, yc).array
            fit_img += subfilter_img

        windowFunction = np.outer(hann(self.config.footprintSize), hann(self.config.footprintSize))
        windowFunction /= np.max(windowFunction)

        psf_img *= windowFunction
        fit_img *= windowFunction
        psf_norm = np.sum(psf_img[psf_img > np.max(psf_img)/4])
        psf_img /= psf_norm
        fit_norm = np.sum(fit_img[fit_img > np.max(fit_img)/4])
        fit_img /= fit_norm
        psf_metric = 2*np.sum(np.abs(psf_img - fit_img))/np.sum(psf_img + fit_img)
        return (psf_metric, psf_gaussian)

    @staticmethod
    def create_psf_image_in_bbox(psf, bbox, xc, yc):

        bbox2 = bbox.clippedTo(psf.computeImageBBox(geom.Point2D(xc, yc)))
        psf_img = afwImage.ImageF(bbox)
        psf_img[bbox2].array[:, :] = psf.computeImage(geom.Point2D(xc, yc))[bbox2].array
        return psf_img

    def initialize_dcr_catalog(self):
        """Create an empty catalog with the columns defined for the DCR schema

        Returns
        -------
        cat : `lsst.afw.table.SourceCatalog`
            Empty catalog with the correct schema.
        """
        cat = afwTable.SourceCatalog(self.schema)
        cat.defineCentroid(self.centroidName)
        return cat

    def make_dcr_catalog(self, refCat, dcrFpLookupTable, fluxLookupTable, templateFootprints,
                         coaddFootprints):
        """Build the output catalog of sub-band fluxes and model footprints.

        Each source is written as a pair of records that share the same id: one
        holding the un-shifted model, and one with ``isCoaddModel`` set holding
        the DCR-smeared model of the source as it appears in the coadd. All of
        the other columns are identical between the two.

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            Catalog of the sources that were modeled.
        dcrFpLookupTable : `dict` [`int`, `dict`]
            Subfilter models of each source, indexed on source id then visit.
            The subfilter weights are read from the ``modelFlux`` column.
        fluxLookupTable : `dict` [`int`, `float`]
            Weighted mean flux of each source, indexed on source id.
        templateFootprints : `dict` [`int`, \
                `lsst.afw.detection.HeavyFootprintF`]
            Un-shifted model of each source, indexed on source id.
        coaddFootprints : `dict` [`int`, `lsst.afw.detection.HeavyFootprintF`]
            DCR-smeared model of each source, indexed on source id.

        Returns
        -------
        dcrCorrectionCatalog : `lsst.afw.table.SourceCatalog`
            Catalog with two records per modeled source.
        """
        dcrCorrectionCatalog = self.initialize_dcr_catalog()
        dcrGen = wavelengthGenerator(self.effectiveWavelength,
                                     self.bandwidth,
                                     self.config.dcrNumSubfilters)
        subfilterEffectiveWavelengths = [np.mean(wl) for wl in dcrGen]
        for refSrc in refCat:
            srcId = refSrc.getId()
            if srcId not in templateFootprints:
                continue
            models = dcrFpLookupTable[srcId]
            visits = [visit for visit in models]
            # At this point the subfilter fractions are the same for each visit
            # so we can take the values from the first visit
            model = models[visits[0]]
            for isCoaddModel, footprints in ((False, templateFootprints), (True, coaddFootprints)):
                src = dcrCorrectionCatalog.addNew()
                src.setId(srcId)
                src['isCoaddModel'] = isCoaddModel
                src['numSubfilters'] = self.config.dcrNumSubfilters
                src['modelFlux'] = fluxLookupTable[srcId]
                src['coord_ra'] = refSrc['coord_ra']
                src['coord_dec'] = refSrc['coord_dec']
                src['base_SdssCentroid_x'], src['base_SdssCentroid_y'] = refSrc.getCentroid()
                src.setFootprint(footprints[srcId])

                for subfilter in range(self.config.dcrNumSubfilters):
                    src[f'subfilterWeight_{subfilter}'] = model[subfilter]['modelFlux']
                    src[f'subfilterWavelength_{subfilter}'] = subfilterEffectiveWavelengths[subfilter]

        return dcrCorrectionCatalog

    def make_warp_footprints(self, catalog, warp, psf):
        image_footprints = self.initialize_dcr_catalog()
        fp_ctrl = afwDet.HeavyFootprintCtrl()
        if self.config.doTaperFootprint:
            windowFunction = np.outer(hann(self.config.footprintSize), hann(self.config.footprintSize))
            windowFunction /= np.max(windowFunction)
        else:
            windowFunction = None
        # Extract cutouts from the image centered on each source, and reject
        # any with a bad fit to the catalog flux or containing invalid values.
        lookupTable = self.build_image_lookup_table(catalog, warp, image_footprints,
                                                    windowFunction=windowFunction, fp_ctrl=fp_ctrl)
        # Update the lookup table with DCR-shifted PSFs for each source, for
        # each subfilter.
        dcrShift = calculateDcr(warp.visitInfo, warp.getWcs(),
                                self.effectiveWavelength,
                                self.bandwidth,
                                self.config.dcrNumSubfilters,
                                )
        self.update_subfilter_psf_lookup_table(lookupTable, catalog, psf, dcrShift,
                                               fp_ctrl=fp_ctrl, windowFunction=windowFunction)
        # Also record the same PSF with no DCR shift applied. Stacking these
        # gives the model of the source with the DCR of the template visits
        # removed, which is what must be shifted by the DCR of the science
        # visit and added back when the template is built.
        self.update_unshifted_psf_lookup_table(lookupTable, catalog, psf,
                                               windowFunction=windowFunction)
        # Determine the best fit scale factors for each source, using the
        # flux of the source and the DCR-shifted PSFs for each subfilter
        for record in catalog:
            recId = record.getId()
            if lookupTable[recId] is not None:
                image_fp = lookupTable[recId]['cutout']
                psf_fps = lookupTable[recId]['subfilterPsf']
                scales = self.minimize_footprint_residuals(image_fp, psf_fps)
                for psf_fp, scale in zip(psf_fps, scales):
                    psf_fp['modelFlux'] = scale
        return lookupTable

    def update_unshifted_psf_lookup_table(self, lookupTable, catalog, psf, windowFunction=None):
        """Add the PSF model with no DCR shift applied to the lookup table.

        This uses the same bounding box, PSF and window function as
        `update_subfilter_psf_lookup_table`, so that the two sets of models can
        be stacked over visits in exactly the same way. The only difference is
        that no DCR shift is applied.

        Parameters
        ----------
        lookupTable : `dict` [`int`, `dict`]
            Lookup table of the models for each source, indexed on source id.
            Updated in place to add an ``unshiftedPsf`` entry, which is the
            un-shifted PSF image as a `numpy.ndarray`.
        catalog : `lsst.afw.table.SourceCatalog`
            Catalog of the sources to model.
        psf : `lsst.afw.detection.Psf`
            Gaussian approximation of the PSF of the visit.
        windowFunction : `numpy.ndarray`, optional
            Taper to apply to the model, to reduce edge artifacts.
        """
        boxSize = geom.Extent2I(self.config.footprintSize, self.config.footprintSize)
        for record in catalog:
            if lookupTable[record.getId()] is None:
                # Skip any records that we were not able to extract a clean
                # image cutout for.
                continue
            xc, yc = record.getCentroid()
            bbox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=boxSize)
            psf_img = self.create_psf_image_in_bbox(psf, bbox, xc, yc)
            if windowFunction is not None:
                psf_img.array *= windowFunction
            lookupTable[record.getId()]['unshiftedPsf'] = psf_img.array

    def update_subfilter_psf_lookup_table(self, lookupTable, catalog, psf, dcrShift,
                                          fp_ctrl=afwDet.HeavyFootprintCtrl(), windowFunction=None):

        boxSize = geom.Extent2I(self.config.footprintSize, self.config.footprintSize)
        for subfilter, shift in enumerate(dcrShift):
            # instantiate the catalog, and define the centroid
            cat = self.initialize_dcr_catalog()
            # Next define footprints
            for record in catalog:
                if lookupTable[record.getId()] is None:
                    # Skip any records that we were not able to extract a clean
                    # image cutout for.
                    continue
                xc, yc = record.getCentroid()
                bbox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=boxSize)
                # shift format is numpy (y,x)
                xc += shift[1]
                yc += shift[0]
                src = cat.addNew()
                src.setId(record.getId())
                subFlux = 1/self.config.dcrNumSubfilters
                src["modelFlux"] = subFlux
                src['base_SdssCentroid_x'] = xc
                src['base_SdssCentroid_y'] = yc
                foot = afwDet.Footprint(afwGeom.SpanSet(bbox))
                foot.addPeak(xc, yc, subFlux)
                # Note, we don't just use
                # afwImage.ImageF(warp.psf.computeImage(geom.Point2D(xc, yc)),
                #                 deep=True)
                # because we need the shifted bbox
                psf_img = self.create_psf_image_in_bbox(psf, bbox, xc, yc)
                if windowFunction is not None:
                    psf_img.array *= windowFunction
                psf_mask = afwImage.Mask(bbox)
                psf_variance = afwImage.ImageF(bbox)
                psf_mimage = afwImage.MaskedImageF(psf_img, psf_mask, psf_variance)

                heavy_fp = afwDet.HeavyFootprintF(foot, psf_mimage, fp_ctrl)
                src.setFootprint(heavy_fp)
                lookupTable[record.getId()]['subfilterPsf'].append(src)

    def build_image_lookup_table(self, catalog, warp, image_footprints, windowFunction=None,
                                 fp_ctrl=afwDet.HeavyFootprintCtrl()):

        image_footprints = self.initialize_dcr_catalog()
        lookupTable = {}
        boxSize = geom.Extent2I(self.config.footprintSize, self.config.footprintSize)
        coreBoxSize = geom.Extent2I(self.config.footprintSize//2, self.config.footprintSize//2)
        # Next define footprints
        for record in catalog:
            xc, yc = record.getCentroid()
            bbox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=boxSize)
            coreBBox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=coreBoxSize)
            if np.any(warp[coreBBox].mask.array & warp.mask.getPlaneBitMask('NO_DATA')):
                lookupTable[record.getId()] = None
                continue

            spans = afwGeom.SpanSet(bbox)
            base_psf = warp.psf.computeImage(geom.Point2D(xc, yc)).array
            bbox_psf = warp.psf.computeImageBBox(geom.Point2D(xc, yc))
            cutout_arr = warp[bbox_psf].image.array
            flux = fit_footprints(base_psf, cutout_arr)
            if not np.isfinite(flux):
                lookupTable[record.getId()] = None
                continue
            deltaFlux = 2*abs(flux - record.getCalibInstFlux())/(flux + record.getCalibInstFlux())
            if deltaFlux > .5:
                # If the fit flux is much brighter than the calibration
                # flux, skip the source since it is more likely to
                # create artifacts.
                lookupTable[record.getId()] = None
                continue
            cutout_mi = warp[bbox].maskedImage.clone()
            if np.any(np.isnan(cutout_mi.image.array)):
                lookupTable[record.getId()] = None
                continue
            cutout = image_footprints.addNew()
            cutout.setId(record.getId())
            cutout["modelFlux"] = flux
            cutout['base_SdssCentroid_x'] = xc
            cutout['base_SdssCentroid_y'] = yc
            foot = afwDet.Footprint(spans)
            foot.addPeak(xc, yc, flux)
            if windowFunction is not None:
                cutout_mi.image.array *= windowFunction
            cutout.setFootprint(afwDet.HeavyFootprintF(foot, cutout_mi, fp_ctrl))
            lookupTable[record.getId()] = {}
            lookupTable[record.getId()]['cutout'] = cutout
            lookupTable[record.getId()]['subfilterPsf'] = []
        return lookupTable

    def minimize_footprint_residuals(self, image_fp, psf_fps):
        """Fit the fraction of a source's flux belonging to each subfilter.

        Parameters
        ----------
        image_fp : `lsst.afw.table.SourceRecord`
            Image cutout of the source, with its fit flux in ``modelFlux``.
        psf_fps : `list` [`lsst.afw.table.SourceRecord`]
            DCR-shifted PSF model of each subfilter.

        Returns
        -------
        scales : `list` [`float`]
            Fraction of the flux of ``image_fp`` belonging to each subfilter.
        """
        scales0 = [image_fp['modelFlux']*psf_fp['modelFlux'] for psf_fp in psf_fps]
        nSubfilters = len(psf_fps)
        # ``fill`` must be set: it defaults to NaN, which would poison any
        # pixel of the bbox that is outside the footprint.
        img = image_fp.getFootprint().extractImage(fill=0.).array
        psf_arrays = [psf.getFootprint().extractImage(fill=0.).array for psf in psf_fps]

        def residuals(scales):
            residual = img.copy()
            for psf, scale in zip(psf_arrays, scales):
                residual -= scale*psf
            # Removing the mean makes the fit insensitive to any residual
            # background in the cutout. Minimizing the sum of squares of this
            # is equivalent to minimizing the standard deviation of the
            # residual, but returning the individual residuals instead of a
            # single number lets `least_squares` use the Jacobian and converge
            # far faster.
            return (residual - np.mean(residual)).ravel()
        minFluxFit = self.config.minimumModelFraction*image_fp['modelFlux']
        maxFluxFit = self.config.maximumModelFraction*image_fp['modelFlux']
        scaleFit = least_squares(residuals, scales0,
                                 bounds=[[minFluxFit]*nSubfilters, [maxFluxFit]*nSubfilters])
        scales = [scale/image_fp['modelFlux'] for scale in scaleFit.x]
        return scales

    def calculateTemplateResidual(self, templateCoadd, dcrFpLookupTable, cutoutLookupTable,
                                  unshiftedLookupTable):
        """Stack the per-visit models of each source over all of its visits.

        Two models are produced for each source. The DCR-smeared model is the
        appearance of the source in the coadd, and is subtracted to make
        ``residual``. The un-shifted model is the same stack with no DCR shift
        applied, and is the model that must be shifted by the DCR of a science
        visit and added back in its place.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.Exposure`
            The coadd that the models are subtracted from.
        dcrFpLookupTable : `dict` [`int`, `dict`]
            DCR-shifted subfilter models of each source, indexed on source id
            and then visit. The ``modelFlux`` of each is updated in place to
            the subfilter weight fit across all visits.
        cutoutLookupTable : `dict` [`int`, `dict`]
            Image cutouts of each source, indexed on source id then visit.
        unshiftedLookupTable : `dict` [`int`, `dict`]
            Un-shifted PSF models of each source, indexed on source id then
            visit.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct with attributes:

            ``residual``
                The coadd with the DCR-smeared model of every source subtracted
                (`lsst.afw.image.Exposure`).
            ``fluxLookupTable``
                Weighted mean flux of each source, indexed on source id
                (`dict` [`int`, `float`]).
            ``templateFootprints``
                Un-shifted model of each source, indexed on source id
                (`dict` [`int`, `lsst.afw.detection.HeavyFootprintF`]).
            ``coaddFootprints``
                DCR-smeared model of each source, indexed on source id
                (`dict` [`int`, `lsst.afw.detection.HeavyFootprintF`]).
        """
        inputs = templateCoadd.getInfo().getCoaddInputs()
        weightLookup = {}
        for visit in inputs.ccds['visit']:
            inds = inputs.ccds['visit'] == visit
            weightLookup[visit] = np.mean(inputs.ccds['weight'][inds])
        scaleLookup = {}

        templateFootprints = {}
        coaddFootprints = {}
        fp_ctrl = afwDet.HeavyFootprintCtrl()
        residual = templateCoadd.clone()
        fluxLookupTable = {}
        for recId in dcrFpLookupTable:
            visits = list(dcrFpLookupTable[recId])
            if not visits:
                continue
            scales = [[fp['modelFlux'] for fp in dcrFpLookupTable[recId][visit]] for visit in visits]
            recScales = np.median(scales, axis=0)
            scaleSum = np.sum(recScales)
            if not np.isfinite(scaleSum) or scaleSum <= 0:
                self.log.debug("Subfilter weights for source %d do not sum to a positive value;"
                               " skipping.", recId)
                continue
            # The subfilter weights are fit per visit, but the sub-band flux of
            # a source does not vary between visits, so use the same weights
            # for every visit. Normalizing them to sum to one makes the
            # correction conserve the flux of the source.
            scalesSingle = recScales/scaleSum
            for visit in visits:
                for fp, scale in zip(dcrFpLookupTable[recId][visit], scalesSingle):
                    fp['modelFlux'] = scale
            try:
                model, unshiftedModel, flux = stack_dcr_footprints(dcrFpLookupTable[recId],
                                                                   cutoutLookupTable[recId],
                                                                   unshiftedLookupTable[recId],
                                                                   weightLookup
                                                                   )
            except RuntimeError:
                continue
            fluxLookupTable[recId] = flux
            scaleLookup[recId] = scalesSingle
            # The bbox and centroid are the same for every visit, since they
            # are both set from the reference catalog.
            refVisit = visits[0]
            bbox = cutoutLookupTable[recId][refVisit].getFootprint().getBBox()
            spans = afwGeom.SpanSet(bbox)
            residual[bbox].image.array -= model

            xc = cutoutLookupTable[recId][refVisit]['base_SdssCentroid_x']
            yc = cutoutLookupTable[recId][refVisit]['base_SdssCentroid_y']
            foot = afwDet.Footprint(spans)
            foot.addPeak(xc, yc, flux)
            # Store both models. The DCR-smeared one is subtracted from the
            # coadd, and the un-shifted one is shifted by the DCR of the
            # science visit and added back in its place.
            model_mi = templateCoadd[bbox].maskedImage.clone()
            model_mi.image.array = unshiftedModel
            templateFootprints[recId] = afwDet.HeavyFootprintF(foot, model_mi, fp_ctrl)
            model_mi.image.array = model
            coaddFootprints[recId] = afwDet.HeavyFootprintF(foot, model_mi, fp_ctrl)
        return pipeBase.Struct(residual=residual,
                               fluxLookupTable=fluxLookupTable,
                               templateFootprints=templateFootprints,
                               coaddFootprints=coaddFootprints,
                               )


def fit_footprints(model, image):
    model_flat = model.ravel()
    image_flat = image.ravel()
    cov = np.cov(image_flat*model_flat, model_flat*model_flat)[0, 1]
    varM = np.var(model**2)
    scale = cov / varM
    return scale


def stack_dcr_footprints(dcrFootprints, cutouts, unshifted, weightLookup):
    """Stack the per-visit models of one source over all of its visits.

    Parameters
    ----------
    dcrFootprints : `dict` [`int`, `list` [`lsst.afw.table.SourceRecord`]]
        DCR-shifted subfilter models for each visit, with the subfilter weight
        in the ``modelFlux`` column.
    cutouts : `dict` [`int`, `lsst.afw.table.SourceRecord`]
        Image cutout for each visit, with the fit flux in ``modelFlux``.
    unshifted : `dict` [`int`, `numpy.ndarray`]
        Un-shifted PSF model for each visit, on the same bounding box as the
        shifted models.
    weightLookup : `dict` [`int`, `float`]
        Coadd weight of each visit.

    Returns
    -------
    model : `numpy.ndarray`
        The DCR-smeared model, matching the appearance of the source in the
        coadd.
    unshiftedModel : `numpy.ndarray`
        The same stack with no DCR shift applied.
    flux : `float`
        Weighted mean flux of the source.

    Raises
    ------
    RuntimeError
        If none of the visits are present in ``weightLookup``.
    """
    models = []
    unshiftedModels = []
    weights = []
    bbox = None
    fluxes = []
    for visit in cutouts:
        flux = cutouts[visit]['modelFlux']
        if visit in weightLookup:
            weight = weightLookup[visit]
            bbox = cutouts[visit].getFootprint().getBBox()
        else:
            continue
        weights.append(weight)
        dcrFPs = dcrFootprints[visit]
        # dcrFPs is a list of the shifted footprints for all subfilters.
        # Stack each and weight with the fitted subfilter fraction.
        stack = [dcrFp.getFootprint().extractImage(bbox=bbox, fill=0).array*dcrFp['modelFlux']
                 for dcrFp in dcrFPs]
        models.append(np.sum(stack, axis=0)*flux*weight)
        # The un-shifted model is built on the same bbox, so that the two
        # stacks are pixel matched and can be used interchangeably.
        unshiftedVisit = unshifted[visit]
        if unshiftedVisit.shape != models[-1].shape:
            raise RuntimeError(f"Un-shifted model shape {unshiftedVisit.shape} does not match the "
                               f"shifted model shape {models[-1].shape} for visit {visit}.")
        unshiftedModels.append(unshiftedVisit*flux*weight)
        fluxes.append(flux*weight)
    if bbox is None:
        raise RuntimeError("None of the visits of this source are in the coadd.")
    weightSum = np.sum(weights)
    if not np.isfinite(weightSum) or weightSum <= 0:
        raise RuntimeError(f"The coadd weights of the visits of this source sum to {weightSum}, so the "
                           "models cannot be normalized.")
    return (np.sum(models, axis=0)/weightSum,
            np.sum(unshiftedModels, axis=0)/weightSum,
            np.sum(fluxes)/weightSum)


def getPsfMajorMinorAxes(psf, position=None, useFwhm=False):
    sigmaToFwhm = 2*np.log(2*np.sqrt(2))
    if position is None:
        position = psf.getAveragePosition()
    shape = psf.computeShape(position)
    trace = shape.getIxx() + shape.getIyy()
    diff = (shape.getIxx() - shape.getIyy())/2.
    det = np.sqrt(diff**2 + shape.getIxy()**2)

    lam_major = trace/2. + det
    lam_minor = trace/2. - det

    major_sigma = np.sqrt(lam_major) if lam_major > 0 else 0.
    minor_sigma = np.sqrt(lam_minor) if lam_minor > 0 else 0.
    if useFwhm:
        return (sigmaToFwhm*major_sigma, sigmaToFwhm*minor_sigma)
    else:
        return (major_sigma, minor_sigma)
