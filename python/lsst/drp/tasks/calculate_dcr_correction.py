
import numpy as np
from scipy import ndimage
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
        name="{fakesType}dcrCoadd{warpTypeSuffix}",
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
        default=1000,
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
        warpRefList = self.prepareInputs(inputData.pop("inputWarps"), skyInfo.bbox)
        self.log.info("Found %d input warps", len(warpRefList))
        if len(warpRefList) == 0:
            self.log.warning("No coadd temporary exposures found")
            return

        templateCoadd = inputData.pop("templateCoadd")
        objectCatalog = inputData.pop("objectCatalog")
        retStruct = self.run(
            warpRefList=warpRefList,
            templateCoadd=templateCoadd,
            objectCatalog=objectCatalog,
            effectiveWavelength=throughput.effectiveWavelength,
            bandwidth=throughput.bandwidth,
        )

        butlerQC.put(retStruct, outputRefs)
        return retStruct

    def prepareInputs(self, refList, coadd_bbox):
        """Prepare the input warps for coaddition by measuring the weight for
        each warp.

        Before coadding these Warps together compute the weight for each
        Warp.

        Parameters
        ----------
        refList : `list`
            List of dataset handles (data references) to warp.
        psfMatchedWarpRefList : `list` | None, optional
            List of dataset handles (data references) to psfMatchedWarp.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``warpRefList``
                `list` of dataset handles (data references) to warp.
        """
        # compute warpRefList: a list of warpRef that actually exist
        warpRefList = []

        for warpRef in refList:
            warp = warpRef.get(parameters={"bbox": coadd_bbox})
            # Ignore any input warp that is empty of data
            if np.isnan(warp.image.array).all():
                continue
            maskedImage = warp.getMaskedImage()

            if "BUNIT" not in warp.metadata:
                raise ValueError(f"Warp {warpRef.dataId} has no BUNIT metadata")
            if warp.metadata["BUNIT"] != "nJy":
                raise ValueError(
                    f"Warp {warpRef.dataId} has BUNIT {warp.metadata['BUNIT']}, expected nJy"
                )

            del maskedImage
            del warp

            warpRefList.append(warpRef)

        return warpRefList

    def run(self, warpRefList, templateCoadd, objectCatalog, effectiveWavelength, bandwidth):
        self.metadata['effectiveWavelength'] = effectiveWavelength
        self.metadata['bandwidth'] = bandwidth
        self.log.info("Dividing %fnm bandwidth into %d subfilters with %fnm effective wavelength",
                      bandwidth, self.config.dcrNumSubfilters, effectiveWavelength)
        refCat = self.filter_object_catalog(objectCatalog)
        dcrFpLookupTable = {}
        cutoutLookupTable = {}
        recordVisitCount = {}
        for record in refCat:
            recId = record.getId()
            dcrFpLookupTable[recId] = {}
            cutoutLookupTable[recId] = {}
            recordVisitCount[recId] = 0
        for warpRef in warpRefList:
            visit = warpRef.dataId['visit']
            # Generate a lookup table with the shifted PSF models for each
            # subfilter, and the image cutouts for each object in the catalog
            lookupTableSingle = self.make_warp_footprints(refCat, warpRef.get(),
                                                          effectiveWavelength, bandwidth)
            # Reformat the per-visit lookup table into two new tables with a
            # different ordering, both indexed by source record first and
            # having an inner lookup table over visit.
            # That way we can solve for the best fit to the subfilters across
            # all visits at once for a single source.
            for record in refCat:
                recId = record.getId()
                if lookupTableSingle[recId] is not None:
                    dcrFpLookupTable[recId][visit] = lookupTableSingle[recId]['subfilterPsf']
                    cutoutLookupTable[recId][visit] = lookupTableSingle[recId]['cutout']
                    recordVisitCount[recId] += 1
        # Drop any records that were removed from too many visits
        badRecords = np.array([recordVisitCount[record.getId()] < self.config.minNVisits
                              for record in refCat])
        if np.any(badRecords):
            for badRec in refCat[badRecords]:
                recId = badRec.getId()
                dcrFpLookupTable.pop(recId)
                cutoutLookupTable.pop(recId)
                recordVisitCount.pop(recId)
            refCat = refCat[~badRecords].copy(deep=True)
        self.log.info("Calculating DCR correction for %d surviving sources, and dropping %d sources",
                      len(refCat), np.sum(badRecords))
        # Calculate one model per source
        results = self.calculateTemplateResidual(templateCoadd, dcrFpLookupTable, cutoutLookupTable)

        # Convert the lookup table to a source catalog with heavy footprints
        # containing the unshifted PSF model of the coadd at the source
        # location, and columns containing the overall flux and fractional flux
        # per subfilter 
        dcrCorrectionCatalog = self.make_dcr_catalog(refCat, dcrFpLookupTable, results.fluxLookupTable,
                                                     results.template_models,
                                                     effectiveWavelength=effectiveWavelength,
                                                     bandwidth=bandwidth)
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
        snr = objectCat.getCalibInstFlux()/objectCat.getCalibInstFluxErr()
        goodSnr = (snr > self.config.minimumSNR) & (snr < self.config.maximumSNR)
        # Exclude flagged objects that probably won't compute
        goodCentroid = ~objectCat['base_SdssCentroid_flag']
        goodShape = ~objectCat['base_SdssShape_flag']
        # Exclude extended objects
        goodExtendedness = objectCat['base_ClassificationSizeExtendedness_value'] < 0.5
        # Do not included deblended parents, only the children
        notParent = objectCat['parent'] > 0
        # The source needs to fit in the defined footprint.
        # If it's larger, it's either trailed, extended, or just very bright
        # None of those cases will be fit well by the DCR model
        maxFootprintArea = self.config.footprintSize**2
        goodArea = objectCat['base_FootprintArea_value'] < maxFootprintArea
        srcUse = goodSnr & goodCentroid & goodShape & goodExtendedness & notParent & goodArea
        return objectCat[srcUse].copy(deep=True)

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

    def make_dcr_catalog(self, refCat, dcrFpLookupTable, fluxLookupTable, template_models,
                         effectiveWavelength=None, bandwidth=None):
        """Summary

        Parameters
        ----------
        refCat : `lsst.afw.table.SourceCatalog`
            Description
        dcrFpLookupTable : TYPE
            Description
        fluxLookupTable : TYPE
            Description
        effectiveWavelength : TYPE
            Description
        bandwidth : TYPE
            Description

        Returns
        -------
        dcrCorrectionCatalog : `lsst.afw.table.SourceCatalog`
            Description
        """
        dcrCorrectionCatalog = self.initialize_dcr_catalog()
        dcrGen = wavelengthGenerator(effectiveWavelength,
                                     bandwidth,
                                     self.config.dcrNumSubfilters)
        subfilterEffectiveWavelengths = [np.mean(wl) for wl in dcrGen]
        for refSrc in refCat:
            srcId = refSrc.getId()
            if srcId not in fluxLookupTable:
                continue
            models = dcrFpLookupTable[srcId]
            templateModel = template_models[srcId]
            visits = [visit for visit in models]
            # At this point the subfilter fractions are the same for each visit
            # so we can take the values from the first visit
            model = models[visits[0]]
            src = dcrCorrectionCatalog.addNew()
            src.setId(srcId)
            src['numSubfilters'] = self.config.dcrNumSubfilters
            src['modelFlux'] = fluxLookupTable[srcId]
            src['coord_ra'] = refSrc['coord_ra']
            src['coord_dec'] = refSrc['coord_dec']
            src['base_SdssCentroid_x'], src['base_SdssCentroid_y'] = refSrc.getCentroid()
            src.setFootprint(templateModel.getFootprint())

            for subfilter in range(self.config.dcrNumSubfilters):
                src[f'subfilterWeight_{subfilter}'] = model[subfilter]['modelFlux']
                src[f'subfilterWavelength_{subfilter}'] = subfilterEffectiveWavelengths[subfilter]
            
        return dcrCorrectionCatalog

    def make_warp_footprints(self, catalog, warp, effectiveWavelength, bandwidth):
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
        self.update_subfilter_psf_lookup_table(lookupTable, catalog, warp, effectiveWavelength, bandwidth,
                                               windowFunction=windowFunction, fp_ctrl=fp_ctrl)
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
        return(lookupTable)

    def update_subfilter_psf_lookup_table(self, lookupTable, catalog, warp, effectiveWavelength, bandwidth,
                                          windowFunction=None, fp_ctrl=afwDet.HeavyFootprintCtrl()):
        dcrShift = calculateDcr(warp.visitInfo, warp.getWcs(),
                                effectiveWavelength,
                                bandwidth,
                                self.config.dcrNumSubfilters,
                                )
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
                bbox2 = bbox.clippedTo(warp.psf.computeImageBBox(geom.Point2D(xc, yc)))
                psf_img = afwImage.ImageF(bbox)
                psf_img[bbox2].array[:, :] = warp.psf.computeImage(geom.Point2D(xc, yc))[bbox2].array
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
        scales0 = [image_fp['modelFlux']*psf_fp['modelFlux'] for psf_fp in psf_fps]
        nSubfilters = len(psf_fps)
        img = image_fp.getFootprint().extractImage().array
        psf_arrays = [psf.getFootprint().extractImage().array for psf in psf_fps]

        def minimize_residual(scales):
            residual = img.copy()
            for psf, scale in zip(psf_arrays, scales):
                residual -= scale*psf
            return np.std(residual)
        minFluxFit = self.config.minimumModelFraction*image_fp['modelFlux']
        maxFluxFit = self.config.maximumModelFraction*image_fp['modelFlux']
        scaleFit = least_squares(minimize_residual, scales0,
                                 bounds=[[minFluxFit]*nSubfilters, [maxFluxFit]*nSubfilters])
        scales = [scale/image_fp['modelFlux'] for scale in scaleFit.x]
        return scales

    def calculateTemplateResidual(self, templateCoadd, dcrFpLookupTable, cutoutLookupTable):
        inputs = templateCoadd.getInfo().getCoaddInputs()
        weightLookup = {}
        for visit in inputs.ccds['visit']:
            inds = inputs.ccds['visit'] == visit
            weightLookup[visit] = np.mean(inputs.ccds['weight'][inds])
        scaleLookup = {}
        dcrFpLookupTableNew = dcrFpLookupTable.copy()

        template_models = self.initialize_dcr_catalog()
        fp_ctrl = afwDet.HeavyFootprintCtrl()
        residual = templateCoadd.clone()
        # modelExposure = templateCoadd.clone()
        # modelExposure.image.array *= 0
        fluxLookupTable = {}
        for recId in dcrFpLookupTable:
            scales = []
            for visit in dcrFpLookupTable[recId]:
                scales.append([fp['modelFlux'] for fp in dcrFpLookupTable[recId][visit]])
            recScales = np.median(scales, axis=0)
            scalesSingle = recScales/np.sum(recScales)
            # Update the modelFlux entries to be the same for all visits
            # for each record
            for visit in dcrFpLookupTableNew[recId]:
                for fp, scale in zip(dcrFpLookupTableNew[recId][visit], scalesSingle):
                    fp['modelFlux'] = scale
            try:
                model, flux = stack_dcr_footprints(dcrFpLookupTableNew[recId],
                                                   cutoutLookupTable[recId],
                                                   weightLookup
                                                   )
            except RuntimeError:
                continue
            fluxLookupTable[recId] = flux
            scaleLookup[recId] = scalesSingle
            # The bbox will be the same for all visits,
            # so just grab the last one
            bbox = cutoutLookupTable[recId][visit].getFootprint().getBBox()
            spans = afwGeom.SpanSet(bbox)
            # modelExposure[bbox].image.array += model
            residual[bbox].image.array -= model

            # Add the heavy footprint of the stacked source
            # This can be subtracted from the original coadd, so that the DCR
            # model can be added in its place without having to store the
            # dcrResidual image
            cutout = template_models.addNew()
            cutout.setId(recId)
            cutout["modelFlux"] = flux
            # The centroid will be the same for all visits,
            # so just grab the last one
            xc = cutoutLookupTable[recId][visit]['base_SdssCentroid_x']
            yc = cutoutLookupTable[recId][visit]['base_SdssCentroid_y']
            cutout['base_SdssCentroid_x'] = xc
            cutout['base_SdssCentroid_y'] = yc
            foot = afwDet.Footprint(spans)
            foot.addPeak(xc, yc, flux)
            model_mi = templateCoadd[bbox].maskedImage.clone()
            model_mi.image.array = model
            cutout.setFootprint(afwDet.HeavyFootprintF(foot, model_mi, fp_ctrl))
        # return(modelExposure, residual, fluxLookupTable)
        return pipeBase.Struct(residual=residual,
                               fluxLookupTable=fluxLookupTable,
                               template_models=template_models,
                               )


def fit_footprints(model, image):
    model_flat = model.ravel()
    image_flat = image.ravel()
    cov = np.cov(image_flat*model_flat, model_flat*model_flat)[0, 1]
    varM = np.var(model**2)
    scale = cov / varM
    return scale


def stack_dcr_footprints(dcrFootprints, cutouts, weightLookup):
    models = []
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
        fluxes.append(flux*weight)
    if bbox is None:
        raise RuntimeError
    return (np.sum(models, axis=0)/np.sum(weights), np.sum(fluxes)/np.sum(weights))
