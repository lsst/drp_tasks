
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.signal.windows import hann

import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.geom as geom
from lsst.ip.diffim.dcrModel import calculateDcr, wavelengthGenerator
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
        doc=(
            "Input list of warps to be assembled i.e. stacked."
            "Note that this will often be different than the inputCoaddName."
            "WarpType (e.g. direct, psfMatched) is controlled by the warpType config parameter"
        ),
        name="{inputWarpName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    templateExposure = pipeBase.connectionTypes.Input(
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
    dcrCorrectionCatalog = pipeBase.connectionTypes.Output(
        doc="Output catalog of sub-band fluxes and footprints",
        name="{fakesType}dcr_correction_catalog",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    modelCatalog = pipeBase.connectionTypes.Output(
        doc="Model catalog of the stacked dcrCorrectionCatalog."
        "First subtract this model from the coadd template, then add the DCR"
        " correction.",
        name="{fakesType}dcr_model_catalog",
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
    effectiveWavelength = pexConfig.Field(
        doc="Effective wavelength of the filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
    )
    bandwidth = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
    )
    minimumSNR = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
        default=30,
    )
    maximumSNR = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
        default=1000,
    )
    minimumModelFraction = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
        default=0.15,
    )
    maximumModelFraction = pexConfig.Field(
        doc="Bandwidth of the physical filter, in nm."
        "Required if transmission curves aren't used."
        "Support for using transmission curves is to be added in DM-13668.",
        dtype=float,
        default=0.7,
    )
    footprintSize = pexConfig.Field(
        dtype=int,
        doc="Size of the footprints to calculate the DCR correctionin around objects.",
        default=35,
    )
    taperFootprint = pexConfig.Field(
        dtype=bool,
        doc="Weight the PSF model by a hanning window function to reduce edge artifacts.",
        default=True,
    )
    doWriteDcrResidual = pexConfig.Field(
        dtype=bool,
        doc="Write the residual coadd exposure after removing the DCR modeled sources.",
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
        centroidName = "base_SdssCentroid"
        control = measBase.SdssCentroidControl()
        self.schema.getAliasMap().set("slot_Centroid", centroidName)
        self.centroider = measBase.SdssCentroidAlgorithm(control, centroidName, self.schema)

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
        outputDataId = butlerQC.quantum.dataId

        skyInfo = makeSkyInfo(
            skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"]
        )

        # Construct list of input Deferred Datasets
        warpRefList = inputData["inputWarps"]

        inputs = self.prepareInputs(warpRefList, skyInfo.bbox)
        self.log.info("Found %d %s", len(inputs.warpRefList), self.getTempExpDatasetName(self.warpType))
        if len(inputs.warpRefList) == 0:
            self.log.warning("No coadd temporary exposures found")
            return

        retStruct = self.run(
            warpRefList=inputs.warpRefList,
            weightList=inputs.weightList,
            templateCoadd=inputs.templateCoadd,
            objectCatalog=inputs.objectCatalog,
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
            ``weightList``
                `list` of weightings.
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        # compute warpRefList: a list of warpRef that actually exist
        # and weightList: a list of the weight of the associated coadd warp
        warpRefList = []
        weightList = []

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
            statObj = afwMath.makeStatistics(
                maskedImage.getVariance(), maskedImage.getMask(), afwMath.MEANCLIP, statsCtrl
            )
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP)
            weight = 1.0 / float(meanVar)
            if not np.isfinite(weight):
                self.log.warning("Non-finite weight for %s: skipping", warpRef.dataId)
                continue
            self.log.info("Weight of warp %s = %0.3f", warpRef.dataId, weight)

            del maskedImage
            del warp

            warpRefList.append(warpRef)
            weightList.append(weight)

        return pipeBase.Struct(
            warpRefList=warpRefList,
            weightList=weightList,
        )

    def run(self, warpRefList, weightList, templateCoadd, objectCatalog):
        refCat = self.filter_object_catalog(objectCatalog)
        dcrFpLookupTable = {}
        cutoutLookupTable = {}
        recordVisitCount = {}
        for record in refCat:
            recId = record.getId()
            dcrFpLookupTable[recId] = {}
            cutoutLookupTable[recId] = {}
            recordVisitCount[recId] = 0
        for warp in warpRefList:
            visit = warp.visitInfo.getId()
            print(visit)
            # Generate a lookup table with the shifted PSF models for each
            # subfilter, and the image cutouts for each object in the catalog
            lookupTableSingle = self.make_warp_footprints(refCat, warp)
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
        # Drop any records that were removed from all visits
        # Note: should this be a configurable minimum instead? E.g. min 3?
        badRecords = np.array([recordVisitCount[record.getId()] == 0 for record in refCat])
        if np.any(badRecords):
            for badRec in refCat[badRecords]:
                recId = badRec.getId()
                dcrFpLookupTable.pop(recId)
                cutoutLookupTable.pop(recId)
                recordVisitCount.pop(recId)
            refCat = refCat[~badRecords].copy(deep=True)
        # Calculate one model per source
        results = self.calculateTemplateResidual(templateCoadd, dcrFpLookupTable, cutoutLookupTable)

        # Convert the lookup table to a source catalog with heavy footprints
        # containing the unshifted PSF model of the coadd at the source
        # location, and columns containing the overall flux and fractional flux
        # per subfilter 
        dcrCorrectionCatalog = self.make_dcr_catalog(refCat, dcrFpLookupTable, results.fluxLookupTable)
        return pipeBase.Struct(dcrResidual=results.dcrResidual,
                               dcrCorrectionCatalog=dcrCorrectionCatalog,
                               modelCatalog=results.template_models)

    def filter_object_catalog(self, objectCat):
        # Only include moderately bright objects
        # Faint objects won't have enough signal to fit DCR,
        # and bright objects will saturate the model in the bounding box
        # and create unwanted artifacts.
        snr = objectCat.getCalibInstFlux()/objectCat.getCalibInstFluxErr()
        goodSnr = snr > self.config.minimumSNR
        refCat = objectCat[snr > self.config.minimumSNR].copy(deep=True)
        # Exclude flagged objects that probably won't compute
        refCat = refCat[~refCat['base_SdssCentroid_flag']].copy(deep=True)
        goodCentroid = ~refCat['base_SdssCentroid_flag']
        refCat = refCat[~refCat['base_SdssShape_flag']].copy(deep=True)
        goodShape = ~refCat['base_SdssShape_flag']
        # Exclude extended objects
        refCat = refCat[refCat['base_ClassificationSizeExtendedness_value'] < 0.8].copy(deep=True)
        goodExtendedness = refCat['base_ClassificationSizeExtendedness_value'] < 0.5
        # Do not included deblended parents, only the children
        refCat = refCat[refCat['parent'] > 0].copy(deep=True)
        notParent = refCat['parent'] > 0
        # The source needs to fit in the defined footprint.
        # If it's larger, it's either trailed, extended, or just very bright
        # None of those cases will be fit well by the DCR model
        maxFootprintArea = (self.config.footprintSize - self.configfootprintBufferSize)**2
        goodArea = refCat['base_FootprintArea_value'] < maxFootprintArea
        srcUse = goodSnr & goodCentroid & goodShape & goodExtendedness & notParent & goodArea
        return refCat[srcUse].copy(deep=True)

    def initialize_dcr_catalog(self):
        cat = afwTable.SourceCatalog(self.schema)
        cat.defineCentroid(self.centroidName)
        return cat

    def make_dcr_catalog(self, refCat, dcrFpLookupTable, fluxLookupTable):
        dcrCorrectionCatalog = self.initialize_dcr_catalog()
        dcrGen = wavelengthGenerator(self.config.effectiveWavelength,
                                     self.config.bandwidth,
                                     self.config.dcrNumSubfilters)
        subfilterEffectiveWavelengths = [np.mean(wl) for wl in dcrGen]
        for refSrc in refCat:
            srcId = refSrc.getId()
            models = dcrFpLookupTable[srcId]
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

            for subfilter in range(self.config.dcrNumSubfilters):
                src[f'subfilterWeight_{subfilter}'] = model[subfilter]['modelFlux']
                src[f'subfilterWavelength_{subfilter}'] = subfilterEffectiveWavelengths[subfilter]
            
        return dcrCorrectionCatalog

    def make_warp_footprints(self, catalog, warp):
        dcrShift = calculateDcr(warp.visitInfo, warp.getWcs(),
                                self.config.effectiveWavelength,
                                self.config.bandwidth,
                                self.config.dcrNumSubfilters,
                                )
        image_footprints = self.initialize_dcr_catalog()
        fp_ctrl = afwDet.HeavyFootprintCtrl()
        lookupTable = {}
        boxSize = geom.Extent2I(self.config.footprintSize, self.config.footprintSize)
        if self.config.taperFootprint:
            windowFunction = np.outer(hann(self.config.footprintSize), hann(self.config.footprintSize))
            windowFunction /= np.max(windowFunction)
        else:
            windowFunction = np.ones((self.config.footprintSize, self.config.footprintSize))
        coreBoxSize = geom.Extent2I(self.config.footprintSize//2, self.config.footprintSize//2)
        for subfilter, shift in enumerate(dcrShift):
            windowFunction_shift = ndimage.shift(windowFunction, shift)
            # instantiate the catalog, and define the centroid
            cat = self.initialize_dcr_catalog()
            # Next define footprints
            for record in catalog:
                xc, yc = record.getCentroid()
                bbox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=boxSize)
                coreBBox = geom.Box2I.makeCenteredBox(center=record.getCentroid(), size=coreBoxSize)
                if np.any(warp[coreBBox].mask.array & warp.mask.getPlaneBitMask('NO_DATA')):
                    lookupTable[record.getId()] = None
                    continue
                spans = afwGeom.SpanSet(bbox)
                if subfilter == 0:
                    # Only fill the image catalog once
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
                    cutout = image_footprints.addNew()
                    cutout.setId(record.getId())
                    cutout["modelFlux"] = flux
                    cutout['base_SdssCentroid_x'] = xc
                    cutout['base_SdssCentroid_y'] = yc
                    foot = afwDet.Footprint(spans)
                    foot.addPeak(xc, yc, flux)
                    cutout_mi = warp[bbox].maskedImage.clone()
                    cutout_mi.image.array *= windowFunction
                    cutout.setFootprint(afwDet.HeavyFootprintF(foot, cutout_mi, fp_ctrl))
                    lookupTable[record.getId()] = {}
                    lookupTable[record.getId()]['cutout'] = cutout
                    lookupTable[record.getId()]['subfilterPsf'] = []
                else:
                    # Catch any records that were removed in an earlier 
                    # iteration
                    if lookupTable[record.getId()] is None:
                        continue
                # shift format is numpy (y,x)
                xc += shift[1]
                yc += shift[0]
                src = cat.addNew()
                src.setId(record.getId())
                subFlux = 1/self.config.dcrNumSubfilters
                src["modelFlux"] = subFlux
                src['base_SdssCentroid_x'] = xc
                src['base_SdssCentroid_y'] = yc
                foot = afwDet.Footprint(spans)
                foot.addPeak(xc, yc, subFlux)
                # Note, we don't just use 
                # afwImage.ImageF(warp.psf.computeImage(geom.Point2D(xc, yc)),
                #                 deep=True)
                # because we need the shifted bbox
                bbox2 = bbox.clippedTo(warp.psf.computeImageBBox(geom.Point2D(xc, yc)))
                psf_img = afwImage.ImageF(bbox)
                psf_img[bbox2].array[:, :] = warp.psf.computeImage(geom.Point2D(xc, yc))[bbox2].array
                psf_img.array *= windowFunction_shift
                psf_mask = afwImage.Mask(bbox)
                psf_variance = afwImage.ImageF(bbox)
                psf_mimage = afwImage.MaskedImageF(psf_img, psf_mask, psf_variance)
                
                heavy_fp = afwDet.HeavyFootprintF(foot, psf_mimage, fp_ctrl)
                src.setFootprint(heavy_fp)
                lookupTable[record.getId()]['subfilterPsf'].append(src)

        for record in catalog:
            recId = record.getId()
            if lookupTable[recId] is not None:
                image_fp = lookupTable[recId]['cutout']
                psf_fps = lookupTable[recId]['subfilterPsf']
                scales = self.minimize_footprint_residuals(image_fp, psf_fps)
                for psf_fp, scale in zip(psf_fps, scales):
                    psf_fp['modelFlux'] = scale
        return(lookupTable)

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
            scaleLookup[recId] = recScales/np.sum(recScales)
            # Update the modelFlux entries to be the same for all visits
            # for each record
            for visit in dcrFpLookupTableNew[recId]:
                for fp, scale in zip(dcrFpLookupTableNew[recId][visit], scaleLookup[recId]):
                    fp['modelFlux'] = scale
            model, flux = stack_dcr_footprints(dcrFpLookupTableNew[recId],
                                               cutoutLookupTable[recId],
                                               weightLookup
                                               )
            fluxLookupTable[recId] = flux
            # The bbox will be the same for all visits, so just grab the last one
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
            # The centroid will be the same for all visits, so just grab the last one
            xc = cutoutLookupTable[recId][visit]['base_SdssCentroid_x']
            yc = cutoutLookupTable[recId][visit]['base_SdssCentroid_y']
            cutout['base_SdssCentroid_x'] = xc
            cutout['base_SdssCentroid_y'] = yc
            foot = afwDet.Footprint(spans)
            foot.addPeak(xc, yc, flux)
            cutout.setFootprint(afwDet.HeavyFootprintF(foot, model, fp_ctrl))
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
    return (np.sum(models, axis=0)/np.sum(weights), np.sum(fluxes)/np.sum(weights))
