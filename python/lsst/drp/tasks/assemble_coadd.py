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

__all__ = (
    "AssembleCoaddConfig",
    "AssembleCoaddConnections",
    "AssembleCoaddTask",
)

import logging
import warnings

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.coadd.utils as coaddUtils
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.pipe.base as pipeBase
import numpy
from deprecated.sphinx import deprecated
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pipe.tasks.coaddBase import CoaddBaseTask, makeSkyInfo, subBBoxIter
from lsst.pipe.tasks.healSparseMapping import HealSparseInputMapTask
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.pipe.tasks.scaleZeroPoint import ScaleZeroPointTask
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod

log = logging.getLogger(__name__)


class AssembleCoaddConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "warpType": "direct",
        "warpTypeSuffix": "",
    },
):
    inputWarps = pipeBase.connectionTypes.Input(
        doc=(
            "Input list of warps to be assemebled i.e. stacked."
            "WarpType (e.g. direct, psfMatched) is controlled by the "
            "warpType config parameter"
        ),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded " "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    selectedVisits = pipeBase.connectionTypes.Input(
        doc="Selected visits to be coadded.",
        name="{outputCoaddName}Visits",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "tract", "patch", "skymap", "band"),
    )
    brightObjectMask = pipeBase.connectionTypes.PrerequisiteInput(
        doc=(
            "Input Bright Object Mask mask produced with external catalogs "
            "to be applied to the mask plane BRIGHT_OBJECT."
        ),
        name="brightObjectMask",
        storageClass="ObjectMaskCatalog",
        dimensions=("tract", "patch", "skymap", "band"),
        minimum=0,
    )
    coaddExposure = pipeBase.connectionTypes.Output(
        doc="Output coadded exposure, produced by stacking input warps",
        name="{outputCoaddName}Coadd{warpTypeSuffix}",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    nImage = pipeBase.connectionTypes.Output(
        doc="Output image of number of input images per pixel",
        name="{outputCoaddName}Coadd_nImage",
        storageClass="ImageU",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    inputMap = pipeBase.connectionTypes.Output(
        doc="Output healsparse map of input images",
        name="{outputCoaddName}Coadd_inputMap",
        storageClass="HealSparseMap",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.doMaskBrightObjects:
            self.prerequisiteInputs.remove("brightObjectMask")

        if not config.doSelectVisits:
            self.inputs.remove("selectedVisits")

        if not config.doNImage:
            self.outputs.remove("nImage")

        if not self.config.doInputMap:
            self.outputs.remove("inputMap")


class AssembleCoaddConfig(
    CoaddBaseTask.ConfigClass, pipeBase.PipelineTaskConfig, pipelineConnections=AssembleCoaddConnections
):
    warpType = pexConfig.Field(
        doc="Warp name: one of 'direct' or 'psfMatched'",
        dtype=str,
        default="direct",
    )
    subregionSize = pexConfig.ListField(
        dtype=int,
        doc="Width, height of stack subregion size; "
        "make small enough that a full stack of images will fit into memory "
        " at once.",
        length=2,
        default=(2000, 2000),
    )
    statistic = pexConfig.Field(
        dtype=str,
        doc="Main stacking statistic for aggregating over the epochs.",
        default="MEANCLIP",
    )
    doOnlineForMean = pexConfig.Field(
        dtype=bool,
        doc='Perform online coaddition when statistic="MEAN" to save memory?',
        default=False,
    )
    doSigmaClip = pexConfig.Field(
        dtype=bool,
        doc="Perform sigma clipped outlier rejection with MEANCLIP statistic?",
        deprecated=True,
        default=False,
    )
    sigmaClip = pexConfig.Field(
        dtype=float,
        doc="Sigma for outlier rejection; ignored if non-clipping statistic " "selected.",
        default=3.0,
    )
    clipIter = pexConfig.Field(
        dtype=int,
        doc="Number of iterations of outlier rejection; ignored if " "non-clipping statistic selected.",
        default=2,
    )
    calcErrorFromInputVariance = pexConfig.Field(
        dtype=bool,
        doc="Calculate coadd variance from input variance by stacking "
        "statistic. Passed to "
        "StatisticsControl.setCalcErrorFromInputVariance()",
        default=True,
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to adjust the photometric zero point of the coadd temp " "exposures",
    )
    doInterp = pexConfig.Field(
        doc="Interpolate over NaN pixels? Also extrapolate, if necessary, but " "the results are ugly.",
        dtype=bool,
        default=True,
    )
    interpImage = pexConfig.ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over NaN pixels",
    )
    doWrite = pexConfig.Field(
        doc="Persist coadd?",
        dtype=bool,
        default=True,
    )
    doNImage = pexConfig.Field(
        doc="Create image of number of contributing exposures for each pixel",
        dtype=bool,
        default=False,
    )
    doUsePsfMatchedPolygons = pexConfig.Field(
        doc="Use ValidPolygons from shrunk Psf-Matched Calexps? Should be set "
        "to True by CompareWarp only.",
        dtype=bool,
        default=False,
    )
    maskPropagationThresholds = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        doc=(
            "Threshold (in fractional weight) of rejection at which we "
            "propagate a mask plane to the coadd; that is, we set the mask "
            "bit on the coadd if the fraction the rejected frames "
            "would have contributed exceeds this value."
        ),
        default={"SAT": 0.1},
    )
    removeMaskPlanes = pexConfig.ListField(
        dtype=str,
        default=["NOT_DEBLENDED"],
        doc="Mask planes to remove before coadding",
    )
    doMaskBrightObjects = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Set mask and flag bits for bright objects?",
    )
    brightObjectMaskName = pexConfig.Field(
        dtype=str,
        default="BRIGHT_OBJECT",
        doc="Name of mask bit used for bright objects",
    )
    coaddPsf = pexConfig.ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=measAlg.CoaddPsfConfig,
    )
    doAttachTransmissionCurve = pexConfig.Field(
        dtype=bool,
        default=False,
        optional=False,
        doc=(
            "Attach a piecewise TransmissionCurve for the coadd? "
            "(requires all input Exposures to have TransmissionCurves)."
        ),
    )
    hasFakes = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Should be set to True if fake sources have been inserted into the input data.",
    )
    doSelectVisits = pexConfig.Field(
        doc="Coadd only visits selected by a SelectVisitsTask",
        dtype=bool,
        default=False,
    )
    doInputMap = pexConfig.Field(
        doc="Create a bitwise map of coadd inputs",
        dtype=bool,
        default=False,
    )
    inputMapper = pexConfig.ConfigurableField(
        doc="Input map creation subtask.",
        target=HealSparseInputMapTask,
    )

    def setDefaults(self):
        super().setDefaults()
        self.badMaskPlanes = ["NO_DATA", "BAD", "SAT", "EDGE"]

    def validate(self):
        super().validate()
        if self.doPsfMatch:  # TODO: Remove this in DM-39841
            # Backwards compatibility.
            # Configs do not have loggers
            log.warning("Config doPsfMatch deprecated. Setting warpType='psfMatched'")
            self.warpType = "psfMatched"
        if self.doSigmaClip and self.statistic != "MEANCLIP":
            log.warning('doSigmaClip deprecated. To replicate behavior, setting statistic to "MEANCLIP"')
            self.statistic = "MEANCLIP"
        if self.doInterp and self.statistic not in ["MEAN", "MEDIAN", "MEANCLIP", "VARIANCE", "VARIANCECLIP"]:
            raise ValueError(
                "Must set doInterp=False for statistic=%s, which does not "
                "compute and set a non-zero coadd variance estimate." % (self.statistic)
            )

        unstackableStats = ["NOTHING", "ERROR", "ORMASK"]
        if not hasattr(afwMath.Property, self.statistic) or self.statistic in unstackableStats:
            stackableStats = [
                str(k) for k in afwMath.Property.__members__.keys() if str(k) not in unstackableStats
            ]
            raise ValueError(
                "statistic %s is not allowed. Please choose one of %s." % (self.statistic, stackableStats)
            )


class AssembleCoaddTask(CoaddBaseTask, pipeBase.PipelineTask):
    """Assemble a coadded image from a set of warps.

    Each Warp that goes into a coadd will typically have an independent
    photometric zero-point. Therefore, we must scale each Warp to set it to
    a common photometric zeropoint. WarpType may be one of 'direct' or
    'psfMatched', and the boolean configs `config.makeDirect` and
    `config.makePsfMatched` set which of the warp types will be coadded.
    The coadd is computed as a mean with optional outlier rejection.
    Criteria for outlier rejection are set in `AssembleCoaddConfig`.
    Finally, Warps can have bad 'NaN' pixels which received no input from the
    source calExps. We interpolate over these bad (NaN) pixels.

    `AssembleCoaddTask` uses several sub-tasks. These are

    - `~lsst.pipe.tasks.ScaleZeroPointTask`
    - create and use an ``imageScaler`` object to scale the photometric
      zeropoint for each Warp
    - `~lsst.pipe.tasks.InterpImageTask`
    - interpolate across bad pixels (NaN) in the final coadd

    You can retarget these subtasks if you wish.

    Raises
    ------
    RuntimeError
        Raised if unable to define mask plane for bright objects.

    Notes
    -----
    Debugging:
    `AssembleCoaddTask` has no debug variables of its own. Some of the
    subtasks may support `~lsst.base.lsstDebug` variables. See the
    documentation for the subtasks for further information.
    """

    ConfigClass = AssembleCoaddConfig
    _DefaultName = "assembleCoadd"

    def __init__(self, *args, **kwargs):
        # TODO: DM-17415 better way to handle previously allowed passed args
        # e.g.`AssembleCoaddTask(config)`
        if args:
            argNames = ["config", "name", "parentTask", "log"]
            kwargs.update({k: v for k, v in zip(argNames, args)})
            warnings.warn(
                "AssembleCoadd received positional args, and casting them as kwargs: %s. "
                "PipelineTask will not take positional args" % argNames,
                FutureWarning,
                stacklevel=2,
            )

        super().__init__(**kwargs)
        self.makeSubtask("interpImage")
        self.makeSubtask("scaleZeroPoint")

        if self.config.doMaskBrightObjects:
            mask = afwImage.Mask()
            try:
                self.brightObjectBitmask = 1 << mask.addMaskPlane(self.config.brightObjectMaskName)
            except pexExceptions.LsstCppException:
                raise RuntimeError(
                    "Unable to define mask plane for bright objects; planes used are %s"
                    % mask.getMaskPlaneDict().keys()
                )
            del mask

        if self.config.doInputMap:
            self.makeSubtask("inputMapper")

        self.warpType = self.config.warpType

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputData = butlerQC.get(inputRefs)

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData
        # needs it
        skyMap = inputData["skyMap"]
        outputDataId = butlerQC.quantum.dataId

        inputData["skyInfo"] = makeSkyInfo(
            skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"]
        )

        if self.config.doSelectVisits:
            warpRefList = self.filterWarps(inputData["inputWarps"], inputData["selectedVisits"])
        else:
            warpRefList = inputData["inputWarps"]

        inputs = self.prepareInputs(warpRefList)
        self.log.info("Found %d %s", len(inputs.tempExpRefList), self.getTempExpDatasetName(self.warpType))
        if len(inputs.tempExpRefList) == 0:
            raise pipeBase.NoWorkFound("No coadd temporary exposures found")

        supplementaryData = self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)
        retStruct = self.run(
            inputData["skyInfo"],
            inputs.tempExpRefList,
            inputs.imageScalerList,
            inputs.weightList,
            supplementaryData=supplementaryData,
        )

        inputData.setdefault("brightObjectMask", None)
        if self.config.doMaskBrightObjects and inputData["brightObjectMask"] is None:
            log.warning("doMaskBrightObjects is set to True, but brightObjectMask not loaded")
        self.processResults(retStruct.coaddExposure, inputData["brightObjectMask"], outputDataId)

        if self.config.doWrite:
            butlerQC.put(retStruct, outputRefs)
        return retStruct

    def processResults(self, coaddExposure, brightObjectMasks=None, dataId=None):
        """Interpolate over missing data and mask bright stars.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The coadded exposure to process.
        brightObjectMasks : `lsst.afw.table` or `None`, optional
            Table of bright objects to mask.
        dataId : `lsst.daf.butler.DataId` or `None`, optional
            Data identification.
        """
        if self.config.doInterp:
            self.interpImage.run(coaddExposure.getMaskedImage(), planeName="NO_DATA")
            # The variance must be positive; work around for DM-3201.
            varArray = coaddExposure.variance.array
            with numpy.errstate(invalid="ignore"):
                varArray[:] = numpy.where(varArray > 0, varArray, numpy.inf)

        if self.config.doMaskBrightObjects:
            self.setBrightObjectMasks(coaddExposure, brightObjectMasks, dataId)

    def _makeSupplementaryData(self, butlerQC, inputRefs, outputRefs):
        """Make additional inputs to run() specific to subclasses (Gen3).

        Duplicates interface of `runQuantum` method.
        Available to be implemented by subclasses only if they need the
        coadd dataRef for performing preliminary processing before
        assembling the coadd.

        Parameters
        ----------
        butlerQC : `~lsst.pipe.base.ButlerQuantumContext`
            Gen3 Butler object for fetching additional data products before
            running the Task specialized for quantum being processed.
        inputRefs : `~lsst.pipe.base.InputQuantizedConnection`
            Attributes are the names of the connections describing input
            dataset types. Values are DatasetRefs that task consumes for the
            corresponding dataset type. DataIds are guaranteed to match data
            objects in ``inputData``.
        outputRefs : `~lsst.pipe.base.OutputQuantizedConnection`
            Attributes are the names of the connections describing output
            dataset types. Values are DatasetRefs that task is to produce
            for the corresponding dataset type.
        """
        return pipeBase.Struct()

    @deprecated(
        reason="makeSupplementaryDataGen3 is deprecated in favor of _makeSupplementaryData",
        version="v25.0",
        category=FutureWarning,
    )
    def makeSupplementaryDataGen3(self, butlerQC, inputRefs, outputRefs):
        return self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)

    def prepareInputs(self, refList):
        """Prepare the input warps for coaddition by measuring the weight for
        each warp and the scaling for the photometric zero point.

        Each Warp has its own photometric zeropoint and background variance.
        Before coadding these Warps together, compute a scale factor to
        normalize the photometric zeropoint and compute the weight for each
        Warp.

        Parameters
        ----------
        refList : `list`
            List of data references to tempExp.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``tempExprefList``
                `list` of data references to tempExp.
            ``weightList``
                `list` of weightings.
            ``imageScalerList``
                `list` of image scalers.
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        # compute tempExpRefList: a list of tempExpRef that actually exist
        # and weightList: a list of the weight of the associated coadd tempExp
        # and imageScalerList: a list of scale factors for the associated coadd
        # tempExp.
        tempExpRefList = []
        weightList = []
        imageScalerList = []
        tempExpName = self.getTempExpDatasetName(self.warpType)
        for tempExpRef in refList:
            tempExp = tempExpRef.get()
            # Ignore any input warp that is empty of data
            if numpy.isnan(tempExp.image.array).all():
                continue
            maskedImage = tempExp.getMaskedImage()
            imageScaler = self.scaleZeroPoint.computeImageScaler(
                exposure=tempExp,
                dataRef=tempExpRef,  # FIXME
            )
            try:
                imageScaler.scaleMaskedImage(maskedImage)
            except Exception as e:
                self.log.warning("Scaling failed for %s (skipping it): %s", tempExpRef.dataId, e)
                continue
            statObj = afwMath.makeStatistics(
                maskedImage.getVariance(), maskedImage.getMask(), afwMath.MEANCLIP, statsCtrl
            )
            meanVar, meanVarErr = statObj.getResult(afwMath.MEANCLIP)
            weight = 1.0 / float(meanVar)
            if not numpy.isfinite(weight):
                self.log.warning("Non-finite weight for %s: skipping", tempExpRef.dataId)
                continue
            self.log.info("Weight of %s %s = %0.3f", tempExpName, tempExpRef.dataId, weight)

            del maskedImage
            del tempExp

            tempExpRefList.append(tempExpRef)
            weightList.append(weight)
            imageScalerList.append(imageScaler)

        return pipeBase.Struct(
            tempExpRefList=tempExpRefList, weightList=weightList, imageScalerList=imageScalerList
        )

    def prepareStats(self, mask=None):
        """Prepare the statistics for coadding images.

        Parameters
        ----------
        mask : `int`, optional
            Bit mask value to exclude from coaddition.

        Returns
        -------
        stats : `~lsst.pipe.base.Struct`
            Statistics as a struct with attributes:

            ``statsCtrl``
                Statistics control object for coadd
                (`~lsst.afw.math.StatisticsControl`).
            ``statsFlags``
                Statistic for coadd (`~lsst.afw.math.Property`).
        """
        if mask is None:
            mask = self.getBadPixelMask()
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(mask)
        statsCtrl.setNanSafe(True)
        statsCtrl.setWeighted(True)
        statsCtrl.setCalcErrorFromInputVariance(self.config.calcErrorFromInputVariance)
        for plane, threshold in self.config.maskPropagationThresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        statsFlags = afwMath.stringToStatisticsProperty(self.config.statistic)
        return pipeBase.Struct(ctrl=statsCtrl, flags=statsFlags)

    @timeMethod
    def run(
        self,
        skyInfo,
        tempExpRefList,
        imageScalerList,
        weightList,
        altMaskList=None,
        mask=None,
        supplementaryData=None,
    ):
        """Assemble a coadd from input warps.

        Assemble the coadd using the provided list of coaddTempExps. Since
        the full coadd covers a patch (a large area), the assembly is
        performed over small areas on the image at a time in order to
        conserve memory usage. Iterate over subregions within the outer
        bbox of the patch using `assembleSubregion` to stack the corresponding
        subregions from the coaddTempExps with the statistic specified.
        Set the edge bits the coadd mask based on the weight map.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            Struct with geometric information about the patch.
        tempExpRefList : `list`
            List of data references to Warps (previously called CoaddTempExps).
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`, optional
            List of alternate masks to use rather than those stored with
            tempExp.
        mask : `int`, optional
            Bit mask value to exclude from coaddition.
        supplementaryData : `~lsst.pipe.base.Struct`, optional
            Struct with additional data products needed to assemble coadd.
            Only used by subclasses that implement ``_makeSupplementaryData``
            and override `run`.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``coaddExposure``
                Coadded exposure (`~lsst.afw.image.Exposure`).
            ``nImage``
                Exposure count image (`~lsst.afw.image.Image`), if requested.
            ``inputMap``
                Bit-wise map of inputs, if requested.
            ``warpRefList``
                Input list of refs to the warps
                (`~lsst.daf.butler.DeferredDatasetHandle`) (unmodified).
            ``imageScalerList``
                Input list of image scalers (`list`) (unmodified).
            ``weightList``
                Input list of weights (`list`) (unmodified).

        Raises
        ------
        lsst.pipe.base.NoWorkFound
            Raised if no data references are provided.
        """
        tempExpName = self.getTempExpDatasetName(self.warpType)
        self.log.info("Assembling %s %s", len(tempExpRefList), tempExpName)
        if not tempExpRefList:
            raise pipeBase.NoWorkFound("No exposures provided for co-addition.")

        stats = self.prepareStats(mask=mask)

        if altMaskList is None:
            altMaskList = [None] * len(tempExpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        coaddExposure.setPhotoCalib(self.scaleZeroPoint.getPhotoCalib())
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, tempExpRefList, weightList)
        coaddMaskedImage = coaddExposure.getMaskedImage()
        subregionSizeArr = self.config.subregionSize
        subregionSize = geom.Extent2I(subregionSizeArr[0], subregionSizeArr[1])
        # if nImage is requested, create a zero one which can be passed to
        # assembleSubregion.
        if self.config.doNImage:
            nImage = afwImage.ImageU(skyInfo.bbox)
        else:
            nImage = None
        # If inputMap is requested, create the initial version that can be
        # masked in assembleSubregion.
        if self.config.doInputMap:
            self.inputMapper.build_ccd_input_map(
                skyInfo.bbox, skyInfo.wcs, coaddExposure.getInfo().getCoaddInputs().ccds
            )

        if self.config.doOnlineForMean and self.config.statistic == "MEAN":
            try:
                self.assembleOnlineMeanCoadd(
                    coaddExposure,
                    tempExpRefList,
                    imageScalerList,
                    weightList,
                    altMaskList,
                    stats.ctrl,
                    nImage=nImage,
                )
            except Exception as e:
                self.log.exception("Cannot compute online coadd %s", e)
                raise
        else:
            for subBBox in subBBoxIter(skyInfo.bbox, subregionSize):
                try:
                    self.assembleSubregion(
                        coaddExposure,
                        subBBox,
                        tempExpRefList,
                        imageScalerList,
                        weightList,
                        altMaskList,
                        stats.flags,
                        stats.ctrl,
                        nImage=nImage,
                    )
                except Exception as e:
                    self.log.exception("Cannot compute coadd %s: %s", subBBox, e)
                    raise

        # If inputMap is requested, we must finalize the map after the
        # accumulation.
        if self.config.doInputMap:
            self.inputMapper.finalize_ccd_input_map_mask()
            inputMap = self.inputMapper.ccd_input_map
        else:
            inputMap = None

        self.setInexactPsf(coaddMaskedImage.getMask())
        # Despite the name, the following doesn't really deal with "EDGE"
        # pixels: it identifies pixels that didn't receive any unmasked inputs
        # (as occurs around the edge of the field).
        coaddUtils.setCoaddEdgeBits(coaddMaskedImage.getMask(), coaddMaskedImage.getVariance())
        return pipeBase.Struct(
            coaddExposure=coaddExposure,
            nImage=nImage,
            warpRefList=tempExpRefList,
            imageScalerList=imageScalerList,
            weightList=weightList,
            inputMap=inputMap,
        )

    def assembleMetadata(self, coaddExposure, tempExpRefList, weightList):
        """Set the metadata for the coadd.

        This basic implementation sets the filter from the first input.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        tempExpRefList : `list`
            List of data references to tempExp.
        weightList : `list`
            List of weights.

        Raises
        ------
        AssertionError
            Raised if there is a length mismatch.
        """
        assert len(tempExpRefList) == len(weightList), "Length mismatch"

        # We load a single pixel of each coaddTempExp, because we just want to
        # get at the metadata (and we need more than just the PropertySet that
        # contains the header), which is not possible with the current butler
        # (see #2777).
        bbox = geom.Box2I(coaddExposure.getBBox().getMin(), geom.Extent2I(1, 1))

        tempExpList = [tempExpRef.get(parameters={"bbox": bbox}) for tempExpRef in tempExpRefList]

        numCcds = sum(len(tempExp.getInfo().getCoaddInputs().ccds) for tempExp in tempExpList)

        # Set the coadd FilterLabel to the band of the first input exposure:
        # Coadds are calibrated, so the physical label is now meaningless.
        coaddExposure.setFilter(afwImage.FilterLabel(tempExpList[0].getFilter().bandLabel))
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()
        coaddInputs.ccds.reserve(numCcds)
        coaddInputs.visits.reserve(len(tempExpList))

        for tempExp, weight in zip(tempExpList, weightList):
            self.inputRecorder.addVisitToCoadd(coaddInputs, tempExp, weight)

        if self.config.doUsePsfMatchedPolygons:
            self.shrinkValidPolygons(coaddInputs)

        coaddInputs.visits.sort()
        coaddInputs.ccds.sort()
        if self.warpType == "psfMatched":
            # The modelPsf BBox for a psfMatchedWarp/coaddTempExp was
            # dynamically defined by ModelPsfMatchTask as the square box
            # bounding its spatially-variable, pre-matched WarpedPsf.
            # Likewise, set the PSF of a PSF-Matched Coadd to the modelPsf
            # having the maximum width (sufficient because square)
            modelPsfList = [tempExp.getPsf() for tempExp in tempExpList]
            modelPsfWidthList = [
                modelPsf.computeBBox(modelPsf.getAveragePosition()).getWidth() for modelPsf in modelPsfList
            ]
            psf = modelPsfList[modelPsfWidthList.index(max(modelPsfWidthList))]
        else:
            psf = measAlg.CoaddPsf(
                coaddInputs.ccds, coaddExposure.getWcs(), self.config.coaddPsf.makeControl()
            )
        coaddExposure.setPsf(psf)
        apCorrMap = measAlg.makeCoaddApCorrMap(
            coaddInputs.ccds, coaddExposure.getBBox(afwImage.PARENT), coaddExposure.getWcs()
        )
        coaddExposure.getInfo().setApCorrMap(apCorrMap)
        if self.config.doAttachTransmissionCurve:
            transmissionCurve = measAlg.makeCoaddTransmissionCurve(coaddExposure.getWcs(), coaddInputs.ccds)
            coaddExposure.getInfo().setTransmissionCurve(transmissionCurve)

    def assembleSubregion(
        self,
        coaddExposure,
        bbox,
        tempExpRefList,
        imageScalerList,
        weightList,
        altMaskList,
        statsFlags,
        statsCtrl,
        nImage=None,
    ):
        """Assemble the coadd for a sub-region.

        For each coaddTempExp, check for (and swap in) an alternative mask
        if one is passed. Remove mask planes listed in
        `config.removeMaskPlanes`. Finally, stack the actual exposures using
        `lsst.afw.math.statisticsStack` with the statistic specified by
        statsFlags. Typically, the statsFlag will be one of lsst.afw.math.MEAN
        for a mean-stack or `lsst.afw.math.MEANCLIP` for outlier rejection
        using an N-sigma clipped mean where N and iterations are specified by
        statsCtrl.  Assign the stacked subregion back to the coadd.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        bbox : `lsst.geom.Box`
            Sub-region to coadd.
        tempExpRefList : `list`
            List of data reference to tempExp.
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            tempExp, or None.  Each element is dict with keys = mask plane
            name to which to add the spans.
        statsFlags : `lsst.afw.math.Property`
            Property object for statistic for coadd.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        nImage : `lsst.afw.image.ImageU`, optional
            Keeps track of exposure count for each pixel.
        """
        self.log.debug("Computing coadd over %s", bbox)

        coaddExposure.mask.addMaskPlane("REJECTED")
        coaddExposure.mask.addMaskPlane("CLIPPED")
        coaddExposure.mask.addMaskPlane("SENSOR_EDGE")
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        maskedImageList = []
        if nImage is not None:
            subNImage = afwImage.ImageU(bbox.getWidth(), bbox.getHeight())
        for tempExpRef, imageScaler, altMask in zip(tempExpRefList, imageScalerList, altMaskList):
            exposure = tempExpRef.get(parameters={"bbox": bbox})

            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            imageScaler.scaleMaskedImage(maskedImage)

            # Add 1 for each pixel which is not excluded by the exclude mask.
            # In legacyCoadd, pixels may also be excluded by
            # afwMath.statisticsStack.
            if nImage is not None:
                subNImage.getArray()[maskedImage.getMask().getArray() & statsCtrl.getAndMask() == 0] += 1
            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)
            maskedImageList.append(maskedImage)

            if self.config.doInputMap:
                visit = exposure.getInfo().getCoaddInputs().visits[0].getId()
                self.inputMapper.mask_warp_bbox(bbox, visit, mask, statsCtrl.getAndMask())

        with self.timer("stack"):
            coaddSubregion = afwMath.statisticsStack(
                maskedImageList,
                statsFlags,
                statsCtrl,
                weightList,
                clipped,  # also set output to CLIPPED if sigma-clipped
                maskMap,
            )
        coaddExposure.maskedImage.assign(coaddSubregion, bbox)
        if nImage is not None:
            nImage.assign(subNImage, bbox)

    def assembleOnlineMeanCoadd(
        self, coaddExposure, tempExpRefList, imageScalerList, weightList, altMaskList, statsCtrl, nImage=None
    ):
        """Assemble the coadd using the "online" method.

        This method takes a running sum of images and weights to save memory.
        It only works for MEAN statistics.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        tempExpRefList : `list`
            List of data reference to tempExp.
        imageScalerList : `list`
            List of image scalers.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            tempExp, or None.  Each element is dict with keys = mask plane
            name to which to add the spans.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        nImage : `lsst.afw.image.ImageU`, optional
            Keeps track of exposure count for each pixel.
        """
        self.log.debug("Computing online coadd.")

        coaddExposure.mask.addMaskPlane("REJECTED")
        coaddExposure.mask.addMaskPlane("CLIPPED")
        coaddExposure.mask.addMaskPlane("SENSOR_EDGE")
        maskMap = self.setRejectedMaskMapping(statsCtrl)
        thresholdDict = AccumulatorMeanStack.stats_ctrl_to_threshold_dict(statsCtrl)

        bbox = coaddExposure.maskedImage.getBBox()

        stacker = AccumulatorMeanStack(
            coaddExposure.image.array.shape,
            statsCtrl.getAndMask(),
            mask_threshold_dict=thresholdDict,
            mask_map=maskMap,
            no_good_pixels_mask=statsCtrl.getNoGoodPixelsMask(),
            calc_error_from_input_variance=self.config.calcErrorFromInputVariance,
            compute_n_image=(nImage is not None),
        )

        for tempExpRef, imageScaler, altMask, weight in zip(
            tempExpRefList, imageScalerList, altMaskList, weightList
        ):
            exposure = tempExpRef.get()
            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            imageScaler.scaleMaskedImage(maskedImage)
            if self.config.removeMaskPlanes:
                self.removeMaskPlanes(maskedImage)

            stacker.add_masked_image(maskedImage, weight=weight)

            if self.config.doInputMap:
                visit = exposure.getInfo().getCoaddInputs().visits[0].getId()
                self.inputMapper.mask_warp_bbox(bbox, visit, mask, statsCtrl.getAndMask())

        stacker.fill_stacked_masked_image(coaddExposure.maskedImage)

        if nImage is not None:
            nImage.array[:, :] = stacker.n_image

    def removeMaskPlanes(self, maskedImage):
        """Unset the mask of an image for mask planes specified in the config.

        Parameters
        ----------
        maskedImage : `lsst.afw.image.MaskedImage`
            The masked image to be modified.

        Raises
        ------
        InvalidParameterError
            Raised if no mask plane with that name was found.
        """
        mask = maskedImage.getMask()
        for maskPlane in self.config.removeMaskPlanes:
            try:
                mask &= ~mask.getPlaneBitMask(maskPlane)
            except pexExceptions.InvalidParameterError:
                self.log.debug(
                    "Unable to remove mask plane %s: no mask plane with that name was found.", maskPlane
                )

    @staticmethod
    def setRejectedMaskMapping(statsCtrl):
        """Map certain mask planes of the warps to new planes for the coadd.

        If a pixel is rejected due to a mask value other than EDGE, NO_DATA,
        or CLIPPED, set it to REJECTED on the coadd.
        If a pixel is rejected due to EDGE, set the coadd pixel to SENSOR_EDGE.
        If a pixel is rejected due to CLIPPED, set the coadd pixel to CLIPPED.

        Parameters
        ----------
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.

        Returns
        -------
        maskMap : `list` of `tuple` of `int`
            A list of mappings of mask planes of the warped exposures to
            mask planes of the coadd.
        """
        edge = afwImage.Mask.getPlaneBitMask("EDGE")
        noData = afwImage.Mask.getPlaneBitMask("NO_DATA")
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        toReject = statsCtrl.getAndMask() & (~noData) & (~edge) & (~clipped)
        maskMap = [
            (toReject, afwImage.Mask.getPlaneBitMask("REJECTED")),
            (edge, afwImage.Mask.getPlaneBitMask("SENSOR_EDGE")),
            (clipped, clipped),
        ]
        return maskMap

    def applyAltMaskPlanes(self, mask, altMaskSpans):
        """Apply in place alt mask formatted as SpanSets to a mask.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Original mask.
        altMaskSpans : `dict`
            SpanSet lists to apply. Each element contains the new mask
            plane name (e.g. "CLIPPED and/or "NO_DATA") as the key,
            and list of SpanSets to apply to the mask.

        Returns
        -------
        mask : `lsst.afw.image.Mask`
            Updated mask.
        """
        if self.config.doUsePsfMatchedPolygons:
            if ("NO_DATA" in altMaskSpans) and ("NO_DATA" in self.config.badMaskPlanes):
                # Clear away any other masks outside the validPolygons. These
                # pixels are no longer contributing to inexact PSFs, and will
                # still be rejected because of NO_DATA.
                # self.config.doUsePsfMatchedPolygons should be True only in
                # CompareWarpAssemble. This mask-clearing step must only occur
                # *before* applying the new masks below.
                for spanSet in altMaskSpans["NO_DATA"]:
                    spanSet.clippedTo(mask.getBBox()).clearMask(mask, self.getBadPixelMask())

        for plane, spanSetList in altMaskSpans.items():
            maskClipValue = mask.addMaskPlane(plane)
            for spanSet in spanSetList:
                spanSet.clippedTo(mask.getBBox()).setMask(mask, 2**maskClipValue)
        return mask

    def shrinkValidPolygons(self, coaddInputs):
        """Shrink coaddInputs' ccds' ValidPolygons in place.

        Either modify each ccd's validPolygon in place, or if CoaddInputs
        does not have a validPolygon, create one from its bbox.

        Parameters
        ----------
        coaddInputs : `lsst.afw.image.coaddInputs`
            Original mask.
        """
        for ccd in coaddInputs.ccds:
            polyOrig = ccd.getValidPolygon()
            validPolyBBox = polyOrig.getBBox() if polyOrig else ccd.getBBox()
            validPolyBBox.grow(-self.config.matchingKernelSize // 2)
            if polyOrig:
                validPolygon = polyOrig.intersectionSingle(validPolyBBox)
            else:
                validPolygon = afwGeom.polygon.Polygon(geom.Box2D(validPolyBBox))
            ccd.setValidPolygon(validPolygon)

    def setBrightObjectMasks(self, exposure, brightObjectMasks, dataId=None):
        """Set the bright object masks.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            Exposure under consideration.
        brightObjectMasks : `lsst.afw.table`
            Table of bright objects to mask.
        dataId : `lsst.daf.butler.DataId`, optional
            Data identifier dict for patch.
        """
        if brightObjectMasks is None:
            self.log.warning("Unable to apply bright object mask: none supplied")
            return
        self.log.info("Applying %d bright object masks to %s", len(brightObjectMasks), dataId)
        mask = exposure.getMaskedImage().getMask()
        wcs = exposure.getWcs()
        plateScale = wcs.getPixelScale().asArcseconds()

        for rec in brightObjectMasks:
            center = geom.PointI(wcs.skyToPixel(rec.getCoord()))
            if rec["type"] == "box":
                assert rec["angle"] == 0.0, "Angle != 0 for mask object %s" % rec["id"]
                width = rec["width"].asArcseconds() / plateScale  # convert to pixels
                height = rec["height"].asArcseconds() / plateScale  # convert to pixels

                halfSize = geom.ExtentI(0.5 * width, 0.5 * height)
                bbox = geom.Box2I(center - halfSize, center + halfSize)

                bbox = geom.BoxI(
                    geom.PointI(int(center[0] - 0.5 * width), int(center[1] - 0.5 * height)),
                    geom.PointI(int(center[0] + 0.5 * width), int(center[1] + 0.5 * height)),
                )
                spans = afwGeom.SpanSet(bbox)
            elif rec["type"] == "circle":
                radius = int(rec["radius"].asArcseconds() / plateScale)  # convert to pixels
                spans = afwGeom.SpanSet.fromShape(radius, offset=center)
            else:
                self.log.warning("Unexpected region type %s at %s", rec["type"], center)
                continue
            spans.clippedTo(mask.getBBox()).setMask(mask, self.brightObjectBitmask)

    def setInexactPsf(self, mask):
        """Set INEXACT_PSF mask plane.

        If any of the input images isn't represented in the coadd (due to
        clipped pixels or chip gaps), the `CoaddPsf` will be inexact. Flag
        these pixels.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Coadded exposure's mask, modified in-place.
        """
        mask.addMaskPlane("INEXACT_PSF")
        inexactPsf = mask.getPlaneBitMask("INEXACT_PSF")
        sensorEdge = mask.getPlaneBitMask("SENSOR_EDGE")  # chip edges (so PSF is discontinuous)
        clipped = mask.getPlaneBitMask("CLIPPED")  # pixels clipped from coadd
        rejected = mask.getPlaneBitMask("REJECTED")  # pixels rejected from coadd due to masks
        array = mask.getArray()
        selected = array & (sensorEdge | clipped | rejected) > 0
        array[selected] |= inexactPsf

    def filterWarps(self, inputs, goodVisits):
        """Return list of only inputRefs with visitId in goodVisits ordered by
        goodVisit.

        Parameters
        ----------
        inputs : `list` of `~lsst.pipe.base.connections.DeferredDatasetRef`
            List of `lsst.pipe.base.connections.DeferredDatasetRef` with dataId
            containing visit.
        goodVisit : `dict`
            Dictionary with good visitIds as the keys. Value ignored.

        Returns
        -------
        filterInputs : `list` [`lsst.pipe.base.connections.DeferredDatasetRef`]
            Filtered and sorted list of inputRefs with visitId in goodVisits
            ordered by goodVisit.
        """
        inputWarpDict = {inputRef.ref.dataId["visit"]: inputRef for inputRef in inputs}
        filteredInputs = []
        for visit in goodVisits.keys():
            if visit in inputWarpDict:
                filteredInputs.append(inputWarpDict[visit])
        return filteredInputs


def countMaskFromFootprint(mask, footprint, bitmask, ignoreMask):
    """Function to count the number of pixels with a specific mask in a
    footprint.

    Find the intersection of mask & footprint. Count all pixels in the mask
    that are in the intersection that have bitmask set but do not have
    ignoreMask set. Return the count.

    Parameters
    ----------
    mask : `lsst.afw.image.Mask`
        Mask to define intersection region by.
    footprint : `lsst.afw.detection.Footprint`
        Footprint to define the intersection region by.
    bitmask : `Unknown`
        Specific mask that we wish to count the number of occurances of.
    ignoreMask : `Unknown`
        Pixels to not consider.

    Returns
    -------
    result : `int`
        Number of pixels in footprint with specified mask.
    """
    bbox = footprint.getBBox()
    bbox.clip(mask.getBBox(afwImage.PARENT))
    fp = afwImage.Mask(bbox)
    subMask = mask.Factory(mask, bbox, afwImage.PARENT)
    footprint.spans.setMask(fp, bitmask)
    return numpy.logical_and(
        (subMask.getArray() & fp.getArray()) > 0, (subMask.getArray() & ignoreMask) == 0
    ).sum()
