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

from __future__ import annotations

__all__ = [
    "AssembleCoaddTask",
    "AssembleCoaddConnections",
    "AssembleCoaddConfig",
    "CompareWarpAssembleCoaddTask",
    "CompareWarpAssembleCoaddConfig",
]

import copy
import logging
import warnings

import lsstDebug
import numpy

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.coadd.utils as coaddUtils
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
import lsst.pex.config as pexConfig
import lsst.pex.exceptions as pexExceptions
import lsst.pipe.base as pipeBase
import lsst.utils as utils
from lsst.meas.algorithms import AccumulatorMeanStack, MaskStreaksTask, ScaleVarianceTask, SourceDetectionTask
from lsst.pipe.tasks.coaddBase import (
    CoaddBaseTask,
    makeSkyInfo,
    removeMaskPlanes,
    reorderAndPadList,
    setRejectedMaskMapping,
    subBBoxIter,
)
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

        if not self.config.doWriteArtifactMasks:
            self.outputs.remove("artifactMasks")


class AssembleCoaddConfig(
    CoaddBaseTask.ConfigClass, pipeBase.PipelineTaskConfig, pipelineConnections=AssembleCoaddConnections
):
    warpType = pexConfig.ChoiceField(
        doc="Warp name: one of 'direct' or 'psfMatched'",
        dtype=str,
        default="direct",
        allowed={
            "direct": "Weighted mean of directWarps, with outlier rejection",
            "psfMatched": "Weighted mean of PSF-matched warps",
        },
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
    doScaleZeroPoint = pexConfig.Field(
        dtype=bool,
        doc="Scale the photometric zero point of the coadd temp exposures "
        "such that the magnitude zero point results in a flux in nJy.",
        default=False,
        deprecated="Now that visits are scaled to nJy it is no longer necessary or "
        "recommended to scale the zero point, so this will be removed "
        "after v29.",
    )
    scaleZeroPoint = pexConfig.ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to adjust the photometric zero point of the coadd temp " "exposures",
        deprecated="Now that visits are scaled to nJy it is no longer necessary or "
        "recommended to scale the zero point, so this will be removed "
        "after v29.",
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
    doWriteArtifactMasks = pexConfig.Field(
        doc="Persist artifact masks? Should be True for CompareWarp only.",
        dtype=bool,
        default=False,
    )
    doNImage = pexConfig.Field(
        doc="Create image of number of contributing exposures for each pixel",
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
        self.coaddPsf.cacheSize = 0

    def validate(self):
        super().validate()
        if self.doInterp and self.statistic not in ["MEAN", "MEDIAN", "MEANCLIP", "VARIANCE", "VARIANCECLIP"]:
            raise pexConfig.FieldValidationError(
                self.__class__.doInterp,
                self,
                f"Must set doInterp=False for statistic={self.statistic}, which does not "
                "compute and set a non-zero coadd variance estimate.",
            )

        unstackableStats = ["NOTHING", "ERROR", "ORMASK"]
        if not hasattr(afwMath.Property, self.statistic) or self.statistic in unstackableStats:
            stackableStats = [
                str(k) for k in afwMath.Property.__members__.keys() if str(k) not in unstackableStats
            ]
            raise pexConfig.FieldValidationError(
                self.__class__.statistic,
                self,
                f"statistic {self.statistic} is not allowed. Please choose one of {stackableStats}.",
            )

        # Admittedly, it's odd for a parent class to condition on a child class
        # but such is the case until the CompareWarp refactor in DM-38630.
        if self.doWriteArtifactMasks and not isinstance(self, CompareWarpAssembleCoaddConfig):
            raise pexConfig.FieldValidationError(
                self.__class__.doWriteArtifactMasks,
                self,
                "doWriteArtifactMasks is only valid for CompareWarpAssembleCoaddConfig.",
            )


class AssembleCoaddTask(CoaddBaseTask, pipeBase.PipelineTask):
    """Assemble a coadded image from a set of warps.

    Each Warp that goes into a coadd will have its flux calibrated to
    nJy. WarpType may be one of 'direct' or
    'psfMatched', and the boolean configs `config.makeDirect` and
    `config.makePsfMatched` set which of the warp types will be coadded.
    The coadd is computed as a mean with optional outlier rejection.
    Criteria for outlier rejection are set in `AssembleCoaddConfig`.
    Finally, Warps can have bad 'NaN' pixels which received no input from the
    source calExps. We interpolate over these bad (NaN) pixels.

    `AssembleCoaddTask` uses several sub-tasks. These are

    - `~lsst.pipe.tasks.ScaleZeroPointTask`
    - create and use an ``imageScaler`` object to scale the photometric
      zeropoint for each Warp (deprecated and will be removed in DM-49083).
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

    _doUsePsfMatchedPolygons: bool = False
    """Use ValidPolygons from shrunk Psf-Matched Calexps?

    This needs to be set to True by child classes that use compare Psf-Matched
    warps to non Psf-Matched warps.
    """

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
        if self.config.doScaleZeroPoint:
            # Remove completely in DM-49083
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

        # Although psfMatchedWarps are specifically required by only by a
        # specific set of a subclass, and since runQuantum is not overridden,
        # the selection and filtering must happen here on a conditional basis.
        # Otherwise, the elements in the various lists will not line up.
        # This design cries out for a refactor, which is planned in DM-38630.
        if self._doUsePsfMatchedPolygons:
            if self.config.doSelectVisits:
                psfMatchedWarpRefList = self.filterWarps(
                    inputData["psfMatchedWarps"],
                    inputData["selectedVisits"],
                )
            else:
                psfMatchedWarpRefList = inputData["psfMatchedWarps"]
        else:
            psfMatchedWarpRefList = []

        inputs = self.prepareInputs(warpRefList, inputData["skyInfo"].bbox, psfMatchedWarpRefList)
        self.log.info("Found %d %s", len(inputs.warpRefList), self.getTempExpDatasetName(self.warpType))
        if len(inputs.warpRefList) == 0:
            raise pipeBase.NoWorkFound("No coadd temporary exposures found")

        supplementaryData = self._makeSupplementaryData(butlerQC, inputRefs, outputRefs)
        retStruct = self.run(
            inputData["skyInfo"],
            warpRefList=inputs.warpRefList,
            imageScalerList=inputs.imageScalerList,
            weightList=inputs.weightList,
            psfMatchedWarpRefList=inputs.psfMatchedWarpRefList,
            supplementaryData=supplementaryData,
        )

        inputData.setdefault("brightObjectMask", None)
        if self.config.doMaskBrightObjects and inputData["brightObjectMask"] is None:
            log.warning("doMaskBrightObjects is set to True, but brightObjectMask not loaded")
        self.processResults(retStruct.coaddExposure, inputData["brightObjectMask"], outputDataId)

        if self.config.doWriteArtifactMasks:
            artifactMasksRefList = reorderAndPadList(
                outputRefs.artifactMasks,
                [ref.dataId for ref in outputRefs.artifactMasks],
                [ref.dataId for ref in inputs.warpRefList],
            )
            for altMask, warpRef, outputRef in zip(
                retStruct.altMaskList, inputs.warpRefList, artifactMasksRefList, strict=True
            ):
                mask = warpRef.get(component="mask", parameters={"bbox": retStruct.coaddExposure.getBBox()})
                self.applyAltMaskPlanes(mask, altMask)
                butlerQC.put(mask, outputRef)

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
        coadd dataRef/handle for performing preliminary processing before
        assembling the coadd.

        Parameters
        ----------
        butlerQC : `~lsst.pipe.base.QuantumContext`
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

    def prepareInputs(self, refList, coadd_bbox, psfMatchedWarpRefList=None):
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
            ``imageScalerList``
                `list` of image scalers.
                Deprecated and will be removed in DM-49083.
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumSigmaClip(self.config.sigmaClip)
        statsCtrl.setNumIter(self.config.clipIter)
        statsCtrl.setAndMask(self.getBadPixelMask())
        statsCtrl.setNanSafe(True)
        # compute warpRefList: a list of warpRef that actually exist
        # and weightList: a list of the weight of the associated coadd warp
        # and imageScalerList: a list of scale factors for the associated coadd
        # warp. (deprecated and will be removed in DM-49083)
        warpRefList = []
        weightList = []
        # Remove in DM-49083
        imageScalerList = []
        outputPsfMatchedWarpRefList = []

        # Convert psfMatchedWarpRefList to a dict, so we can look them up by
        # their dataId.
        # Note: Some warps may not have a corresponding psfMatchedWarp, which
        # could have happened due to a failure in the PSF matching, or not
        # having enough good pixels to support the PSF-matching kernel.

        if psfMatchedWarpRefList is None:
            psfMatchedWarpRefDict = {ref.dataId: None for ref in refList}
        else:
            psfMatchedWarpRefDict = {ref.dataId: ref for ref in psfMatchedWarpRefList}

        warpName = self.getTempExpDatasetName(self.warpType)
        for warpRef in refList:
            warp = warpRef.get(parameters={"bbox": coadd_bbox})
            # Ignore any input warp that is empty of data
            if numpy.isnan(warp.image.array).all():
                continue
            maskedImage = warp.getMaskedImage()

            # Allow users to scale the zero point for backward
            # compatibility. Remove imageScalarList in DM-49083.
            if self.config.doScaleZeroPoint:
                imageScaler = self.scaleZeroPoint.computeImageScaler(
                    exposure=warp,
                    dataRef=warpRef,  # FIXME
                )
                try:
                    imageScaler.scaleMaskedImage(maskedImage)
                except Exception as e:
                    self.log.warning("Scaling failed for %s (skipping it): %s", warpRef.dataId, e)
                    continue
                imageScalerList.append(imageScaler)
            else:
                imageScalerList.append(None)
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
            if not numpy.isfinite(weight):
                self.log.warning("Non-finite weight for %s: skipping", warpRef.dataId)
                continue
            self.log.info("Weight of %s %s = %0.3f", warpName, warpRef.dataId, weight)

            del maskedImage
            del warp

            warpRefList.append(warpRef)
            weightList.append(weight)

            dataId = warpRef.dataId
            psfMatchedWarpRef = psfMatchedWarpRefDict.get(dataId, None)
            outputPsfMatchedWarpRefList.append(psfMatchedWarpRef)

        return pipeBase.Struct(
            warpRefList=warpRefList,
            weightList=weightList,
            imageScalerList=imageScalerList,
            psfMatchedWarpRefList=outputPsfMatchedWarpRefList,
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
        *,
        warpRefList,
        imageScalerList,
        weightList,
        psfMatchedWarpRefList=None,
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
        warpRefList : `list`
            List of dataset handles (data references) to Warps
            (previously called CoaddTempExps).
        imageScalerList : `list`
            List of image scalers. Deprecated and will be removed after v29
            in DM-49083.
        weightList : `list`
            List of weights.
        psfMatchedWarpRefList : `list`, optional
            List of dataset handles (data references) to psfMatchedWarps.
        altMaskList : `list`, optional
            List of alternate masks to use rather than those stored with
            warp.
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
                Input list of dataset handles (data refs) to the warps
                (`~lsst.daf.butler.DeferredDatasetHandle`) (unmodified).
            ``imageScalerList``
                Input list of image scalers (`list`) (unmodified).
                Deprecated and will be removed after v29 in DM-49083.
            ``weightList``
                Input list of weights (`list`) (unmodified).

        Raises
        ------
        lsst.pipe.base.NoWorkFound
            Raised if no dataset handles (data references) are provided.
        """
        warpName = self.getTempExpDatasetName(self.warpType)
        self.log.info("Assembling %s %s", len(warpRefList), warpName)
        if not warpRefList:
            raise pipeBase.NoWorkFound("No exposures provided for co-addition.")

        stats = self.prepareStats(mask=mask)

        if altMaskList is None:
            altMaskList = [None] * len(warpRefList)

        coaddExposure = afwImage.ExposureF(skyInfo.bbox, skyInfo.wcs)
        # Deprecated, keep only the `else` branch in DM-49083.
        if self.config.doScaleZeroPoint:
            coaddExposure.setPhotoCalib(self.scaleZeroPoint.getPhotoCalib())
        else:
            coaddExposure.setPhotoCalib(afwImage.PhotoCalib(1.0))
        coaddExposure.getInfo().setCoaddInputs(self.inputRecorder.makeCoaddInputs())
        self.assembleMetadata(coaddExposure, warpRefList, weightList, psfMatchedWarpRefList)
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
                    warpRefList,
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
                        warpRefList,
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
            warpRefList=warpRefList,
            imageScalerList=imageScalerList,
            weightList=weightList,
            inputMap=inputMap,
            altMaskList=altMaskList,
        )

    def assembleMetadata(self, coaddExposure, warpRefList, weightList, psfMatchedWarpRefList=None):
        """Set the metadata for the coadd.

        This basic implementation sets the filter from the first input.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        warpRefList : `list`
            List of dataset handles (data references) to warp.
        weightList : `list`
            List of weights.
        psfMatchedWarpRefList : `list` | None, optional
            List of dataset handles (data references) to psfMatchedWarps.

        Raises
        ------
        AssertionError
            Raised if there is a length mismatch.
        """
        assert len(warpRefList) == len(weightList), "Length mismatch"

        if psfMatchedWarpRefList:
            assert len(warpRefList) == len(psfMatchedWarpRefList), "Length mismatch"

        # We load a single pixel of each coaddTempExp, because we just want to
        # get at the metadata (and we need more than just the PropertySet that
        # contains the header), which is not possible with the current butler
        # (see #2777).
        bbox = geom.Box2I(coaddExposure.getBBox().getMin(), geom.Extent2I(1, 1))

        warpList = [warpRef.get(parameters={"bbox": bbox}) for warpRef in warpRefList]

        numCcds = sum(len(warp.getInfo().getCoaddInputs().ccds) for warp in warpList)

        # Set the coadd FilterLabel to the band of the first input exposure:
        # Coadds are calibrated, so the physical label is now meaningless.
        coaddExposure.setFilter(afwImage.FilterLabel(warpList[0].getFilter().bandLabel))
        coaddInputs = coaddExposure.getInfo().getCoaddInputs()
        coaddInputs.ccds.reserve(numCcds)
        coaddInputs.visits.reserve(len(warpList))

        # Set the exposure units to nJy
        # No need to check after DM-49083.
        if not self.config.doScaleZeroPoint:
            coaddExposure.metadata["BUNIT"] = "nJy"

        # psfMatchedWarpRefList should be empty except in CompareWarpCoadd.
        if self._doUsePsfMatchedPolygons:
            # Set validPolygons for warp before addVisitToCoadd
            self._setValidPolygons(warpList, psfMatchedWarpRefList)

        for warp, weight in zip(warpList, weightList):
            self.inputRecorder.addVisitToCoadd(coaddInputs, warp, weight)

        coaddInputs.visits.sort()
        coaddInputs.ccds.sort()
        if self.warpType == "psfMatched":
            # The modelPsf BBox for a psfMatchedWarp/coaddTempExp was
            # dynamically defined by ModelPsfMatchTask as the square box
            # bounding its spatially-variable, pre-matched WarpedPsf.
            # Likewise, set the PSF of a PSF-Matched Coadd to the modelPsf
            # having the maximum width (sufficient because square)
            modelPsfList = [warp.getPsf() for warp in warpList]
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
        warpRefList,
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
        warpRefList : `list`
            List of dataset handles (data references) to warp.
        imageScalerList : `list`
            List of image scalers.
            Deprecated and will be removed after v29 in DM-49083.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            warp, or None.  Each element is dict with keys = mask plane
            name to which to add the spans.
        statsFlags : `lsst.afw.math.Property`
            Property object for statistic for coadd.
        statsCtrl : `lsst.afw.math.StatisticsControl`
            Statistics control object for coadd.
        nImage : `lsst.afw.image.ImageU`, optional
            Keeps track of exposure count for each pixel.
        """
        self.log.info("Stacking subregion %s", bbox)

        coaddExposure.mask.addMaskPlane("REJECTED")
        coaddExposure.mask.addMaskPlane("CLIPPED")
        coaddExposure.mask.addMaskPlane("SENSOR_EDGE")
        maskMap = setRejectedMaskMapping(statsCtrl)
        clipped = afwImage.Mask.getPlaneBitMask("CLIPPED")
        maskedImageList = []
        if nImage is not None:
            subNImage = afwImage.ImageU(bbox.getWidth(), bbox.getHeight())
        for warpRef, imageScaler, altMask in zip(warpRefList, imageScalerList, altMaskList):
            exposure = warpRef.get(parameters={"bbox": bbox})

            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            # Remove in DM-49083
            if imageScaler is not None:
                imageScaler.scaleMaskedImage(maskedImage)

            # Add 1 for each pixel which is not excluded by the exclude mask.
            # In legacyCoadd, pixels may also be excluded by
            # afwMath.statisticsStack.
            if nImage is not None:
                subNImage.getArray()[maskedImage.getMask().getArray() & statsCtrl.getAndMask() == 0] += 1
            if self.config.removeMaskPlanes:
                removeMaskPlanes(maskedImage.mask, self.config.removeMaskPlanes, logger=self.log)
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
        self, coaddExposure, warpRefList, imageScalerList, weightList, altMaskList, statsCtrl, nImage=None
    ):
        """Assemble the coadd using the "online" method.

        This method takes a running sum of images and weights to save memory.
        It only works for MEAN statistics.

        Parameters
        ----------
        coaddExposure : `lsst.afw.image.Exposure`
            The target exposure for the coadd.
        warpRefList : `list`
            List of dataset handles (data references) to warp.
        imageScalerList : `list`
            List of image scalers.
            Deprecated and will be removed after v29 in DM-49083.
        weightList : `list`
            List of weights.
        altMaskList : `list`
            List of alternate masks to use rather than those stored with
            warp, or None.  Each element is dict with keys = mask plane
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
        maskMap = setRejectedMaskMapping(statsCtrl)
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

        for warpRef, imageScaler, altMask, weight in zip(
            warpRefList, imageScalerList, altMaskList, weightList
        ):
            exposure = warpRef.get(parameters={"bbox": bbox})
            maskedImage = exposure.getMaskedImage()
            mask = maskedImage.getMask()
            if altMask is not None:
                self.applyAltMaskPlanes(mask, altMask)
            # Remove in DM-49083.
            if imageScaler is not None:
                imageScaler.scaleMaskedImage(maskedImage)
            if self.config.removeMaskPlanes:
                removeMaskPlanes(maskedImage.mask, self.config.removeMaskPlanes, logger=self.log)

            stacker.add_masked_image(maskedImage, weight=weight)

            if self.config.doInputMap:
                visit = exposure.getInfo().getCoaddInputs().visits[0].getId()
                self.inputMapper.mask_warp_bbox(bbox, visit, mask, statsCtrl.getAndMask())

        stacker.fill_stacked_masked_image(coaddExposure.maskedImage)

        if nImage is not None:
            nImage.array[:, :] = stacker.n_image

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
        if self._doUsePsfMatchedPolygons:
            if ("NO_DATA" in altMaskSpans) and ("NO_DATA" in self.config.badMaskPlanes):
                # Clear away any other masks outside the validPolygons. These
                # pixels are no longer contributing to inexact PSFs, and will
                # still be rejected because of NO_DATA.
                # self._doUsePsfMatchedPolygons should be True only in
                # CompareWarpAssemble. This mask-clearing step must only occur
                # *before* applying the new masks below.
                for spanSet in altMaskSpans["NO_DATA"]:
                    spanSet.clippedTo(mask.getBBox()).clearMask(mask, self.getBadPixelMask())

        for plane, spanSetList in altMaskSpans.items():
            maskClipValue = mask.addMaskPlane(plane)
            for spanSet in spanSetList:
                spanSet.clippedTo(mask.getBBox()).setMask(mask, 2**maskClipValue)
        return mask

    def _setValidPolygons(self, warpList, psfMatchedWarpRefList):
        """Set the valid polygons for the warps the same as psfMatchedWarps, if
        it exists.

        Parameters
        ----------
        warpList : `Iterable` [`lsst.afw.image.Exposure`]
            List of warps.
        psfMatchedWarpRefList : `Iterable` \
            [`lsst.daf.butler.DeferredDatasetHandle`]
            List of references to psfMatchedWarps, in the same order as
            ``warpList``.

        Raises
        ------
        ValueError
            Raised if PSF-matched warps have detectors that are absent in the
            (direct) warps.
        """
        for warp, psfMatchedWarpRef in zip(warpList, psfMatchedWarpRefList):
            if psfMatchedWarpRef is None:
                continue

            psfMatchedCcdTable = psfMatchedWarpRef.get(component="coaddInputs").ccds
            ccdTable = warp.getInfo().getCoaddInputs().ccds

            # In some (literal) edge cases, a small part of the CCD may be
            # present in directWarp that gets excluded in psfMatchedWarp. It is
            # okay to leave validPolygon for those CCDs empty. However, the
            # converse is not expected, and an error is raised in that case.
            if not set(psfMatchedCcdTable["id"]).issubset(ccdTable["id"]):
                visit = psfMatchedWarpRef.dataId["visit"]
                raise ValueError(f"PSF-matched warp has additional CCDs for {visit=}")

            if len(psfMatchedCcdTable) < len(ccdTable):
                self.log.debug(
                    "PSF-matched warp has missing CCDs for visit = %d, which leaves some CCDs in the direct "
                    "warp without a validPolygon",
                    psfMatchedWarpRef.dataId["visit"],
                )

            for psfMatchedCcdRow in psfMatchedCcdTable:
                if not psfMatchedCcdRow.validPolygon:
                    self.log.warning(
                        "No validPolygon in PSF-matched warp found for %s. This is likely due to a mismatch "
                        "in the LSST Science Pipelines version used to produce the warps and the current "
                        "version. To avoid this warning, regenerate the warps with the current version.",
                        psfMatchedCcdRow.id,
                    )
                else:
                    ccdRow = ccdTable.find(value=psfMatchedCcdRow.id, key=ccdTable.getIdKey())
                    ccdRow.validPolygon = psfMatchedCcdRow.validPolygon

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
        plateScale = wcs.getPixelScale(exposure.getBBox().getCenter()).asArcseconds()

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


class CompareWarpAssembleCoaddConnections(AssembleCoaddConnections):
    psfMatchedWarps = pipeBase.connectionTypes.Input(
        doc=(
            "PSF-Matched Warps are required by CompareWarp regardless of the coadd type requested. "
            "Only PSF-Matched Warps make sense for image subtraction. "
            "Therefore, they must be an additional declared input."
        ),
        name="{inputCoaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit"),
        deferLoad=True,
        multiple=True,
    )
    templateCoadd = pipeBase.connectionTypes.Output(
        doc=(
            "Model of the static sky, used to find temporal artifacts. Typically a PSF-Matched, "
            "sigma-clipped coadd. Written if and only if assembleStaticSkyModel.doWrite=True"
        ),
        name="{outputCoaddName}CoaddPsfMatched",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )
    artifactMasks = pipeBase.connectionTypes.Output(
        doc="Mask of artifacts detected in the coadd",
        name="compare_warp_artifact_mask",
        storageClass="Mask",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.assembleStaticSkyModel.doWrite:
            self.outputs.remove("templateCoadd")
        config.validate()


class CompareWarpAssembleCoaddConfig(
    AssembleCoaddConfig, pipelineConnections=CompareWarpAssembleCoaddConnections
):
    assembleStaticSkyModel = pexConfig.ConfigurableField(
        target=AssembleCoaddTask,
        doc="Task to assemble an artifact-free, PSF-matched Coadd to serve as "
        "a naive/first-iteration model of the static sky.",
    )
    detect = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect outlier sources on difference between each psfMatched warp and static sky model",
    )
    detectTemplate = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources on static sky model. Only used if doPreserveContainedBySource is True",
    )
    maskStreaks = pexConfig.ConfigurableField(
        target=MaskStreaksTask,
        doc="Detect streaks on difference between each psfMatched warp and static sky model. Only used if "
        "doFilterMorphological is True. Adds a mask plane to an exposure, with the mask plane name set by"
        "streakMaskName",
    )
    streakMaskName = pexConfig.Field(dtype=str, default="STREAK", doc="Name of mask bit used for streaks")
    maxNumEpochs = pexConfig.Field(
        doc="Charactistic maximum local number of epochs/visits in which an artifact candidate can appear  "
        "and still be masked.  The effective maxNumEpochs is a broken linear function of local "
        "number of epochs (N): min(maxFractionEpochsLow*N, maxNumEpochs + maxFractionEpochsHigh*N). "
        "For each footprint detected on the image difference between the psfMatched warp and static sky "
        "model, if a significant fraction of pixels (defined by spatialThreshold) are residuals in more "
        "than the computed effective maxNumEpochs, the artifact candidate is deemed persistant rather "
        "than transient and not masked.",
        dtype=int,
        default=2,
    )
    maxFractionEpochsLow = pexConfig.RangeField(
        doc="Fraction of local number of epochs (N) to use as effective maxNumEpochs for low N. "
        "Effective maxNumEpochs = "
        "min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)",
        dtype=float,
        default=0.4,
        min=0.0,
        max=1.0,
    )
    maxFractionEpochsHigh = pexConfig.RangeField(
        doc="Fraction of local number of epochs (N) to use as effective maxNumEpochs for high N. "
        "Effective maxNumEpochs = "
        "min(maxFractionEpochsLow * N, maxNumEpochs + maxFractionEpochsHigh * N)",
        dtype=float,
        default=0.03,
        min=0.0,
        max=1.0,
    )
    spatialThreshold = pexConfig.RangeField(
        doc="Unitless fraction of pixels defining how much of the outlier region has to meet the "
        "temporal criteria. If 0, clip all. If 1, clip none.",
        dtype=float,
        default=0.5,
        min=0.0,
        max=1.0,
        inclusiveMin=True,
        inclusiveMax=True,
    )
    doScaleWarpVariance = pexConfig.Field(
        doc="Rescale Warp variance plane using empirical noise?",
        dtype=bool,
        default=True,
    )
    scaleWarpVariance = pexConfig.ConfigurableField(
        target=ScaleVarianceTask,
        doc="Rescale variance on warps",
    )
    doPreserveContainedBySource = pexConfig.Field(
        doc="Rescue artifacts from clipping that completely lie within a footprint detected"
        "on the PsfMatched Template Coadd. Replicates a behavior of SafeClip.",
        dtype=bool,
        default=True,
    )
    doPrefilterArtifacts = pexConfig.Field(
        doc="Ignore artifact candidates that are mostly covered by the bad pixel mask, "
        "because they will be excluded anyway. This prevents them from contributing "
        "to the outlier epoch count image and potentially being labeled as persistant."
        "'Mostly' is defined by the config 'prefilterArtifactsRatio'.",
        dtype=bool,
        default=True,
    )
    prefilterArtifactsMaskPlanes = pexConfig.ListField(
        doc="Prefilter artifact candidates that are mostly covered by these bad mask planes.",
        dtype=str,
        default=("NO_DATA", "BAD", "SAT", "SUSPECT"),
    )
    prefilterArtifactsRatio = pexConfig.Field(
        doc="Prefilter artifact candidates with less than this fraction overlapping good pixels",
        dtype=float,
        default=0.05,
    )
    doFilterMorphological = pexConfig.Field(
        doc="Filter artifact candidates based on morphological criteria, i.g. those that appear to "
        "be streaks.",
        dtype=bool,
        default=False,
    )
    growStreakFp = pexConfig.Field(
        doc="Grow streak footprints by this number multiplied by the PSF width", dtype=float, default=5
    )

    def setDefaults(self):
        AssembleCoaddConfig.setDefaults(self)
        self.statistic = "MEAN"

        # Real EDGE removed by psfMatched NO_DATA border half the width of the
        # matching kernel. CompareWarp applies psfMatched EDGE pixels to
        # directWarps before assembling.
        if "EDGE" in self.badMaskPlanes:
            self.badMaskPlanes.remove("EDGE")
        self.removeMaskPlanes.append("EDGE")
        self.assembleStaticSkyModel.badMaskPlanes = [
            "NO_DATA",
        ]
        self.assembleStaticSkyModel.warpType = "psfMatched"
        self.assembleStaticSkyModel.connections.warpType = "psfMatched"
        self.assembleStaticSkyModel.statistic = "MEANCLIP"
        self.assembleStaticSkyModel.sigmaClip = 2.5
        self.assembleStaticSkyModel.clipIter = 3
        self.assembleStaticSkyModel.calcErrorFromInputVariance = False
        self.assembleStaticSkyModel.doWrite = False
        self.detect.doTempLocalBackground = False
        self.detect.reEstimateBackground = False
        self.detect.returnOriginalFootprints = False
        self.detect.thresholdPolarity = "both"
        self.detect.thresholdValue = 5
        self.detect.minPixels = 4
        self.detect.isotropicGrow = True
        self.detect.thresholdType = "pixel_stdev"
        self.detect.nSigmaToGrow = 0.4
        # The default nSigmaToGrow for SourceDetectionTask is already 2.4,
        # Explicitly restating because ratio with detect.nSigmaToGrow matters
        self.detectTemplate.nSigmaToGrow = 2.4
        # Higher thresholds make smaller and fewer protected zones around
        # bright stars. Sources with snr < 50 tend to subtract OK empirically
        self.detectTemplate.thresholdValue = 50
        self.detectTemplate.doTempLocalBackground = False
        self.detectTemplate.reEstimateBackground = False
        self.detectTemplate.returnOriginalFootprints = False

    def validate(self):
        super().validate()
        if self.assembleStaticSkyModel.doNImage:
            raise ValueError(
                "No dataset type exists for a PSF-Matched Template N Image."
                "Please set assembleStaticSkyModel.doNImage=False"
            )

        if self.assembleStaticSkyModel.doWrite and (self.warpType == self.assembleStaticSkyModel.warpType):
            raise ValueError(
                "warpType (%s) == assembleStaticSkyModel.warpType (%s) and will compete for "
                "the same dataset name. Please set assembleStaticSkyModel.doWrite to False "
                "or warpType to 'direct'. assembleStaticSkyModel.warpType should ways be "
                "'PsfMatched'" % (self.warpType, self.assembleStaticSkyModel.warpType)
            )


class CompareWarpAssembleCoaddTask(AssembleCoaddTask):
    """Assemble a compareWarp coadded image from a set of warps
    by masking artifacts detected by comparing PSF-matched warps.

    In ``AssembleCoaddTask``, we compute the coadd as an clipped mean (i.e.,
    we clip outliers). The problem with doing this is that when computing the
    coadd PSF at a given location, individual visit PSFs from visits with
    outlier pixels contribute to the coadd PSF and cannot be treated correctly.
    In this task, we correct for this behavior by creating a new badMaskPlane
    'CLIPPED' which marks pixels in the individual warps suspected to contain
    an artifact. We populate this plane on the input warps by comparing
    PSF-matched warps with a PSF-matched median coadd which serves as a
    model of the static sky. Any group of pixels that deviates from the
    PSF-matched template coadd by more than config.detect.threshold sigma,
    is an artifact candidate. The candidates are then filtered to remove
    variable sources and sources that are difficult to subtract such as
    bright stars. This filter is configured using the config parameters
    ``temporalThreshold`` and ``spatialThreshold``. The temporalThreshold is
    the maximum fraction of epochs that the deviation can appear in and still
    be considered an artifact. The spatialThreshold is the maximum fraction of
    pixels in the footprint of the deviation that appear in other epochs
    (where other epochs is defined by the temporalThreshold). If the deviant
    region meets this criteria of having a significant percentage of pixels
    that deviate in only a few epochs, these pixels have the 'CLIPPED' bit
    set in the mask. These regions will not contribute to the final coadd.
    Furthermore, any routine to determine the coadd PSF can now be cognizant
    of clipped regions. Note that the algorithm implemented by this task is
    preliminary and works correctly for HSC data. Parameter modifications and
    or considerable redesigning of the algorithm is likley required for other
    surveys.

    ``CompareWarpAssembleCoaddTask`` sub-classes
    ``AssembleCoaddTask`` and instantiates ``AssembleCoaddTask``
    as a subtask to generate the TemplateCoadd (the model of the static sky).

    Notes
    -----
    Debugging:
    This task supports the following debug variables:
    - ``saveCountIm``
        If True then save the Epoch Count Image as a fits file in the `figPath`
    - ``figPath``
        Path to save the debug fits images and figures
    """

    ConfigClass = CompareWarpAssembleCoaddConfig
    _DefaultName = "compareWarpAssembleCoadd"

    # See the parent class for docstring.
    _doUsePsfMatchedPolygons: bool = True

    def __init__(self, *args, **kwargs):
        AssembleCoaddTask.__init__(self, *args, **kwargs)
        self.makeSubtask("assembleStaticSkyModel")
        detectionSchema = afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("detect", schema=detectionSchema)
        if self.config.doPreserveContainedBySource:
            self.makeSubtask("detectTemplate", schema=afwTable.SourceTable.makeMinimalSchema())
        if self.config.doScaleWarpVariance:
            self.makeSubtask("scaleWarpVariance")
        if self.config.doFilterMorphological:
            self.makeSubtask("maskStreaks")

    @utils.inheritDoc(AssembleCoaddTask)
    def _makeSupplementaryData(self, butlerQC, inputRefs, outputRefs):
        """Generate a templateCoadd to use as a naive model of static sky to
        subtract from PSF-Matched warps.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``templateCoadd``
                Coadded exposure (`lsst.afw.image.Exposure`).
            ``nImage``
                Keeps track of exposure count for each pixel
                (`lsst.afw.image.ImageU`).

        Raises
        ------
        RuntimeError
            Raised if ``templateCoadd`` is `None`.
        """
        # Ensure that psfMatchedWarps are used as input warps for template
        # generation.
        staticSkyModelInputRefs = copy.deepcopy(inputRefs)
        staticSkyModelInputRefs.inputWarps = inputRefs.psfMatchedWarps

        # Because subtasks don't have connections we have to make one.
        # The main task's `templateCoadd` is the subtask's `coaddExposure`
        staticSkyModelOutputRefs = copy.deepcopy(outputRefs)
        if self.config.assembleStaticSkyModel.doWrite:
            staticSkyModelOutputRefs.coaddExposure = staticSkyModelOutputRefs.templateCoadd
            # Remove template coadd from both subtask's and main tasks outputs,
            # because it is handled by the subtask as `coaddExposure`
            del outputRefs.templateCoadd
            del staticSkyModelOutputRefs.templateCoadd

        # A PSF-Matched nImage does not exist as a dataset type
        if "nImage" in staticSkyModelOutputRefs.keys():
            del staticSkyModelOutputRefs.nImage

        templateCoadd = self.assembleStaticSkyModel.runQuantum(
            butlerQC, staticSkyModelInputRefs, staticSkyModelOutputRefs
        )
        if templateCoadd is None:
            raise RuntimeError(self._noTemplateMessage(self.assembleStaticSkyModel.warpType))

        return pipeBase.Struct(
            templateCoadd=templateCoadd.coaddExposure,
            nImage=templateCoadd.nImage,
            warpRefList=templateCoadd.warpRefList,
            imageScalerList=templateCoadd.imageScalerList,
            weightList=templateCoadd.weightList,
            psfMatchedWarpRefList=inputRefs.psfMatchedWarps,
        )

    def _noTemplateMessage(self, warpType):
        warpName = warpType[0].upper() + warpType[1:]
        message = """No %(warpName)s warps were found to build the template coadd which is
            required to run CompareWarpAssembleCoaddTask. To continue assembling this type of coadd,
            first either rerun makeCoaddTempExp with config.make%(warpName)s=True or
            coaddDriver with config.makeCoadTempExp.make%(warpName)s=True, before assembleCoadd.

            Alternatively, to use another algorithm with existing warps, retarget the CoaddDriverConfig to
            another algorithm like:

                from lsst.pipe.tasks.assembleCoadd import SafeClipAssembleCoaddTask
                config.assemble.retarget(SafeClipAssembleCoaddTask)
        """ % {
            "warpName": warpName
        }
        return message

    @utils.inheritDoc(AssembleCoaddTask)
    @timeMethod
    def run(
        self,
        skyInfo,
        *,
        warpRefList,
        imageScalerList,
        weightList,
        psfMatchedWarpRefList,
        supplementaryData,
    ):
        """Notes
        -----
        Assemble the coadd.

        Find artifacts and apply them to the warps' masks creating a list of
        alternative masks with a new "CLIPPED" plane and updated "NO_DATA"
        plane. Then pass these alternative masks to the base class's ``run``
        method.
        """
        # Check and match the order of the supplementaryData
        # (PSF-matched) inputs to the order of the direct inputs,
        # so that the artifact mask is applied to the right warp
        dataIds = [ref.dataId for ref in warpRefList]
        psfMatchedDataIds = [ref.dataId for ref in supplementaryData.warpRefList]

        if dataIds != psfMatchedDataIds:
            self.log.info("Reordering and or/padding PSF-matched visit input list")
            supplementaryData.warpRefList = reorderAndPadList(
                supplementaryData.warpRefList, psfMatchedDataIds, dataIds
            )
            # Remove in DM-49083
            supplementaryData.imageScalerList = reorderAndPadList(
                supplementaryData.imageScalerList, psfMatchedDataIds, dataIds
            )

        # Use PSF-Matched Warps (and corresponding scalers) and coadd to find
        # artifacts.
        spanSetMaskList = self.findArtifacts(
            supplementaryData.templateCoadd, supplementaryData.warpRefList, supplementaryData.imageScalerList
        )

        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)

        result = AssembleCoaddTask.run(
            self,
            skyInfo,
            warpRefList=warpRefList,
            imageScalerList=imageScalerList,
            weightList=weightList,
            altMaskList=spanSetMaskList,
            mask=badPixelMask,
            psfMatchedWarpRefList=psfMatchedWarpRefList,
        )

        # Propagate PSF-matched EDGE pixels to coadd SENSOR_EDGE and
        # INEXACT_PSF. Psf-Matching moves the real edge inwards.
        self.applyAltEdgeMask(result.coaddExposure.maskedImage.mask, spanSetMaskList)
        return result

    def applyAltEdgeMask(self, mask, altMaskList):
        """Propagate alt EDGE mask to SENSOR_EDGE AND INEXACT_PSF planes.

        Parameters
        ----------
        mask : `lsst.afw.image.Mask`
            Original mask.
        altMaskList : `list` of `dict`
            List of Dicts containing ``spanSet`` lists.
            Each element contains the new mask plane name (e.g. "CLIPPED
            and/or "NO_DATA") as the key, and list of ``SpanSets`` to apply to
            the mask.
        """
        maskValue = mask.getPlaneBitMask(["SENSOR_EDGE", "INEXACT_PSF"])
        for visitMask in altMaskList:
            if "EDGE" in visitMask:
                for spanSet in visitMask["EDGE"]:
                    spanSet.clippedTo(mask.getBBox()).setMask(mask, maskValue)

    def findArtifacts(self, templateCoadd, warpRefList, imageScalerList):
        """Find artifacts.

        Loop through warps twice. The first loop builds a map with the count
        of how many epochs each pixel deviates from the templateCoadd by more
        than ``config.chiThreshold`` sigma. The second loop takes each
        difference image and filters the artifacts detected in each using
        count map to filter out variable sources and sources that are
        difficult to subtract cleanly.

        Parameters
        ----------
        templateCoadd : `lsst.afw.image.Exposure`
            Exposure to serve as model of static sky.
        warpRefList : `list`
            List of dataset handles (data references) to warps.
        imageScalerList : `list`
            List of image scalers.
            Deprecated and will be removed after v29.

        Returns
        -------
        altMasks : `list` of `dict`
            List of dicts containing information about CLIPPED
            (i.e., artifacts), NO_DATA, and EDGE pixels.
        """
        self.log.debug("Generating Count Image, and mask lists.")
        coaddBBox = templateCoadd.getBBox()
        slateIm = afwImage.ImageU(coaddBBox)
        epochCountImage = afwImage.ImageU(coaddBBox)
        nImage = afwImage.ImageU(coaddBBox)
        spanSetArtifactList = []
        spanSetNoDataMaskList = []
        spanSetEdgeList = []
        spanSetBadMorphoList = []
        badPixelMask = self.getBadPixelMask()

        # mask of the warp diffs should = that of only the warp
        templateCoadd.mask.clearAllMaskPlanes()

        if self.config.doPreserveContainedBySource:
            templateFootprints = self.detectTemplate.detectFootprints(templateCoadd)
        else:
            templateFootprints = None

        for warpRef, imageScaler in zip(warpRefList, imageScalerList):
            warpDiffExp = self._readAndComputeWarpDiff(warpRef, imageScaler, templateCoadd)
            if warpDiffExp is not None:
                # This nImage only approximates the final nImage because it
                # uses the PSF-matched mask.
                nImage.array += (
                    numpy.isfinite(warpDiffExp.image.array) * ((warpDiffExp.mask.array & badPixelMask) == 0)
                ).astype(numpy.uint16)
                fpSet = self.detect.detectFootprints(warpDiffExp, doSmooth=False, clearMask=True)
                fpSet.positive.merge(fpSet.negative)
                footprints = fpSet.positive
                slateIm.set(0)
                spanSetList = [footprint.spans for footprint in footprints.getFootprints()]

                # Remove artifacts due to defects before they contribute to
                # the epochCountImage.
                if self.config.doPrefilterArtifacts:
                    spanSetList = self.prefilterArtifacts(spanSetList, warpDiffExp)

                # Clear mask before adding prefiltered spanSets
                self.detect.clearMask(warpDiffExp.mask)
                for spans in spanSetList:
                    spans.setImage(slateIm, 1, doClip=True)
                    spans.setMask(warpDiffExp.mask, warpDiffExp.mask.getPlaneBitMask("DETECTED"))
                epochCountImage += slateIm

                if self.config.doFilterMorphological:
                    maskName = self.config.streakMaskName
                    # clear single frame streak mask if it exists already
                    if maskName in warpDiffExp.mask.getMaskPlaneDict():
                        warpDiffExp.mask.clearMaskPlane(warpDiffExp.mask.getMaskPlane(maskName))
                    else:
                        self.log.debug(f"Did not (need to) clear {maskName} mask because it didn't exist")

                    _ = self.maskStreaks.run(warpDiffExp)
                    streakMask = warpDiffExp.mask
                    spanSetStreak = afwGeom.SpanSet.fromMask(
                        streakMask, streakMask.getPlaneBitMask(maskName)
                    ).split()
                    # Pad the streaks to account for low-surface brightness
                    # wings.
                    psf = warpDiffExp.getPsf()
                    for s, sset in enumerate(spanSetStreak):
                        psfShape = psf.computeShape(sset.computeCentroid())
                        dilation = self.config.growStreakFp * psfShape.getDeterminantRadius()
                        sset_dilated = sset.dilated(int(dilation))
                        spanSetStreak[s] = sset_dilated

                # PSF-Matched warps have less available area (~the matching
                # kernel) because the calexps undergo a second convolution.
                # Pixels with data in the direct warp but not in the
                # PSF-matched warp will not have their artifacts detected.
                # NaNs from the PSF-matched warp therefore must be masked in
                # the direct warp.
                nans = numpy.where(numpy.isnan(warpDiffExp.maskedImage.image.array), 1, 0)
                nansMask = afwImage.makeMaskFromArray(nans.astype(afwImage.MaskPixel))
                nansMask.setXY0(warpDiffExp.getXY0())
                edgeMask = warpDiffExp.mask
                spanSetEdgeMask = afwGeom.SpanSet.fromMask(edgeMask, edgeMask.getPlaneBitMask("EDGE")).split()
            else:
                # If the directWarp has <1% coverage, the psfMatchedWarp can
                # have 0% and not exist. In this case, mask the whole epoch.
                nansMask = afwImage.MaskX(coaddBBox, 1)
                spanSetList = []
                spanSetEdgeMask = []
                spanSetStreak = []

            spanSetNoDataMask = afwGeom.SpanSet.fromMask(nansMask).split()

            spanSetNoDataMaskList.append(spanSetNoDataMask)
            spanSetArtifactList.append(spanSetList)
            spanSetEdgeList.append(spanSetEdgeMask)
            if self.config.doFilterMorphological:
                spanSetBadMorphoList.append(spanSetStreak)

        if lsstDebug.Info(__name__).saveCountIm:
            path = self._dataRef2DebugPath("epochCountIm", warpRefList[0], coaddLevel=True)
            epochCountImage.writeFits(path)

        for i, spanSetList in enumerate(spanSetArtifactList):
            if spanSetList:
                filteredSpanSetList = self.filterArtifacts(
                    spanSetList, epochCountImage, nImage, templateFootprints
                )
                spanSetArtifactList[i] = filteredSpanSetList
            if self.config.doFilterMorphological:
                spanSetArtifactList[i] += spanSetBadMorphoList[i]

        altMasks = []
        for artifacts, noData, edge in zip(spanSetArtifactList, spanSetNoDataMaskList, spanSetEdgeList):
            altMasks.append({"CLIPPED": artifacts, "NO_DATA": noData, "EDGE": edge})
        return altMasks

    def prefilterArtifacts(self, spanSetList, exp):
        """Remove artifact candidates covered by bad mask plane.

        Any future editing of the candidate list that does not depend on
        temporal information should go in this method.

        Parameters
        ----------
        spanSetList : `list` [`lsst.afw.geom.SpanSet`]
            List of SpanSets representing artifact candidates.
        exp : `lsst.afw.image.Exposure`
            Exposure containing mask planes used to prefilter.

        Returns
        -------
        returnSpanSetList : `list` [`lsst.afw.geom.SpanSet`]
            List of SpanSets with artifacts.
        """
        badPixelMask = exp.mask.getPlaneBitMask(self.config.prefilterArtifactsMaskPlanes)
        goodArr = (exp.mask.array & badPixelMask) == 0
        returnSpanSetList = []
        bbox = exp.getBBox()
        x0, y0 = exp.getXY0()
        for i, span in enumerate(spanSetList):
            y, x = span.clippedTo(bbox).indices()
            yIndexLocal = numpy.array(y) - y0
            xIndexLocal = numpy.array(x) - x0
            goodRatio = numpy.count_nonzero(goodArr[yIndexLocal, xIndexLocal]) / span.getArea()
            if goodRatio > self.config.prefilterArtifactsRatio:
                returnSpanSetList.append(span)
        return returnSpanSetList

    def filterArtifacts(self, spanSetList, epochCountImage, nImage, footprintsToExclude=None):
        """Filter artifact candidates.

        Parameters
        ----------
        spanSetList : `list` [`lsst.afw.geom.SpanSet`]
            List of SpanSets representing artifact candidates.
        epochCountImage : `lsst.afw.image.Image`
            Image of accumulated number of warpDiff detections.
        nImage : `lsst.afw.image.ImageU`
            Image of the accumulated number of total epochs contributing.

        Returns
        -------
        maskSpanSetList : `list` [`lsst.afw.geom.SpanSet`]
            List of SpanSets with artifacts.
        """
        maskSpanSetList = []
        x0, y0 = epochCountImage.getXY0()
        for i, span in enumerate(spanSetList):
            y, x = span.indices()
            yIdxLocal = [y1 - y0 for y1 in y]
            xIdxLocal = [x1 - x0 for x1 in x]
            outlierN = epochCountImage.array[yIdxLocal, xIdxLocal]
            totalN = nImage.array[yIdxLocal, xIdxLocal]

            # effectiveMaxNumEpochs is broken line (fraction of N) with
            # characteristic config.maxNumEpochs.
            effMaxNumEpochsHighN = self.config.maxNumEpochs + self.config.maxFractionEpochsHigh * numpy.mean(
                totalN
            )
            effMaxNumEpochsLowN = self.config.maxFractionEpochsLow * numpy.mean(totalN)
            effectiveMaxNumEpochs = int(min(effMaxNumEpochsLowN, effMaxNumEpochsHighN))
            nPixelsBelowThreshold = numpy.count_nonzero((outlierN > 0) & (outlierN <= effectiveMaxNumEpochs))
            percentBelowThreshold = nPixelsBelowThreshold / len(outlierN)
            if percentBelowThreshold > self.config.spatialThreshold:
                maskSpanSetList.append(span)

        if self.config.doPreserveContainedBySource and footprintsToExclude is not None:
            # If a candidate is contained by a footprint on the template coadd,
            # do not clip.
            filteredMaskSpanSetList = []
            for span in maskSpanSetList:
                doKeep = True
                for footprint in footprintsToExclude.positive.getFootprints():
                    if footprint.spans.contains(span):
                        doKeep = False
                        break
                if doKeep:
                    filteredMaskSpanSetList.append(span)
            maskSpanSetList = filteredMaskSpanSetList

        return maskSpanSetList

    def _readAndComputeWarpDiff(self, warpRef, imageScaler, templateCoadd):
        """Fetch a warp from the butler and return a warpDiff.

        Parameters
        ----------
        warpRef : `lsst.daf.butler.DeferredDatasetHandle`
            Dataset handle for the warp.
        imageScaler : `lsst.pipe.tasks.scaleZeroPoint.ImageScaler`
            An image scaler object.
            Deprecated and will be removed after v29 in DM-49083.
        templateCoadd : `lsst.afw.image.Exposure`
            Exposure to be substracted from the scaled warp.

        Returns
        -------
        warp : `lsst.afw.image.Exposure`
            Exposure of the image difference between the warp and template.
        """
        # If the PSF-Matched warp did not exist for this direct warp
        # None is holding its place to maintain order in Gen 3
        if warpRef is None:
            return None

        warp = warpRef.get(parameters={"bbox": templateCoadd.getBBox()})
        # direct image scaler OK for PSF-matched Warp.
        # Remove in DM-49083.
        if imageScaler is not None:
            imageScaler.scaleMaskedImage(warp.getMaskedImage())
        mi = warp.getMaskedImage()
        if self.config.doScaleWarpVariance:
            try:
                self.scaleWarpVariance.run(mi)
            except Exception as exc:
                self.log.warning("Unable to rescale variance of warp (%s); leaving it as-is", exc)
        mi -= templateCoadd.getMaskedImage()
        return warp
