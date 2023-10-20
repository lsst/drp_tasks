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
    "CompareWarpAssembleCoaddConfig",
    "CompareWarpAssembleCoaddTask",
)

import copy
import logging

import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.utils as utils
import lsstDebug
import numpy
from lsst.meas.algorithms import ScaleVarianceTask, SourceDetectionTask
from lsst.pipe.tasks.coaddBase import reorderAndPadList
from lsst.pipe.tasks.maskStreaks import MaskStreaksTask
from lsst.utils.timer import timeMethod

from .assemble_coadd import (
    AssembleCoaddConfig,
    AssembleCoaddConnections,
    AssembleCoaddTask,
)

log = logging.getLogger(__name__)


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
    streakMaskName = pexConfig.Field(
        dtype=str,
        default="STREAK",
        doc="Name of mask bit used for streaks",
    )
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
        doc="Grow streak footprints by this number multiplied by the PSF width",
        dtype=float,
        default=5,
    )

    def setDefaults(self):
        AssembleCoaddConfig.setDefaults(self)
        self.statistic = "MEAN"
        self.doUsePsfMatchedPolygons = True

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
    def run(self, skyInfo, tempExpRefList, imageScalerList, weightList, supplementaryData):
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
        dataIds = [ref.dataId for ref in tempExpRefList]
        psfMatchedDataIds = [ref.dataId for ref in supplementaryData.warpRefList]

        if dataIds != psfMatchedDataIds:
            self.log.info("Reordering and or/padding PSF-matched visit input list")
            supplementaryData.warpRefList = reorderAndPadList(
                supplementaryData.warpRefList, psfMatchedDataIds, dataIds
            )
            supplementaryData.imageScalerList = reorderAndPadList(
                supplementaryData.imageScalerList, psfMatchedDataIds, dataIds
            )

        # Use PSF-Matched Warps (and corresponding scalers) and coadd to find
        # artifacts.
        spanSetMaskList = self.findArtifacts(
            supplementaryData.templateCoadd,
            supplementaryData.warpRefList,
            supplementaryData.imageScalerList,
        )

        badMaskPlanes = self.config.badMaskPlanes[:]
        badMaskPlanes.append("CLIPPED")
        badPixelMask = afwImage.Mask.getPlaneBitMask(badMaskPlanes)

        result = AssembleCoaddTask.run(
            self,
            skyInfo,
            tempExpRefList,
            imageScalerList,
            weightList,
            spanSetMaskList,
            mask=badPixelMask,
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

    def findArtifacts(self, templateCoadd, tempExpRefList, imageScalerList):
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
        tempExpRefList : `list`
            List of data references to warps.
        imageScalerList : `list`
            List of image scalers.

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

        for warpRef, imageScaler in zip(tempExpRefList, imageScalerList):
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
            path = self._dataRef2DebugPath("epochCountIm", tempExpRefList[0], coaddLevel=True)
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
            Handle for the warp.
        imageScaler : `lsst.pipe.tasks.scaleZeroPoint.ImageScaler`
            An image scaler object.
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

        warp = warpRef.get()
        # direct image scaler OK for PSF-matched Warp
        imageScaler.scaleMaskedImage(warp.getMaskedImage())
        mi = warp.getMaskedImage()
        if self.config.doScaleWarpVariance:
            try:
                self.scaleWarpVariance.run(mi)
            except Exception as exc:
                self.log.warning("Unable to rescale variance of warp (%s); leaving it as-is", exc)
        mi -= templateCoadd.getMaskedImage()
        return warp
