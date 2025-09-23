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

__all__ = (
    "AssembleCellCoaddTask",
    "AssembleCellCoaddConfig",
    "ConvertMultipleCellCoaddToExposureTask",
    "EmptyCellCoaddError",
)

import dataclasses

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
from lsst.cell_coadds import (
    CellIdentifiers,
    CoaddApCorrMapStacker,
    CoaddUnits,
    CommonComponents,
    GridContainer,
    MultipleCellCoadd,
    ObservationIdentifiers,
    OwnedImagePlanes,
    PatchIdentifiers,
    SingleCellCoadd,
    UniformGrid,
)
from lsst.daf.butler import DataCoordinate, DeferredDatasetHandle
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pex.config import ConfigField, ConfigurableField, DictField, Field, ListField, RangeField
from lsst.pipe.base import (
    AlgorithmError,
    InMemoryDatasetHandle,
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import makeSkyInfo, removeMaskPlanes, setRejectedMaskMapping
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.pipe.tasks.scaleZeroPoint import ScaleZeroPointTask
from lsst.skymap import BaseSkyMap


class EmptyCellCoaddError(AlgorithmError):
    """Raised if no cells could be populated."""

    def __init__(self):
        msg = "No cells could be populated for the cell coadd."
        super().__init__(msg)

    @property
    def metadata(self) -> dict:
        """There is no metadata associated with this error."""
        return {}


@dataclasses.dataclass
class WarpInputs:
    """Collection of associate inputs along with warps."""

    warp: DeferredDatasetHandle | InMemoryDatasetHandle
    """Handle for the warped exposure."""

    masked_fraction: DeferredDatasetHandle | InMemoryDatasetHandle | None = None
    """Handle for the masked fraction image."""

    artifact_mask: DeferredDatasetHandle | InMemoryDatasetHandle | None = None
    """Handle for the CompareWarp artifact mask."""

    noise_warps: list[DeferredDatasetHandle | InMemoryDatasetHandle] = dataclasses.field(default_factory=list)
    """List of handles for the noise warps"""

    @property
    def dataId(self) -> DataCoordinate:
        """DataID corresponding to the warp.

        Returns
        -------
        data_id : `~lsst.daf.butler.DataCoordinate`
            DataID of the warp.
        """
        return self.warp.dataId


class AssembleCellCoaddConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={"inputWarpName": "deep", "outputCoaddSuffix": "Cell"},
):
    inputWarps = Input(
        doc="Input warps",
        name="{inputWarpName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )

    maskedFractionWarps = Input(
        doc="Mask fraction warps",
        name="{inputWarpName}Coadd_directWarp_maskedFraction",
        storageClass="ImageF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )

    artifactMasks = Input(
        doc="Artifact masks to be applied to the input warps",
        name="compare_warp_artifact_mask",
        storageClass="Mask",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )

    visitSummaryList = Input(
        doc="Input visit-summary catalogs with updated calibration objects. Mainly used for coadd weights.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        multiple=True,
    )

    skyMap = Input(
        doc="Input definition of geometry/bbox and projection/wcs. This must be cell-based.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    multipleCellCoadd = Output(
        doc="Output multiple cell coadd",
        name="{inputWarpName}Coadd{outputCoaddSuffix}",
        storageClass="MultipleCellCoadd",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return

        if config.do_calculate_weight_from_warp:
            del self.visitSummaryList

        if not config.do_use_artifact_mask:
            del self.artifactMasks

        # Dynamically set input connections for noise images, depending on the
        # number of noise realizations specified in the config.
        for n in range(config.num_noise_realizations):
            noise_warps = Input(
                doc="Input noise warps",
                name=f"{config.connections.inputWarpName}Coadd_directWarp_noise{n}",
                storageClass="MaskedImageF",
                dimensions=("tract", "patch", "skymap", "visit", "instrument"),
                deferLoad=True,
                multiple=True,
            )
            setattr(self, f"noise{n}_warps", noise_warps)


class AssembleCellCoaddConfig(PipelineTaskConfig, pipelineConnections=AssembleCellCoaddConnections):
    do_interpolate_coadd = Field[bool](doc="Interpolate over pixels with NO_DATA mask set?", default=True)
    interpolate_coadd = ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over pixels with NO_DATA mask on cell coadds",
    )
    do_scale_zero_point = Field[bool](
        doc="Scale warps to a common zero point? This is not needed if they have absolute flux calibration.",
        default=False,
        deprecated="Now that visits are scaled to nJy it is no longer necessary or "
        "recommended to scale the zero point, so this will be removed "
        "after v29.",
    )
    scale_zero_point = ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to scale warps to a common zero point",
        deprecated="Now that visits are scaled to nJy it is no longer necessary or "
        "recommended to scale the zero point, so this will be removed "
        "after v29.",
    )
    do_calculate_weight_from_warp = Field[bool](
        doc="Calculate coadd weight from the input warp? Otherwise, the weight is obtained from the "
        "visitSummaryList connection. This is meant as a fallback when run outside the pipeline.",
        default=False,
    )
    do_use_artifact_mask = Field[bool](
        doc="Substitute the mask planes input warp with an alternative artifact mask?",
        default=True,
    )
    do_coadd_inverse_aperture_corrections = Field[bool](
        doc="Coadd the inverse aperture corrections for each cell? This is formally the more accurate way "
        "but may be turned off for parity with deepCoadd.",
        default=False,
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that count towards the masked fraction within a cell.",
        default=("BAD", "NO_DATA", "SAT", "CLIPPED"),
    )
    remove_mask_planes = ListField[str](
        doc="Mask planes to remove before coadding",
        default=["NOT_DEBLENDED", "EDGE"],
    )
    calc_error_from_input_variance = Field[bool](
        doc="Calculate coadd variance from input variance by stacking "
        "statistic. Passed to AccumulatorMeanStack.",
        default=True,
    )
    mask_propagation_thresholds = DictField[str, float](
        doc=(
            "Threshold (in fractional weight) of rejection at which we "
            "propagate a mask plane to the coadd; that is, we set the mask "
            "bit on the coadd if the fraction the rejected frames "
            "would have contributed exceeds this value."
        ),
        default={"SAT": 0.1},
    )
    max_maskfrac = RangeField[float](
        doc="Maximum fraction of masked pixels in a cell. This is currently "
        "just a placeholder and is not used now",
        default=0.99,
        min=0.0,
        max=1.0,
        inclusiveMin=True,
        inclusiveMax=False,
    )
    num_noise_realizations = Field[int](
        default=0,
        doc=(
            "Number of noise planes to include in the coadd. "
            "This should not exceed the corresponding config parameter "
            "specified in `MakeDirectWarpConfig`. "
        ),
        check=lambda x: x >= 0,
    )
    psf_warper = ConfigField(
        doc="Configuration for the warper that warps the PSFs. It must have the same configuration used to "
        "warp the images.",
        dtype=afwMath.Warper.ConfigClass,
    )
    psf_dimensions = Field[int](
        default=21,
        doc="Dimensions of the PSF image stamp size to be assigned to cells (must be odd).",
        check=lambda x: (x > 0) and (x % 2 == 1),
    )


class AssembleCellCoaddTask(PipelineTask):
    """Assemble a cell-based coadded image from a set of warps.

    This task reads in the warp one at a time, and accumulates it in all the
    cells that it completely overlaps with. This is the optimal I/O pattern but
    this also implies that it is not possible to build one or only a few cells.

    Each cell coadds is guaranteed to have a well-defined PSF. This is done by
    1) excluding warps that only partially overlap a cell from that cell coadd;
    2) interpolating bad pixels in the warps rather than excluding them;
    3) by computing the coadd as a weighted mean of the warps without clipping;
    4) by computing the coadd PSF as the weighted mean of the PSF of the warps
       with the same weights.

    The cells are (and must be) defined in the skymap, and cannot be configured
    or redefined here. The cells are assumed to be small enough that the PSF is
    assumed to be spatially constant within a cell.

    Raises
    ------
    NoWorkFound
        Raised if no input warps are provided, or no cells could be populated.
    RuntimeError
        Raised if the skymap is not cell-based.

    Notes
    -----
    This is not yet a part of the standard DRP pipeline. As such, the Task and
    especially its Config and Connections are experimental and subject to
    change any time without a formal RFC or standard deprecation procedures
    until it is included in the DRP pipeline.
    """

    ConfigClass = AssembleCellCoaddConfig
    _DefaultName = "assembleCellCoadd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.do_interpolate_coadd:
            self.makeSubtask("interpolate_coadd")
        if self.config.do_scale_zero_point:
            self.makeSubtask("scale_zero_point")

        self.psf_warper = afwMath.Warper.fromConfig(self.config.psf_warper)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.
        if not inputRefs.inputWarps:
            raise NoWorkFound("No input warps provided for co-addition")
        self.log.info("Found %d input warps", len(inputRefs.inputWarps))

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData
        # needs it
        skyMap = butlerQC.get(inputRefs.skyMap)

        if not skyMap.config.tractBuilder.name == "cells":
            raise RuntimeError("AssembleCellCoaddTask requires a cell-based skymap.")

        outputDataId = butlerQC.quantum.dataId

        skyInfo = makeSkyInfo(skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"])
        visitSummaryList = butlerQC.get(getattr(inputRefs, "visitSummaryList", []))

        units = CoaddUnits.legacy if self.config.do_scale_zero_point else CoaddUnits.nJy
        self.common = CommonComponents(
            units=units,
            wcs=skyInfo.patchInfo.wcs,
            band=outputDataId.get("band", None),
            identifiers=PatchIdentifiers.from_data_id(outputDataId),
        )

        inputs: dict[DataCoordinate, WarpInputs] = {}
        for handle in butlerQC.get(inputRefs.inputWarps):
            inputs[handle.dataId] = WarpInputs(warp=handle, noise_warps=[])

        for ref in getattr(inputRefs, "artifactMasks", []):
            inputs[ref.dataId].artifact_mask = butlerQC.get(ref)
        for ref in getattr(inputRefs, "maskedFractionWarps", []):
            inputs[ref.dataId].masked_fraction = butlerQC.get(ref)
        for n in range(self.config.num_noise_realizations):
            for ref in getattr(inputRefs, f"noise{n}_warps"):
                inputs[ref.dataId].noise_warps.append(butlerQC.get(ref))

        returnStruct = self.run(inputs=inputs, skyInfo=skyInfo, visitSummaryList=visitSummaryList)
        butlerQC.put(returnStruct, outputRefs)
        return returnStruct

    @staticmethod
    def _compute_weight(maskedImage, statsCtrl):
        """Compute a weight for a masked image.

        Parameters
        ----------
        maskedImage : `~lsst.afw.image.MaskedImage`
            The masked image to compute the weight.
        statsCtrl : `~lsst.afw.math.StatisticsControl`
            A control (config-like) object for StatisticsStack.

        Returns
        -------
        weight : `float`
            Inverse of the clipped mean variance of the masked image.
        """
        statObj = afwMath.makeStatistics(
            maskedImage.getVariance(), maskedImage.getMask(), afwMath.MEANCLIP, statsCtrl
        )
        meanVar, _ = statObj.getResult(afwMath.MEANCLIP)
        weight = 1.0 / float(meanVar)
        return weight

    @staticmethod
    def _construct_grid(skyInfo):
        """Construct a UniformGrid object from a SkyInfo struct.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object

        Returns
        -------
        grid : `~lsst.cell_coadds.UniformGrid`
            A UniformGrid object.
        """
        padding = skyInfo.patchInfo.getCellBorder()
        grid_bbox = skyInfo.patchInfo.outer_bbox.erodedBy(padding)
        grid = UniformGrid.from_bbox_cell_size(
            grid_bbox,
            skyInfo.patchInfo.getCellInnerDimensions(),
            padding=padding,
        )
        return grid

    def _construct_grid_container(self, skyInfo, statsCtrl):
        """Construct a grid of AccumulatorMeanStack instances.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object
        statsCtrl : `~lsst.afw.math.StatisticsControl`
            A control (config-like) object for StatisticsStack.

        Returns
        -------
        gc : `~lsst.cell_coadds.GridContainer`
            A GridContainer object container one AccumulatorMeanStack per cell.
        """
        grid = self._construct_grid(skyInfo)

        maskMap = setRejectedMaskMapping(statsCtrl)
        self.log.debug("Obtained maskMap = %s for %s", maskMap, skyInfo.patchInfo)
        thresholdDict = AccumulatorMeanStack.stats_ctrl_to_threshold_dict(statsCtrl)

        # Initialize the grid container with AccumulatorMeanStacks
        gc = GridContainer[AccumulatorMeanStack](grid.shape)
        for cellInfo in skyInfo.patchInfo:
            stacker = AccumulatorMeanStack(
                # The shape is for the numpy arrays, hence transposed.
                shape=(cellInfo.outer_bbox.height, cellInfo.outer_bbox.width),
                bit_mask_value=statsCtrl.getAndMask(),
                mask_threshold_dict=thresholdDict,
                calc_error_from_input_variance=self.config.calc_error_from_input_variance,
                compute_n_image=False,
                mask_map=maskMap,
                no_good_pixels_mask=statsCtrl.getNoGoodPixelsMask(),
            )
            gc[cellInfo.index] = stacker

        return gc

    def _construct_stats_control(self):
        """Construct a StatisticsControl object for coadd.

        Unlike AssembleCoaddTask or CompareWarpAssembleCoaddTask, there is
        very little to be configured apart from setting the mask planes and
        optionally mask propagation thresholds.

        Returns
        -------
        statsCtrl : `~lsst.afw.math.StatisticsControl`
            A control object for StatisticsStack.
        """
        statsCtrl = afwMath.StatisticsControl()
        # Hardcode the numIter parameter to the default config value set in
        # CompareWarpAssembleCoaddTask to get consistent weights. This is NOT
        # exposed as a config parameter, since this is only meant to be a
        # fallback option that is not recommended for production.
        statsCtrl.setNumIter(2)
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes))
        statsCtrl.setNanSafe(True)
        for plane, threshold in self.config.mask_propagation_thresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        return statsCtrl

    def _construct_ap_corr_grid_container(self, skyInfo):
        """Construct a grid of CoaddApCorrMapStacker instances.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object

        Returns
        -------
        gc : `~lsst.cell_coadds.GridContainer`
            A GridContainer object container one CoaddApCorrMapStacker per
            cell.
        """
        grid = self._construct_grid(skyInfo)

        # Initialize the grid container with CoaddApCorrMapStacker.
        gc = GridContainer[CoaddApCorrMapStacker](grid.shape)
        for cellInfo in skyInfo.patchInfo:
            stacker = CoaddApCorrMapStacker(
                evaluation_point=cellInfo.inner_bbox.getCenter(),
                do_coadd_inverse_ap_corr=self.config.do_coadd_inverse_aperture_corrections,
            )
            gc[cellInfo.index] = stacker

        return gc

    def run(
        self,
        *,
        inputs: dict[DataCoordinate, WarpInputs],
        skyInfo,
        visitSummaryList: list | None = None,
    ):
        for mask_plane in self.config.bad_mask_planes:
            afwImage.Mask.addMaskPlane(mask_plane)
        for mask_plane in self.config.mask_propagation_thresholds:
            afwImage.Mask.addMaskPlane(mask_plane)

        statsCtrl = self._construct_stats_control()

        warp_stacker_gc = self._construct_grid_container(skyInfo, statsCtrl)
        maskfrac_stacker_gc = self._construct_grid_container(skyInfo, statsCtrl)
        noise_stacker_gc_list = [
            self._construct_grid_container(skyInfo, statsCtrl)
            for n in range(self.config.num_noise_realizations)
        ]
        psf_stacker_gc = GridContainer[AccumulatorMeanStack](warp_stacker_gc.shape)
        psf_bbox_gc = GridContainer[geom.Box2I](warp_stacker_gc.shape)
        ap_corr_stacker_gc = self._construct_ap_corr_grid_container(skyInfo)

        # Make a container to hold the cell centers in sky coordinates now,
        # so we don't have to recompute them for each warp
        # (they share a common WCS). These are needed to find the various
        # warp + detector combinations that contributed to each cell, and later
        # get the corresponding PSFs as well.
        cell_centers_sky = GridContainer[geom.SpherePoint](warp_stacker_gc.shape)
        # Make a container to hold the observation identifiers for each cell.
        observation_identifiers_gc = GridContainer[list](warp_stacker_gc.shape)
        # Populate them.
        for cellInfo in skyInfo.patchInfo:
            # Make a list to hold the observation identifiers for each cell.
            observation_identifiers_gc[cellInfo.index] = []
            cell_centers_sky[cellInfo.index] = skyInfo.wcs.pixelToSky(cellInfo.inner_bbox.getCenter())
            psf_bbox_gc[cellInfo.index] = geom.Box2I.makeCenteredBox(
                geom.Point2D(cellInfo.inner_bbox.getCenter()),
                geom.Extent2I(self.config.psf_dimensions, self.config.psf_dimensions),
            )
            psf_stacker_gc[cellInfo.index] = AccumulatorMeanStack(
                # The shape is for the numpy arrays, hence transposed.
                shape=(self.config.psf_dimensions, self.config.psf_dimensions),
                bit_mask_value=0,
                calc_error_from_input_variance=self.config.calc_error_from_input_variance,
                compute_n_image=False,
            )

        # visit_summary do not have (tract, patch, band, skymap) dimensions.
        if not visitSummaryList:
            visitSummaryList = []
        visitSummaryRefDict = {
            visitSummaryRef.dataId["visit"]: visitSummaryRef for visitSummaryRef in visitSummaryList
        }

        # Read in one warp at a time, and accumulate it in all the cells that
        # it completely overlaps.
        for _, warp_input in inputs.items():
            warp = warp_input.warp.get(parameters={"bbox": skyInfo.bbox})
            masked_fraction_image = (
                warp_input.masked_fraction.get(parameters={"bbox": skyInfo.bbox})
                if warp_input.masked_fraction
                else None
            )

            # Pre-process the warp before coadding.
            # TODO: Can we get these mask names from artifactMask?
            warp.mask.addMaskPlane("CLIPPED")
            warp.mask.addMaskPlane("REJECTED")
            warp.mask.addMaskPlane("SENSOR_EDGE")
            warp.mask.addMaskPlane("INEXACT_PSF")

            if artifact_mask_ref := warp_input.artifact_mask:
                # Apply the artifact mask to the warp.
                artifact_mask = artifact_mask_ref.get()
                assert (
                    warp.mask.getMaskPlaneDict() == artifact_mask.getMaskPlaneDict()
                ), "Mask dicts do not agree."
                warp.mask.array = artifact_mask.array
                del artifact_mask

            if self.config.do_scale_zero_point:
                # Each Warp that goes into a coadd will typically have an
                # independent photometric zero-point. Therefore, we must scale
                # each Warp to set it to a common photometric zeropoint.
                imageScaler = self.scale_zero_point.run(exposure=warp, dataRef=warp_input.warp).imageScaler
                zero_point_scale_factor = imageScaler.scale
                self.log.debug(
                    "Scaled the warp %s by %f to match zero points",
                    warp_input.dataId,
                    zero_point_scale_factor,
                )
            else:
                zero_point_scale_factor = 1.0
                if "BUNIT" not in warp.metadata:
                    raise ValueError(f"Warp {warp_input.dataId} has no BUNIT metadata")
                if warp.metadata["BUNIT"] != "nJy":
                    raise ValueError(
                        f"Warp {warp_input.dataId} has BUNIT {warp.metadata['BUNIT']}, expected nJy"
                    )

            # Coadd the warp onto the cells it completely overlaps.
            edge = afwImage.Mask.getPlaneBitMask(["NO_DATA", "SENSOR_EDGE"])
            reject = afwImage.Mask.getPlaneBitMask(["CLIPPED", "REJECTED"])
            removeMaskPlanes(warp.mask, self.config.remove_mask_planes, self.log)

            # Compute the weight for each CCD in the warp from the visitSummary
            # or from the warp itself, if not provided. Computing the weight
            # from the warp is not recommended, and in that case we compute one
            # weight per warp and not bother with per-detector weights.
            full_ccd_table = warp.getInfo().getCoaddInputs().ccds
            weights: dict[int, float] = dict.fromkeys(
                full_ccd_table["ccd"],
                0.0,
            )  # Mapping from detector to weight.

            if visitSummaryRef := visitSummaryRefDict.get(warp_input.dataId["visit"]):
                visitSummary = visitSummaryRef.get()
                for detector in full_ccd_table["ccd"]:
                    visitSummaryRow = visitSummary.find(detector)
                    mean_variance = visitSummaryRow["meanVar"]
                    mean_variance *= zero_point_scale_factor**2
                    if warp.metadata.get("BUNIT", None) == "nJy":
                        mean_variance *= visitSummaryRow.photoCalib.getCalibrationMean() ** 2
                    weights[detector] = 1.0 / mean_variance
                del visitSummary
            else:
                self.log.debug("No visit summary found for %s; using warp-based weights", warp_input.dataId)
                weight = self._compute_weight(warp, statsCtrl)
                if not np.isfinite(weight):
                    self.log.warn("Non-finite weight for %s: skipping", warp_input.dataId)
                    continue

                for detector in weights:
                    weights[detector] = weight

            noise_warps = [ref.get(parameters={"bbox": skyInfo.bbox}) for ref in warp_input.noise_warps]

            for cellInfo in skyInfo.patchInfo:
                bbox = cellInfo.outer_bbox
                inner_bbox = cellInfo.inner_bbox
                mi = warp[bbox].maskedImage

                if (mi.mask[inner_bbox].array & edge).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has a pixel with SENSOR_EDGE or NO_DATA bit set",
                        warp_input.dataId,
                        cellInfo.index,
                    )
                    continue

                if (mi.mask[inner_bbox].array & reject).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has a pixel with CLIPPED or REJECTED bit set",
                        warp_input.dataId,
                        cellInfo.index,
                    )
                    continue

                # Find the CCD that contributed to this cell.
                if len(warp.getInfo().getCoaddInputs().ccds) == 1:
                    # If there is only one, don't bother with a WCS look up.
                    ccd_row = full_ccd_table[0]
                else:
                    ccd_table = full_ccd_table.subsetContaining(cell_centers_sky[cellInfo.index])

                    if len(ccd_table) == 0:
                        # This condition rarely occurs in test runs, if ever.
                        # But the QG generated upfront in campaign management
                        # land routinely has extra quanta that should be
                        # dropped during runtime. These cases arise when
                        # the tasks upstream didn't process.
                        # See DM-52306 for example.
                        self.log.debug("No CCD found for %s in cell %s", warp_input.dataId, cellInfo.index)
                        continue

                    assert len(ccd_table) == 1, "More than one CCD from a warp found within a cell."
                    ccd_row = ccd_table[0]

                weight = weights[ccd_row["ccd"]]
                if not np.isfinite(weight):
                    self.log.warn(
                        "Non-finite weight for %s in cell %s: skipping", warp_input.dataId, cellInfo.index
                    )
                    continue

                if weight == 0:
                    self.log.info(
                        "Zero weight for %s in cell %s: skipping", warp_input.dataId, cellInfo.index
                    )
                    continue

                observation_identifier = ObservationIdentifiers.from_data_id(
                    warp_input.dataId,
                    backup_detector=ccd_row["ccd"],
                )
                observation_identifiers_gc[cellInfo.index].append(observation_identifier)

                stacker = warp_stacker_gc[cellInfo.index]
                stacker.add_masked_image(mi, weight=weight)
                if masked_fraction_image:
                    maskfrac_stacker_gc[cellInfo.index].add_image(masked_fraction_image[bbox], weight=weight)

                for n in range(self.config.num_noise_realizations):
                    mi = noise_warps[n][bbox]
                    noise_stacker_gc_list[n][cellInfo.index].add_masked_image(mi, weight=weight)

                calexp_point = ccd_row.getWcs().skyToPixel(cell_centers_sky[cellInfo.index])
                undistorted_psf_im = ccd_row.getPsf().computeImage(calexp_point)

                assert undistorted_psf_im.getBBox() == geom.Box2I.makeCenteredBox(
                    calexp_point,
                    undistorted_psf_im.getDimensions(),
                ), "PSF image does not share the coordinates of the 'calexp'"

                # Convert the PSF image from Image to MaskedImage.
                undistorted_psf_maskedImage = afwImage.MaskedImageD(image=undistorted_psf_im)
                # TODO: In DM-43585, use the variance plane value from noise.
                undistorted_psf_maskedImage.variance += 1.0  # Set variance to 1

                warped_psf_maskedImage = self.psf_warper.warpImage(
                    destWcs=skyInfo.wcs,
                    srcImage=undistorted_psf_maskedImage,
                    srcWcs=ccd_row.getWcs(),
                    destBBox=psf_bbox_gc[cellInfo.index],
                )

                # There may be NaNs in the PSF image. Set them to 0.0
                warped_psf_maskedImage.variance.array[np.isnan(warped_psf_maskedImage.image.array)] = 1.0
                warped_psf_maskedImage.image.array[np.isnan(warped_psf_maskedImage.image.array)] = 0.0

                psf_stacker = psf_stacker_gc[cellInfo.index]
                psf_stacker.add_masked_image(warped_psf_maskedImage, weight=weight)

                if (ap_corr_map := warp.getInfo().getApCorrMap()) is not None:
                    ap_corr_stacker_gc[cellInfo.index].add(ap_corr_map, weight=weight)

            del warp

        cells: list[SingleCellCoadd] = []
        for cellInfo in skyInfo.patchInfo:
            if len(observation_identifiers_gc[cellInfo.index]) == 0:
                self.log.debug("Skipping cell %s because it has no input warps", cellInfo.index)
                continue

            cell_masked_image = afwImage.MaskedImageF(cellInfo.outer_bbox)
            cell_maskfrac_image = afwImage.ImageF(cellInfo.outer_bbox)
            cell_noise_images = [
                afwImage.MaskedImageF(cellInfo.outer_bbox) for n in range(self.config.num_noise_realizations)
            ]
            psf_masked_image = afwImage.MaskedImageF(psf_bbox_gc[cellInfo.index])

            warp_stacker_gc[cellInfo.index].fill_stacked_masked_image(cell_masked_image)
            maskfrac_stacker_gc[cellInfo.index].fill_stacked_image(cell_maskfrac_image)
            for n in range(self.config.num_noise_realizations):
                noise_stacker_gc_list[n][cellInfo.index].fill_stacked_masked_image(cell_noise_images[n])
            psf_stacker_gc[cellInfo.index].fill_stacked_masked_image(psf_masked_image)

            if ap_corr_stacker_gc[cellInfo.index].ap_corr_names:
                ap_corr_map = ap_corr_stacker_gc[cellInfo.index].final_ap_corr_map
            else:
                ap_corr_map = None

            # Post-process the coadd before converting to new data structures.
            if self.config.do_interpolate_coadd:
                self.interpolate_coadd.run(cell_masked_image, planeName="NO_DATA")
                for noise_image in cell_noise_images:
                    self.interpolate_coadd.run(noise_image, planeName="NO_DATA")
                # The variance must be positive; work around for DM-3201.
                varArray = cell_masked_image.variance.array
                with np.errstate(invalid="ignore"):
                    varArray[:] = np.where(varArray > 0, varArray, np.inf)

            image_planes = OwnedImagePlanes.from_masked_image(
                masked_image=cell_masked_image,
                mask_fractions=cell_maskfrac_image,
                noise_realizations=[noise_image.image for noise_image in cell_noise_images],
            )
            identifiers = CellIdentifiers(
                cell=cellInfo.index,
                skymap=self.common.identifiers.skymap,
                tract=self.common.identifiers.tract,
                patch=self.common.identifiers.patch,
                band=self.common.identifiers.band,
            )

            singleCellCoadd = SingleCellCoadd(
                outer=image_planes,
                psf=psf_masked_image.image,
                inner_bbox=cellInfo.inner_bbox,
                inputs=observation_identifiers_gc[cellInfo.index],
                common=self.common,
                identifiers=identifiers,
                aperture_correction_map=ap_corr_map,
            )
            # TODO: Attach transmission curve when they become available.
            cells.append(singleCellCoadd)

        if not cells:
            raise EmptyCellCoaddError()

        grid = self._construct_grid(skyInfo)
        multipleCellCoadd = MultipleCellCoadd(
            cells,
            grid=grid,
            outer_cell_size=cellInfo.outer_bbox.getDimensions(),
            inner_bbox=None,
            common=self.common,
            psf_image_size=cells[0].psf_image.getDimensions(),
        )

        return Struct(
            multipleCellCoadd=multipleCellCoadd,
        )


class ConvertMultipleCellCoaddToExposureConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={"inputCoaddName": "deep", "inputCoaddSuffix": "Cell"},
):
    cellCoaddExposure = Input(
        doc="Output coadded exposure, produced by stacking input warps",
        name="{inputCoaddName}Coadd{inputCoaddSuffix}",
        storageClass="MultipleCellCoadd",
        dimensions=("tract", "patch", "skymap", "band"),
    )

    stitchedCoaddExposure = Output(
        doc="Output stitched coadded exposure, produced by stacking input warps",
        name="{inputCoaddName}Coadd{inputCoaddSuffix}_stitched",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
    )


class ConvertMultipleCellCoaddToExposureConfig(
    PipelineTaskConfig, pipelineConnections=ConvertMultipleCellCoaddToExposureConnections
):
    """A trivial PipelineTaskConfig class for
    ConvertMultipleCellCoaddToExposureTask.
    """


class ConvertMultipleCellCoaddToExposureTask(PipelineTask):
    """An after burner PipelineTask that converts a cell-based coadd from
    `MultipleCellCoadd` format to `ExposureF` format.

    The run method stitches the cell-based coadd into contiguous exposure and
    returns it in as an `Exposure` object. This is lossy as it preserves only
    the pixels in the inner bounding box of the cells and discards the values
    in the buffer region.

    Notes
    -----
    This task has no configurable parameters.
    """

    ConfigClass = ConvertMultipleCellCoaddToExposureConfig
    _DefaultName = "convertMultipleCellCoaddToExposure"

    def run(self, cellCoaddExposure):
        return Struct(
            stitchedCoaddExposure=cellCoaddExposure.stitch().asExposure(),
        )
