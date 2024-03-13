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
    "AssembleCellCoaddTask",
    "AssembleCellCoaddConfig",
    "ConvertMultipleCellCoaddToExposureTask",
)


import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import numpy as np
from lsst.cell_coadds import (
    CellIdentifiers,
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
from lsst.meas.algorithms import AccumulatorMeanStack, CoaddPsf, CoaddPsfConfig
from lsst.pex.config import ConfigField, ConfigurableField, Field, ListField, RangeField
from lsst.pipe.base import NoWorkFound, PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import makeSkyInfo
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.pipe.tasks.scaleZeroPoint import ScaleZeroPointTask
from lsst.skymap import BaseSkyMap


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


class AssembleCellCoaddConfig(PipelineTaskConfig, pipelineConnections=AssembleCellCoaddConnections):
    do_interpolate_coadd = Field[bool](doc="Interpolate over pixels with NO_DATA mask set?", default=False)
    interpolate_coadd = ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over pixels with NO_DATA mask on cell coadds",
    )
    do_scale_zero_point = Field[bool](
        doc="Scale warps to a common zero point? This is not needed they have absolute flux calibration.",
        default=False,
    )
    scale_zero_point = ConfigurableField(
        target=ScaleZeroPointTask,
        doc="Task to scale warps to a common zero point",
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that count towards the masked fraction within a cell.",
        default=("BAD", "NO_DATA", "SAT"),
    )
    calc_error_from_input_variance = Field[bool](
        doc="Calculate coadd variance from input variance by stacking "
        "statistic. Passed to AccumulatorMeanStack.",
        default=False,
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
    # The following config options are specific to the CoaddPsf.
    coadd_psf = ConfigField(
        doc="Configuration for CoaddPsf",
        dtype=CoaddPsfConfig,
    )
    input_recorder = ConfigurableField(
        doc="Subtask that helps fill CoaddInputs catalogs added to the final Exposure",
        target=CoaddInputRecorderTask,
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
        Raised if no input warps are provided.
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
        self.makeSubtask("input_recorder")
        if self.config.do_interpolate_coadd:
            self.makeSubtask("interpolate_coadd")
        if self.config.do_scale_zero_point:
            self.makeSubtask("scale_zero_point")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.
        inputData = butlerQC.get(inputRefs)

        if not inputData["inputWarps"]:
            raise NoWorkFound("No input warps provided for co-addition")
        self.log.info("Found %d input warps", len(inputData["inputWarps"]))

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case _makeSupplementaryData
        # needs it
        skyMap = inputData["skyMap"]

        if not skyMap.config.tractBuilder.name == "cells":
            raise RuntimeError("AssembleCellCoaddTask requires a cell-based skymap.")

        outputDataId = butlerQC.quantum.dataId

        inputData["skyInfo"] = makeSkyInfo(
            skyMap, tractId=outputDataId["tract"], patchId=outputDataId["patch"]
        )

        self.common = CommonComponents(
            units=CoaddUnits.legacy,  # until the ScaleZeroPointTask can scale it to nJy.
            wcs=inputData["skyInfo"].patchInfo.wcs,
            band=outputDataId.get("band", None),
            identifiers=PatchIdentifiers.from_data_id(outputDataId),
        )

        returnStruct = self.run(**inputData)
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
        # grid has no notion about border or inner/outer boundaries.
        # So we have to clip the outermost border when constructing the grid.
        grid_bbox = skyInfo.patchInfo.outer_bbox.erodedBy(skyInfo.patchInfo.getCellBorder())
        grid = UniformGrid.from_bbox_cell_size(grid_bbox, skyInfo.patchInfo.getCellInnerDimensions())
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

        # Initialize the grid container with AccumulatorMeanStacks
        gc = GridContainer[AccumulatorMeanStack](grid.shape)
        for cellInfo in skyInfo.patchInfo:
            stacker = AccumulatorMeanStack(
                # The shape is for the numpy arrays, hence transposed.
                shape=(cellInfo.outer_bbox.height, cellInfo.outer_bbox.width),
                bit_mask_value=afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes),
                calc_error_from_input_variance=self.config.calc_error_from_input_variance,
                compute_n_image=False,
            )
            gc[cellInfo.index] = stacker

        return gc

    def _construct_stats_control(self):
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes))
        statsCtrl.setNanSafe(True)
        return statsCtrl

    def run(self, inputWarps, skyInfo, **kwargs):
        statsCtrl = self._construct_stats_control()

        gc = self._construct_grid_container(skyInfo, statsCtrl)
        coadd_inputs_gc = GridContainer(gc.shape)
        observation_identifiers_gc = GridContainer(gc.shape)
        for cellInfo in skyInfo.patchInfo:
            coadd_inputs = self.input_recorder.makeCoaddInputs()
            # Reserve the absolute maximum of how many ccds, visits
            # we could potentially have.
            coadd_inputs.ccds.reserve(len(inputWarps))
            coadd_inputs.visits.reserve(len(inputWarps))
            coadd_inputs_gc[cellInfo.index] = coadd_inputs
            # Make a list to hold the observation identifiers for each cell.
            observation_identifiers_gc[cellInfo.index] = []
        # Read in one warp at a time, and accumulate it in all the cells that
        # it completely overlaps.

        for warpRef in inputWarps:
            warp = warpRef.get()

            # Pre-process the warp before coadding.
            # Each Warp that goes into a coadd will typically have an
            # independent photometric zero-point. Therefore, we must scale each
            # Warp to set it to a common photometric zeropoint.
            if self.config.do_scale_zero_point:
                self.scale_zero_point.run(exposure=warp, dataRef=warpRef)

            # Coadd the warp onto the cells it completely overlaps.
            edge = afwImage.Mask.getPlaneBitMask("EDGE")
            for cellInfo in skyInfo.patchInfo:
                bbox = cellInfo.outer_bbox
                stacker = gc[cellInfo.index]
                mi = warp[bbox].getMaskedImage()

                if (mi.getMask().array & edge).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has an EDGE", warpRef.dataId, cellInfo.index
                    )
                    continue

                weight = self._compute_weight(mi, statsCtrl)
                if not np.isfinite(weight):
                    # Log at the debug level, because this can be quite common.
                    self.log.debug(
                        "Non-finite weight for %s in cell %s: skipping", warpRef.dataId, cellInfo.index
                    )
                    continue

                stacker.add_masked_image(mi, weight=weight)

                coadd_inputs = coadd_inputs_gc[cellInfo.index]
                self.input_recorder.addVisitToCoadd(coadd_inputs, warp[bbox], weight)
                assert len(coadd_inputs.ccds) <= 1, "More than one CCD from a warp found within a cell."
                observation_identifier = ObservationIdentifiers.from_data_id(
                    warpRef.dataId,
                    detector=coadd_inputs.ccds[0]["ccd"],
                )
                observation_identifiers_gc[cellInfo.index].append(observation_identifier)

            del warp

        cells: list[SingleCellCoadd] = []
        for cellInfo in skyInfo.patchInfo:
            coadd_inputs = coadd_inputs_gc[cellInfo.index]

            if len(coadd_inputs.ccds) == 0:
                self.log.info("Skipping cell %s because it has no input warps", cellInfo.index)
                continue

            # Finalize the PSF on the cell coadds.
            coadd_inputs.ccds.sort()
            coadd_inputs.visits.sort()
            cell_coadd_psf = CoaddPsf(coadd_inputs.ccds, skyInfo.wcs, self.config.coadd_psf.makeControl())

            stacker = gc[cellInfo.index]
            cell_masked_image = afwImage.MaskedImageF(cellInfo.outer_bbox)
            stacker.fill_stacked_masked_image(cell_masked_image)

            # Post-process the coadd before converting to new data structures.
            if self.config.do_interpolate_coadd:
                self.interpolate_coadd.run(cell_masked_image, planeName="NO_DATA")
                # The variance must be positive; work around for DM-3201.
                varArray = cell_masked_image.variance.array
                with np.errstate(invalid="ignore"):
                    varArray[:] = np.where(varArray > 0, varArray, np.inf)

            image_planes = OwnedImagePlanes.from_masked_image(cell_masked_image)
            identifiers = CellIdentifiers(
                cell=cellInfo.index,
                skymap=self.common.identifiers.skymap,
                tract=self.common.identifiers.tract,
                patch=self.common.identifiers.patch,
                band=self.common.identifiers.band,
            )

            singleCellCoadd = SingleCellCoadd(
                outer=image_planes,
                psf=cell_coadd_psf.computeKernelImage(cell_coadd_psf.getAveragePosition()),
                inner_bbox=cellInfo.inner_bbox,
                inputs=frozenset(observation_identifiers_gc[cellInfo.index]),
                common=self.common,
                identifiers=identifiers,
            )
            # TODO: Attach transmission curve when they become available.
            cells.append(singleCellCoadd)

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


class ConvertMulipleCellCoaddToExposureConnections(
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
    PipelineTaskConfig, pipelineConnections=ConvertMulipleCellCoaddToExposureConnections
):
    """A trivial PipelineTaskConfig class for
    ConvertMultipleCellCoaddToExposureTask.
    """

    pass


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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputData = butlerQC.get(inputRefs)
        returnStruct = self.run(**inputData)
        butlerQC.put(returnStruct, outputRefs)

    def run(self, cellCoaddExposure):
        return Struct(
            stitchedCoaddExposure=cellCoaddExposure.stitch().asExposure(),
        )
