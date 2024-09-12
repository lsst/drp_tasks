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


import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.geom as geom
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
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pex.config import ConfigField, ConfigurableField, Field, ListField, RangeField
from lsst.pipe.base import NoWorkFound, PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import makeSkyInfo
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

    artifactMasks = Input(
        doc="Artifact masks to be applied to the input warps",
        name="compare_warp_artifact_mask",
        storageClass="Mask",
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
        doc="Scale warps to a common zero point? This is not needed if they have absolute flux calibration.",
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
        default=True,
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
        if self.config.do_interpolate_coadd:
            self.makeSubtask("interpolate_coadd")
        if self.config.do_scale_zero_point:
            self.makeSubtask("scale_zero_point")

        self.psf_warper = afwMath.Warper.fromConfig(self.config.psf_warper)

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

    def _construct_grid_container(self, skyInfo):
        """Construct a grid of AccumulatorMeanStack instances.

        Parameters
        ----------
        skyInfo : `~lsst.pipe.base.Struct`
            A Struct object

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
                bit_mask_value=0,
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

        gc = self._construct_grid_container(skyInfo)
        psf_gc = GridContainer[AccumulatorMeanStack](gc.shape)
        psf_bbox_gc = GridContainer[geom.Box2I](gc.shape)

        # Make a container to hold the cell centers in sky coordinates now,
        # so we don't have to recompute them for each warp
        # (they share a common WCS). These are needed to find the various
        # warp + detector combinations that contributed to each cell, and later
        # get the corresponding PSFs as well.
        cell_centers_sky = GridContainer[geom.SpherePoint](gc.shape)
        # Make a container to hold the observation identifiers for each cell.
        observation_identifiers_gc = GridContainer[list](gc.shape)
        # Populate them.
        for cellInfo in skyInfo.patchInfo:
            # Make a list to hold the observation identifiers for each cell.
            observation_identifiers_gc[cellInfo.index] = []
            cell_centers_sky[cellInfo.index] = skyInfo.wcs.pixelToSky(cellInfo.inner_bbox.getCenter())
            psf_bbox_gc[cellInfo.index] = geom.Box2I.makeCenteredBox(
                geom.Point2D(cellInfo.inner_bbox.getCenter()),
                geom.Extent2I(self.config.psf_dimensions, self.config.psf_dimensions),
            )
            psf_gc[cellInfo.index] = AccumulatorMeanStack(
                # The shape is for the numpy arrays, hence transposed.
                shape=(self.config.psf_dimensions, self.config.psf_dimensions),
                bit_mask_value=0,
                calc_error_from_input_variance=self.config.calc_error_from_input_variance,
                compute_n_image=False,
            )

        artifactMasks = kwargs.get("artifactMasks", [None] * len(inputWarps))

        # Read in one warp at a time, and accumulate it in all the cells that
        # it completely overlaps.
        for warpRef, artifactMaskRef in zip(inputWarps, artifactMasks):
            warp = warpRef.get()

            warp.mask.addMaskPlane("CLIPPED")
            warp.mask.addMaskPlane("REJECTED")

            if artifactMaskRef is not None:
                # Apply the artifact mask to the warp.
                artifactMask = artifactMaskRef.get()
                warp.mask.array |= artifactMask.array
                del artifactMask

            # Pre-process the warp before coadding.
            # Each Warp that goes into a coadd will typically have an
            # independent photometric zero-point. Therefore, we must scale each
            # Warp to set it to a common photometric zeropoint.
            if self.config.do_scale_zero_point:
                self.scale_zero_point.run(exposure=warp, dataRef=warpRef)

            # Coadd the warp onto the cells it completely overlaps.
            edge = afwImage.Mask.getPlaneBitMask("EDGE")
            reject = afwImage.Mask.getPlaneBitMask(["CLIPPED", "REJECTED"])
            for cellInfo in skyInfo.patchInfo:
                bbox = cellInfo.outer_bbox
                mi = warp[bbox].getMaskedImage()

                if (mi.getMask().array & edge).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has an EDGE bit set",
                        warpRef.dataId,
                        cellInfo.index,
                    )
                    continue

                if (mi.getMask().array & reject).any():
                    self.log.debug(
                        "Skipping %s in cell %s because it has a CLIPPED or REJECTED bit set",
                        warpRef.dataId,
                        cellInfo.index,
                    )
                    continue

                weight = self._compute_weight(mi, statsCtrl)
                if not np.isfinite(weight):
                    # Log at the debug level, because this can be quite common.
                    self.log.debug(
                        "Non-finite weight for %s in cell %s: skipping", warpRef.dataId, cellInfo.index
                    )
                    continue

                ccd_table = (
                    warp.getInfo().getCoaddInputs().ccds.subsetContaining(cell_centers_sky[cellInfo.index])
                )
                assert len(ccd_table) > 0, "No CCD from a warp found within a cell."
                assert len(ccd_table) == 1, "More than one CCD from a warp found within a cell."
                ccd_row = ccd_table[0]

                observation_identifier = ObservationIdentifiers.from_data_id(
                    warpRef.dataId,
                    backup_detector=ccd_row["ccd"],
                )
                observation_identifiers_gc[cellInfo.index].append(observation_identifier)

                stacker = gc[cellInfo.index]
                stacker.add_masked_image(mi, weight=weight)

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

                psf_stacker = psf_gc[cellInfo.index]
                psf_stacker.add_masked_image(warped_psf_maskedImage, weight=weight)

            del warp

        cells: list[SingleCellCoadd] = []
        for cellInfo in skyInfo.patchInfo:
            if len(observation_identifiers_gc[cellInfo.index]) == 0:
                self.log.debug("Skipping cell %s because it has no input warps", cellInfo.index)
                continue

            stacker = gc[cellInfo.index]
            cell_masked_image = afwImage.MaskedImageF(cellInfo.outer_bbox)
            psf_masked_image = afwImage.MaskedImageF(psf_bbox_gc[cellInfo.index])
            gc[cellInfo.index].fill_stacked_masked_image(cell_masked_image)
            psf_gc[cellInfo.index].fill_stacked_masked_image(psf_masked_image)

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
                psf=psf_masked_image.image,
                inner_bbox=cellInfo.inner_bbox,
                inputs=observation_identifiers_gc[cellInfo.index],
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
