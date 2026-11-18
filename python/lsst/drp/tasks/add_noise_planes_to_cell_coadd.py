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
    "AddNoisePlanesToCellCoaddTask",
    "AddNoisePlanesToCellCoaddConfig",
)

import logging

import numpy as np

import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
from lsst.cell_coadds import (
    MultipleCellCoadd,
    OwnedImagePlanes,
    SingleCellCoadd,
)
from lsst.meas.algorithms import AccumulatorMeanStack
from lsst.pex.config import (
    ConfigurableField,
    DictField,
    Field,
    ListField,
)
from lsst.pipe.base import (
    NoWorkFound,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import setRejectedMaskMapping
from lsst.pipe.tasks.interpImage import InterpImageTask
from lsst.skymap import BaseSkyMap


class AddNoisePlanesToCellCoaddConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
):
    inputCoadd = Input(
        doc="Input cell-based coadd to which additional noise planes are added.",
        name="deep_coadd",
        storageClass="CellCoadd",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    visitSummaryList = Input(
        doc="Input visit-summary catalogs with updated calibration objects. Only used as a fallback to "
        "compute coadd weights when the weight stored in the coadd provenance is missing or invalid.",
        name="visit_summary",
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

    outputCoadd = Output(
        doc="Output cell-based coadd with additional coadded noise planes appended.",
        name="deep_coadd_with_additional_noise_planes",
        storageClass="CellCoadd",
        dimensions=("tract", "patch", "band", "skymap"),
    )

    config: AddNoisePlanesToCellCoaddConfig

    def __init__(self, *, config: AddNoisePlanesToCellCoaddConfig | None = None):
        super().__init__(config=config)

        if not self.config:
            return

        # Dynamically set input connections for noise warps, depending on the
        # number of noise realizations specified in the config. These mirror
        # the outputs of ``MakeDirectWarpTask`` and the inputs of
        # ``AssembleCellCoaddTask``.
        for n in range(self.config.num_noise_realizations):
            noise_warps = Input(
                doc=f"Input noise warps ({n})",
                name=f"direct_warp_noise{n}",
                storageClass="MaskedImageF",
                dimensions=("tract", "patch", "skymap", "visit", "instrument"),
                deferLoad=True,
                multiple=True,
            )
            setattr(self, f"noise{n}_warps", noise_warps)


class AddNoisePlanesToCellCoaddConfig(
    PipelineTaskConfig, pipelineConnections=AddNoisePlanesToCellCoaddConnections
):
    num_noise_realizations = Field[int](
        default=1,
        doc=(
            "Number of noise warp realizations to coadd and add as additional noise planes. "
            "This should not exceed the number of noise realizations persisted by `MakeDirectWarpTask`."
        ),
        check=lambda x: x >= 0,
    )
    append_to_existing = Field[bool](
        default=True,
        doc="Append the newly coadded noise planes to any existing noise realizations in the input coadd? "
        "If False, the existing noise realizations are discarded and replaced.",
    )
    do_interpolate_coadd = Field[bool](
        doc="Interpolate over pixels with NO_DATA mask set in the coadded noise planes?",
        default=True,
    )
    interpolate_coadd = ConfigurableField(
        target=InterpImageTask,
        doc="Task to interpolate (and extrapolate) over pixels with NO_DATA mask on cell coadds.",
    )
    calc_error_from_input_variance = Field[bool](
        doc="Calculate coadd variance from input variance by stacking statistic. Passed to "
        "AccumulatorMeanStack. This should match the configuration used for the science coadd.",
        default=True,
    )
    bad_mask_planes = ListField[str](
        doc="Mask planes that are rejected when coadding the noise warps. This should match the "
        "configuration used for the science coadd.",
        default=("BAD", "NO_DATA", "SAT", "CLIPPED"),
    )
    mask_propagation_thresholds = DictField[str, float](
        doc=(
            "Threshold (in fractional weight) of rejection at which we propagate a mask plane to the "
            "coadd; that is, we set the mask bit on the coadd if the fraction the rejected frames would "
            "have contributed exceeds this value. This should match the science coadd configuration."
        ),
        default={"SAT": 0.1},
    )


class AddNoisePlanesToCellCoaddTask(PipelineTask):
    """Add additional coadded noise planes to an existing cell-based coadd.

    This task consumes an existing cell-based coadd (``deep_coadd``), the
    per-visit noise warps that were (or could have been) used to build it, and
    the visit-summary catalogs. For each cell in the coadd, it uses the coadd's
    own provenance to determine which ``(visit, detector)`` combinations
    contributed to that cell and with what weight, and coadds the corresponding
    noise warps with those exact weights. The resulting noise images are
    appended as additional noise planes and written out as
    ``deep_coadd_with_additional_noise_planes``.

    Because the per-cell, per-detector weights and the detector footprint
    polygons (in coadd coordinates) are recorded in the coadd provenance, the
    noise stack reproduces the science coadd without re-deriving detector maps,
    overlap fractions, or the fallback logic in `AssembleCellCoaddTask`.

    Raises
    ------
    NoWorkFound
        Raised if no noise realizations are configured.
    RuntimeError
        Raised if the skymap is not cell-based.

    Notes
    -----
    Much of the per-cell stacking logic mirrors `AssembleCellCoaddTask`. The
    one deliberate difference is that each contribution is always restricted to
    its provenance detector polygon (equivalent to the ``deep_copy`` branch in
    `AssembleCellCoaddTask`), which keeps the noise stack consistent with the
    per-detector weights regardless of how many detectors overlap a cell.
    """

    ConfigClass = AddNoisePlanesToCellCoaddConfig
    _DefaultName = "addNoisePlanesToCellCoadd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.do_interpolate_coadd:
            self.makeSubtask("interpolate_coadd")
            # Suppress the warning message about fallback.
            self.interpolate_coadd.log.setLevel(logging.ERROR)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.
        if self.config.num_noise_realizations == 0:
            raise NoWorkFound("No noise realizations configured; nothing to add.")

        skyMap = butlerQC.get(inputRefs.skyMap)
        if not skyMap.config.tractBuilder.name == "cells":
            raise RuntimeError("AddNoisePlanesToCellCoaddTask requires a cell-based skymap.")

        outputDataId = butlerQC.quantum.dataId
        tract_info = skyMap[outputDataId["tract"]]

        cell_coadd = butlerQC.get(inputRefs.inputCoadd)

        # visit-summary refs are keyed by visit for the weight fallback.
        visit_summary_dict = {
            ref.dataId["visit"]: ref for ref in butlerQC.get(getattr(inputRefs, "visitSummaryList", []))
        }

        # One mapping of visit -> noise-warp handle per noise realization.
        noise_handles: list[dict[int, object]] = []
        for n in range(self.config.num_noise_realizations):
            noise_handles.append(
                {ref.dataId["visit"]: ref for ref in butlerQC.get(getattr(inputRefs, f"noise{n}_warps"))}
            )

        returnStruct = self.run(
            cell_coadd=cell_coadd,
            tract_info=tract_info,
            noise_handles=noise_handles,
            visit_summary_dict=visit_summary_dict,
        )
        butlerQC.put(returnStruct, outputRefs)
        return returnStruct

    def _construct_stats_control(self):
        """Construct a StatisticsControl object for the noise coadd.

        This mirrors `AssembleCellCoaddTask._construct_stats_control` so that
        the noise stack uses the same rejection and weighting behavior as the
        science coadd.

        Returns
        -------
        statsCtrl : `~lsst.afw.math.StatisticsControl`
            A control object for StatisticsStack.
        """
        statsCtrl = afwMath.StatisticsControl()
        statsCtrl.setNumIter(2)
        statsCtrl.setAndMask(afwImage.Mask.getPlaneBitMask(self.config.bad_mask_planes))
        statsCtrl.setNanSafe(True)
        for plane, threshold in self.config.mask_propagation_thresholds.items():
            bit = afwImage.Mask.getMaskPlane(plane)
            statsCtrl.setMaskPropagationThreshold(bit, threshold)
        return statsCtrl

    def _make_stacker(self, bbox, statsCtrl, maskMap, thresholdDict):
        """Construct an AccumulatorMeanStack for a single cell.

        Parameters
        ----------
        bbox : `~lsst.geom.Box2I`
            Bounding box (the cell outer bbox) that the stacker covers.
        statsCtrl : `~lsst.afw.math.StatisticsControl`
            A control object for StatisticsStack.
        maskMap : `list`
            Mask mapping obtained from `setRejectedMaskMapping`.
        thresholdDict : `dict`
            Mask-propagation threshold dict for the stacker.

        Returns
        -------
        stacker : `~lsst.meas.algorithms.AccumulatorMeanStack`
            An accumulator sized to ``bbox``.
        """
        return AccumulatorMeanStack(
            # The shape is for the numpy arrays, hence transposed.
            shape=(bbox.height, bbox.width),
            bit_mask_value=statsCtrl.getAndMask(),
            mask_threshold_dict=thresholdDict,
            calc_error_from_input_variance=self.config.calc_error_from_input_variance,
            compute_n_image=False,
            mask_map=maskMap,
            no_good_pixels_mask=statsCtrl.getNoGoodPixelsMask(),
        )

    def _fallback_weight(self, obs_id, visit_summary_dict):
        """Compute a coadd weight for a ``(visit, detector)`` from the
        visit-summary catalog.

        This is only used when the weight recorded in the coadd provenance is
        missing or invalid. It mirrors the weight computation in
        `AssembleCellCoaddTask.run`.

        Parameters
        ----------
        obs_id : `~lsst.cell_coadds.ObservationIdentifiers`
            Identifiers of the contribution whose weight is needed.
        visit_summary_dict : `dict`
            Mapping from visit to a deferred visit-summary handle.

        Returns
        -------
        weight : `float` or `None`
            The computed weight, or `None` if it could not be computed.
        """
        ref = visit_summary_dict.get(obs_id.visit)
        if ref is None:
            return None
        visit_summary = ref.get()
        row = visit_summary.find(obs_id.detector)
        if row is None:
            return None
        mean_variance = row["meanVar"]
        # Coadds are in nJy, so scale the variance by the calibration.
        mean_variance *= row.photoCalib.getCalibrationMean() ** 2
        if not np.isfinite(mean_variance) or mean_variance <= 0:
            return None
        return 1.0 / mean_variance

    def run(
        self,
        *,
        cell_coadd,
        tract_info,
        noise_handles,
        visit_summary_dict=None,
    ):
        """Coadd the noise warps per cell and append them as noise planes.

        Parameters
        ----------
        cell_coadd : `~lsst.images.cells.CellCoadd`
            The input cell-based coadd to which noise planes are added.
        tract_info : `~lsst.skymap.TractInfo`
            Tract geometry, needed to reconstruct the output `CellCoadd`.
        noise_handles : `list` [`dict`]
            One mapping of visit -> noise-warp handle per noise realization.
        visit_summary_dict : `dict`, optional
            Mapping from visit to a deferred visit-summary handle, used only as
            a fallback for weights.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
            A struct with the ``outputCoadd`` attribute.
        """
        from lsst.images.cells import CellCoadd

        if visit_summary_dict is None:
            visit_summary_dict = {}

        num = self.config.num_noise_realizations

        # Make sure the mask planes we touch are registered.
        for mask_plane in self.config.bad_mask_planes:
            afwImage.Mask.addMaskPlane(mask_plane)
        for mask_plane in self.config.mask_propagation_thresholds:
            afwImage.Mask.addMaskPlane(mask_plane)

        statsCtrl = self._construct_stats_control()
        maskMap = setRejectedMaskMapping(statsCtrl)
        thresholdDict = AccumulatorMeanStack.stats_ctrl_to_threshold_dict(statsCtrl)
        no_data_bit = afwImage.Mask.getPlaneBitMask("NO_DATA")

        # Convert to the legacy representation so we can reuse the afw-based
        # per-cell stackers and the stitching machinery. ``visit_polygons``
        # maps each (visit, detector) to its footprint polygon in coadd
        # coordinates.
        legacy = cell_coadd.to_legacy_cell_coadd()
        visit_polygons = legacy.common.visit_polygons

        new_cells: list[SingleCellCoadd] = []
        for legacy_cell in legacy.cells.values():
            outer_bbox = legacy_cell.outer.bbox
            stackers = [self._make_stacker(outer_bbox, statsCtrl, maskMap, thresholdDict) for _ in range(num)]

            for obs_id, coadd_inputs in legacy_cell.inputs.items():
                weight = coadd_inputs.weight
                if not np.isfinite(weight) or weight <= 0:
                    fallback = self._fallback_weight(obs_id, visit_summary_dict)
                    if fallback is None or not np.isfinite(fallback) or fallback <= 0:
                        self.log.warning(
                            "No valid weight for visit %d detector %d in cell %s; skipping.",
                            obs_id.visit,
                            obs_id.detector,
                            legacy_cell.identifiers.cell,
                        )
                        continue
                    weight = fallback

                # Restrict this contribution to the detector footprint recorded
                # in the provenance. A single visit warp spans the full focal
                # plane, so this isolates the pixels of the specific detector
                # that carried this weight.
                outside = None
                if (polygon := visit_polygons.get(obs_id)) is not None:
                    poly_array = polygon.createImage(outer_bbox).array
                    outside = poly_array <= 0

                for n in range(num):
                    handle = noise_handles[n].get(obs_id.visit)
                    if handle is None:
                        self.log.warning(
                            "Missing noise warp %d for visit %d; skipping that contribution to cell %s.",
                            n,
                            obs_id.visit,
                            legacy_cell.identifiers.cell,
                        )
                        continue
                    noise_mi = handle.get(parameters={"bbox": outer_bbox})
                    mi = afwImage.MaskedImageF(noise_mi, deep=True)
                    if outside is not None:
                        mi.image.array[outside] = 0.0
                        mi.variance.array[outside] = np.inf
                        mi.mask.array[outside] |= no_data_bit
                    stackers[n].add_masked_image(mi, weight=weight)

            new_noise_images = []
            for n in range(num):
                cell_noise = afwImage.MaskedImageF(outer_bbox)
                stackers[n].fill_stacked_masked_image(cell_noise)
                if self.config.do_interpolate_coadd and not np.isnan(cell_noise.image.array).all():
                    self.interpolate_coadd.run(cell_noise, planeName="NO_DATA")
                new_noise_images.append(cell_noise.image)

            outer = legacy_cell.outer
            existing_noise = list(outer.noise_realizations) if self.config.append_to_existing else []
            new_planes = OwnedImagePlanes.from_masked_image(
                outer.asMaskedImage(),
                mask_fractions=outer.mask_fractions,
                noise_realizations=existing_noise + new_noise_images,
            )
            new_cells.append(
                SingleCellCoadd(
                    outer=new_planes,
                    psf=legacy_cell.psf_image,
                    inner_bbox=legacy_cell.inner.bbox,
                    inputs=legacy_cell.inputs,
                    common=legacy_cell.common,
                    identifiers=legacy_cell.identifiers,
                    aperture_correction_map=legacy_cell.aperture_correction_map,
                )
            )

        new_legacy = MultipleCellCoadd(
            new_cells,
            grid=legacy.grid,
            outer_cell_size=legacy.outer_cell_size,
            psf_image_size=legacy.psf_image_size,
            common=legacy.common,
            inner_bbox=legacy.inner_bbox,
        )

        output_coadd = CellCoadd.from_legacy_cell_coadd(
            new_legacy,
            tract_info=tract_info,
            bbox=cell_coadd.bbox,
        )

        return Struct(outputCoadd=output_coadd)
