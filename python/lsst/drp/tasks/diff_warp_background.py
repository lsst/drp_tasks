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

__all__ = ()

import dataclasses
import itertools
import math
from collections.abc import Mapping
from typing import ClassVar

import astropy.table
import numpy as np

import lsst.afw.math
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, Camera, Detector
from lsst.afw.geom import Polygon, SinglePolygonException, TransformPoint2ToPoint2, makeWcsPairTransform
from lsst.afw.image import ExposureF, Mask, PhotoCalib
from lsst.afw.table import ExposureCatalog, ExposureRecord
from lsst.daf.butler import DeferredDatasetHandle
from lsst.geom import Box2D
from lsst.pex.config import ChoiceField, Field, ListField
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT
from lsst.skymap import BaseSkyMap, PatchInfo


@dataclasses.dataclass
class WarpDetectorInfo:
    polygon: Polygon
    detector: Detector
    patch_to_detector: TransformPoint2ToPoint2
    photo_calib: PhotoCalib


@dataclasses.dataclass
class WarpData:
    visit_id: int
    detectors: Mapping[int, WarpDetectorInfo]
    exposure: ExposureF

    @classmethod
    def load_and_crop(
        cls,
        warp_handle: DeferredDatasetHandle,
        correction_warp_handle: DeferredDatasetHandle,
        patch_info: PatchInfo,
        detectors: Mapping[int, WarpDetectorInfo],
    ) -> WarpData:
        bbox = patch_info.getInnerBBox()
        exposure = warp_handle.get()[bbox]
        exposure.maskedImage /= correction_warp_handle.get()[bbox]
        return WarpData(
            visit_id=warp_handle.dataId["visit"],
            exposure=exposure,
            detectors=detectors,
        )


@dataclasses.dataclass
class DiffDetectorArrays:
    detector_id: np.ndarray
    x_camera: np.ndarray
    y_camera: np.ndarray
    x_detector: np.ndarray
    y_detector: np.ndarray
    scaling: np.ndarray
    mask: np.ndarray


class DiffWarpBackgroundsConnections(PipelineTaskConnections, dimensions=["patch", "band"]):
    warps = cT.Input(
        "direct_matched_warp",
        storageClass="ExposureF",
        multiple=True,
        deferLoad=True,
        dimensions=["patch", "visit"],
    )
    correction_warps = cT.Input(
        # TODO: nothing makes this dataset right now; need to modify
        # MakeDirectWarpTask to optionally produce it (by warping ones if it
        # doesn't get a background_to_photometric_ratio input).
        "background_to_photometric_ratio_warped",
        storageClass="ExposureF",
        multiple=True,
        deferLoad=True,
        dimensions=["patch", "visit"],
    )
    visit_summaries = cT.Input(
        "visit_summary",
        storageClass="ExposureCatalog",
        multiple=True,
        dimensions=["visit"],
    )
    sky_map = cT.Input(
        BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        storageClass="SkyMap",
        dimensions=["skymap"],
    )
    camera = cT.PrerequisiteInput(
        "camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    diff_table = cT.Output("warp_diff_binned", storageClass="ArrowAstropy", dimensions=["patch", "band"])

    def adjustQuantum(self, inputs, outputs, label, data_id):
        # Trim out inputs for which we don't have some other input for a visit.
        warps_connection, warp_refs = inputs["warps"]
        correction_warps_connection, correction_warp_refs = inputs["correction_warps"]
        visit_summaries_connection, visit_summary_refs = inputs["visit_summaries"]
        warp_refs_by_visit = {ref.dataId["visit"]: ref for ref in warp_refs}
        correction_warp_refs_by_visit = {ref.dataId["visit"]: ref for ref in correction_warp_refs}
        visit_summary_refs_by_visit = {ref.dataId["visit"]: ref for ref in visit_summary_refs}
        visits = sorted(
            warp_refs_by_visit.keys()
            & visit_summary_refs_by_visit.keys()
            & correction_warp_refs_by_visit.keys()
        )
        inputs["warps"] = (warps_connection, [warp_refs_by_visit[v] for v in visits])
        inputs["correction_warps"] = (
            correction_warps_connection,
            [correction_warp_refs_by_visit[v] for v in visits],
        )
        inputs["visit_summaries"] = (
            visit_summaries_connection,
            [visit_summary_refs_by_visit[v] for v in visits],
        )
        return super().adjustQuantum(inputs, outputs, label, data_id)


class DiffWarpBackgroundsConfig(PipelineTaskConfig, pipelineConnections=DiffWarpBackgroundsConnections):
    bin_size = Field[int]("Size of each bin (isotropic in x and y) in patch pixels.", dtype=int, default=128)
    bin_overlap_erosion = Field[int](
        "How much to shrink a bin's bounding box by (on each side) when testing whether "
        "it overlaps a detector.  Must be at least 1 to avoid floating-point round-off problems.",
        dtype=int,
        default=16,
        check=lambda v: v > 0,
    )
    n_reference_visits = Field("Maximum of visits that are diffed against all others.", dtype=int, default=4)
    bad_mask_planes = ListField(
        doc="Names of mask planes to ignore while estimating the background",
        dtype=str,
        default=["EDGE", "DETECTED", "DETECTED_NEGATIVE", "SAT", "BAD", "INTRP", "CR"],
        itemCheck=lambda x: x in Mask().getMaskPlaneDict().keys(),
    )
    statistic = ChoiceField(
        dtype=str,
        doc="Type of statistic to estimate pixel value for each bin",
        default="MEDIAN",
        allowed={"MEAN": "mean", "MEDIAN": "median", "MEANCLIP": "clipped mean"},
    )


class DiffWarpBackgroundsTask(PipelineTask):
    _DefaultName: ClassVar[str] = "diffWarpBackgrounds"
    ConfigClass: ClassVar[type[DiffWarpBackgroundsConfig]] = DiffWarpBackgroundsConfig
    config: DiffWarpBackgroundsConfig

    @staticmethod
    def make_diff_table_dtype() -> np.dtype:
        return np.dtype(
            [
                ("positive_visit_id", np.uint64),
                ("negative_visit_id", np.uint64),
                ("positive_detector_id", np.uint8),
                ("negative_detector_id", np.uint8),
                ("positive_camera_x", np.float64),
                ("positive_camera_y", np.float64),
                ("negative_camera_x", np.float64),
                ("negative_camera_y", np.float64),
                ("positive_detector_x", np.float64),
                ("positive_detector_y", np.float64),
                ("negative_detector_x", np.float64),
                ("negative_detector_y", np.float64),
                ("positive_scaling", np.float64),
                ("negative_scaling", np.float64),
                ("tract_x", np.float64),
                ("tract_y", np.float64),
                ("bin_row", np.uint16),
                ("bin_column", np.uint16),
                ("bin_value", np.float32),
                ("bin_variance", np.float32),
            ]
        )

    def __init__(self, *, config=None, log=None, initInputs=None, **kwargs):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        self.diff_table_dtype = self.make_diff_table_dtype()
        self.stats_flag = getattr(lsst.afw.math, self.config.statistic)
        self.stats_ctrl = lsst.afw.math.StatisticsControl()
        self.stats_ctrl.setAndMask(Mask.getPlaneBitMask(self.config.bad_mask_planes))
        self.stats_ctrl.setNanSafe(True)

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        sky_map = butlerQC.get(inputRefs.sky_map)
        patch_info = sky_map[butlerQC.dataId["tract"]][butlerQC.dataId["patch"]]
        warp_handles = {handle.dataId["visit"]: handle for handle in butlerQC.get(inputRefs.warps)}
        correction_warp_handles = {
            handle.dataId["visit"]: handle for handle in butlerQC.get(inputRefs.correction_warps)
        }
        visit_summaries = {ref.dataId["visit"]: butlerQC.get(ref) for ref in inputRefs.visit_summaries}
        camera = butlerQC.get(inputRefs.camera)
        outputs = self.run(
            patch_info=patch_info,
            warp_handles=warp_handles,
            correction_warp_handles=correction_warp_handles,
            visit_summaries=visit_summaries,
            camera=camera,
        )
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        *,
        patch_info: PatchInfo,
        warp_handles: Mapping[int, DeferredDatasetHandle],
        correction_warp_handles: Mapping[int, DeferredDatasetHandle],
        visit_summaries: Mapping[int, ExposureCatalog],
        camera: Camera,
    ) -> Struct:
        self.log.info("Computing detector polygons for %s visits.", len(visit_summaries))
        # A dictionary mapping visit ID to a nested mapping from detector ID to
        # polygon, with only detectors that overlap the patch and only visits
        # that have at least one overlapping detector.  We'll pop visits from
        # this dictionary as we process them.
        to_do: dict[int, dict[int, Polygon]] = {}
        for visit_id, visit_summary in visit_summaries.items():
            visit_detector_info = {
                record.getId(): detector_info
                for record in visit_summary
                if (detector_info := self._make_detector_info(record, patch_info, camera[record.getId()]))
                is not None
            }
            if visit_detector_info:
                to_do[visit_id] = visit_detector_info
        self.log.info("Selecting up to %s reference visits.", self.config.n_reference_visits)
        reference_visits = self._select_reference_visits(to_do)
        self.log.info("Loading %s reference visit warps.", len(reference_visits))
        reference_warps = [
            WarpData.load_and_crop(
                warp_handles[visit_id],
                correction_warp_handles[visit_id],
                patch_info,
                to_do.pop(visit_id),
            )
            for visit_id in reference_visits
        ]
        self.log.info(
            "Diffing %s reference visit pairs.", len(reference_visits) * (len(reference_visits) - 1) / 2
        )
        diff_table_chunks: list[np.ndarray] = []
        for positive, negative in itertools.combinations(reference_warps.items(), r=2):
            diff_table_chunks.append(self._compute_bin_diffs(positive, negative))
        self.log.info("Diffing reference visits with %s other visits.", len(to_do))
        for visit_id in sorted(to_do.keys()):  # sort just for determinism
            positive = WarpData.load_and_crop(
                warp_handles[visit_id],
                correction_warp_handles[visit_id],
                patch_info,
                to_do.pop(visit_id),
            )
            for negative in reference_warps:
                diff_table_chunks.append(self._compute_bin_diffs(positive, negative))
        return Struct(
            diff_table=astropy.table.Table(np.vstack(diff_table_chunks)),
        )

    def _make_detector_info(
        self, record: ExposureRecord, patch_info: PatchInfo, detector: Detector
    ) -> WarpDetectorInfo | None:
        detector_to_patch = makeWcsPairTransform(record.wcs, patch_info.getWcs())
        if (polygon := record.validPolygon) is None:
            polygon = Polygon(Box2D(record.getBBox()))
        try:
            patch_polygon = polygon.transform(detector_to_patch).intersectionSingle(
                Box2D(patch_info.getInnerBBox())
            )
        except SinglePolygonException:
            return None
        photo_calib = record.getPhotoCalib()
        if photo_calib is None:
            return None
        patch_to_detector = detector_to_patch.inverted()
        return WarpDetectorInfo(
            polygon=patch_polygon,
            detector=detector,
            patch_to_detector=patch_to_detector,
            photo_calib=photo_calib,
        )

    def _select_reference_visits(self, visits: Mapping[int, Mapping[int, WarpDetectorInfo]]) -> list[int]:
        if len(visits) <= self.config.n_reference_visits:
            return list(visits.keys())
        coverage = {
            visit_id: {
                detector_id: detector_info.calculateArea()
                for detector_id, detector_info in visit_detector_info.items()
            }
            for visit_id, visit_detector_info in visits.items()
        }
        target_area = 4 * self.config.bin_size**2

        def score(visit_id: int) -> tuple[int, float]:
            return (
                # Prefer visits with more detectors that have at least the
                # target area in the patch.
                sum(area > target_area for area in coverage[visit_id].values()),
                # To break ties rank by the contribution from the least-covered
                # detector.
                min(coverage[visit_id].values()),
            )

        reference_visits: list[int] = []
        while len(reference_visits) < self.config.n_reference_visits:
            best = max(coverage, key=score)
            reference_visits.append(best)
            detectors_added = coverage.pop(best)
            # In the next iteration, don't consider detectors that area already
            # in the reference list towards coverage.
            for other_visit_coverage in coverage.values():
                for detector_id in detectors_added:
                    other_visit_coverage.pop(detector_id, None)
        return reference_visits

    def _compute_bin_diffs(self, positive: WarpData, negative: WarpData) -> np.ndarray:
        """Compute the piece of the diff table that corresponds on one pair of
        warps.
        """
        diff = positive.exposure.maskedImage.clone()
        diff -= negative.exposure.maskedImage
        width = diff.getWidth()
        height = diff.getHeight()
        nx = math.ceil(width / self.config.binSize)
        ny = math.ceil(height / self.config.binSize)
        bctrl = lsst.afw.math.BackgroundControl(nx, ny, self.stats_ctrl, self.stats_flag)
        # BackgroundControl wants to know how we'll interterpolate, but we're
        # not going to use it to interpolate, so it doesn't matter.
        bctrl.setUndersampleStyle("REDUCE_INTERP_ORDER")
        bctrl.setInterpStyle("AKIMA_SPLINE")
        bkgd = lsst.afw.math.makeBackground(diff, bctrl)
        stats_image = bkgd.getStatsImage()
        result = np.zeros(nx * ny, dtype=self.diff_table_dtype)
        result["positive_visit_id"] = positive.visit_id
        result["negative_visit_id"] = negative.visit_id
        result["bin_row"] = np.arange(0, ny, dtype=result["bin_row"].dtype)[:, np.newaxis]
        result["bin_column"] = np.arange(0, nx, dtype=result["bin_row"].dtype)[np.newaxis, :]
        result["bin_value"] = stats_image.image.array.ravel()
        result["bin_variance"] = stats_image.variance.array.ravel()
        x_centers = bkgd.getBinCentersX()
        y_centers = bkgd.getBinCentersY()
        result["tract_x"], result["tract_y"] = np.meshgrid(x_centers, y_centers)
        corners = self._make_bin_corners_array(Box2D(diff.getBBox()), x_centers, y_centers)
        positive_detector = self._make_detector_arrays(
            corners, result["tract_x"], result["tract_y"], positive.detectors
        )
        result["positive_detector_id"] = positive_detector.detector_id
        result["positive_camera_x"] = positive_detector.x_camera
        result["positive_camera_y"] = positive_detector.y_camera
        result["positive_detector_x"] = positive_detector.x_detector
        result["positive_detector_y"] = positive_detector.y_detector
        result["positive_scaling"] = positive_detector.scaling
        negative_detector = self._make_detector_arrays(corners, x_centers, y_centers, negative.detectors)
        result["negative_detector_id"] = negative_detector.detector_id
        result["negative_camera_x"] = negative_detector.x_camera
        result["negative_camera_y"] = negative_detector.y_camera
        result["negative_detector_x"] = negative_detector.x_detector
        result["negative_detector_y"] = negative_detector.y_detector
        result["negative_scaling"] = negative_detector.scaling
        mask = np.all(
            [
                positive_detector.mask,
                negative_detector.mask,
                np.isfinite(result["bin_value"]),
                result["bin_variance"] > 0,
            ],
            axis=0,
        )
        return result[mask]

    def _make_bin_corners_array(
        self, bbox: Box2D, x_centers: np.ndarray, y_centers: np.ndarray
    ) -> np.ndarray:
        """Construct a 3-d array with the corners of the bins in a single
        warp.  Shape is ``(n_corners=4, y, x)``.
        """
        x_internal_edges = 0.5 * (x_centers[1:] + x_centers[:-1])
        y_internal_edges = 0.5 * (y_centers[1:] + y_centers[:-1])
        x_min_bounds = np.concatenate(
            [
                [bbox.x.min + self.config.bin_overlap_erosion],
                x_internal_edges - self.config.bin_overlap_erosion,
            ]
        )
        x_max_bounds = np.concatenate(
            [
                x_internal_edges + self.config.bin_overlap_erosion,
                [bbox.x.max - self.config.bin_overlap_erosion],
            ]
        )
        y_min_bounds = np.concatenate(
            [
                [bbox.y.min + self.config.bin_overlap_erosion],
                y_internal_edges - self.config.bin_overlap_erosion,
            ]
        )
        y_max_bounds = np.concatenate(
            [
                y_internal_edges + self.config.bin_overlap_erosion,
                [bbox.y.max - self.config.bin_overlap_erosion],
            ]
        )
        return np.array(
            [
                np.meshgrid(x_min_bounds, y_min_bounds),
                np.meshgrid(x_min_bounds, y_max_bounds),
                np.meshgrid(x_max_bounds, y_max_bounds),
                np.meshgrid(x_max_bounds, y_min_bounds),
            ]
        )

    def _make_detector_arrays(
        self,
        corners: np.ndarray,
        tract_x: np.ndarray,
        tract_y: np.ndarray,
        detectors: Mapping[int, WarpDetectorInfo],
    ) -> DiffDetectorArrays:
        """Make arrays for columns in the diff table that must be computed
        detector-by-detector.

        Parameters
        ----------
        corners : `numpy.ndarray`
            ``(4, ny, nx)`` array of bin corners.
        TODO
        detectors : `~collections.abc.Mapping` [ `int`, `WarpDetectorData` ]
            Mapping from detector ID to struct that includes the polygon and
            coordinate transfrom from patch to focal-plane coordinates.

        Returns
        -------
        detector_arrays : `DiffDetectorArrays`
            Struct of per-detector arrays.

        Notes
        -----
        This assumes that the detector polygons are accurate to less than the
        gaps between detectors.  This may not be true if chip gaps are tiny
        and optical distortion is very large (i.e. so detector boundaries
        are not straight lines in tract coordinates).
        """
        id_array = np.zeros(corners.shape[1:], dtype=np.uint8)
        x_camera = np.zeros(id_array.shape, dtype=np.float64)
        y_camera = np.zeros(id_array.shape, dtype=np.float64)
        x_detector = np.zeros(id_array.shape, dtype=np.float64)
        y_detector = np.zeros(id_array.shape, dtype=np.float64)
        scaling = np.zeros(id_array.shape, dtype=np.float64)
        mask = np.zeros(id_array.shape, dtype=bool)
        for detector_id, detector_info in detectors.items():
            detector_mask = np.all(detector_info.polygon.contains(corners), axis=0)
            id_array[detector_mask] = detector_id
            xy_detector = detector_info.patch_to_detector.getMapping().applyForward(
                np.vstack((tract_x[detector_mask], tract_y[detector_mask]))
            )
            scaling[detector_mask] = detector_info.photo_calib.getLocalCalibrationArray(
                xy_detector[0], xy_detector[1]
            )
            x_detector[detector_mask] = xy_detector[0]
            y_detector[detector_mask] = xy_detector[1]
            xy_camera = (
                detector_info.detector.getTransform(PIXELS, FOCAL_PLANE)
                .getMapping()
                .applyForward(xy_detector)
            )
            x_camera[detector_mask] = xy_camera[0]
            y_camera[detector_mask] = xy_camera[1]
            mask = np.logical_or(mask, detector_mask)
        return DiffDetectorArrays(
            detector_id=id_array,
            x_camera=x_camera,
            y_camera=y_camera,
            x_detector=x_detector,
            y_detector=y_detector,
            scaling=scaling,
            mask=mask,
        )
