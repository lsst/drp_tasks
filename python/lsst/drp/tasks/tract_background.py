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
from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from lsst.afw.image import ExposureF, ImageF, FilterLabel
from lsst.afw.math import (
    BackgroundList,
    binImage,
)
from lsst.geom import Box2I, Point2I, Extent2I, AffineTransform, LinearTransform
from lsst.afw.geom import SpanSet, SkyWcs, makeModifiedWcs, makeTransform
from lsst.pex.config import Field, ConfigurableField, ListField
from lsst.pipe.base import (
    InputQuantizedConnection,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.skymap import BaseSkyMap, TractInfo
from lsst.pipe.base import connectionTypes as cT


@dataclasses.dataclass
class BinnedTractGeometry:
    tract_info: TractInfo
    original_bbox: Box2I
    superpixel_size: Extent2I
    binned_bbox: Box2I
    binned_wcs: SkyWcs

    def slices(self, patch_id: int) -> tuple[slice, slice]:
        patch_info = self.tract_info[patch_id]
        patch_bbox = patch_info.getInnerBBox()
        return (
            slice(
                (patch_bbox.y.begin - self.original_bbox.y.begin) // self.superpixel_size.y,
                (patch_bbox.y.end - self.original_bbox.y.begin) // self.superpixel_size.y,
            ),
            slice(
                (patch_bbox.x.begin - self.original_bbox.x.begin) // self.superpixel_size.x,
                (patch_bbox.x.end - self.original_bbox.x.begin) // self.superpixel_size.x,
            ),
        )


class MeasureTractBackgroundConnections(PipelineTaskConnections, dimensions=["tract", "band"]):
    sky_map = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    hf_backgrounds = cT.Input(
        "deep_coadd_hf_background",
        doc="Original high-frequency background.  Bins must exactly divide the inner patch region.",
        storageClass="Background",
        multiple=True,
        dimensions=["patch", "band"],
    )
    lf_backgrounds = cT.Output(
        "deep_coadd_lf_background",
        doc="Remeasured low-frequency background, covering inner patch regions only.",
        storageClass="Background",
        multiple=True,
        dimensions=["patch", "band"],
    )
    sky_map = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    hf_diagnostic_image = cT.Output(
        "deep_coadd_hf_background_image",
        doc="Tract-level image of the original high-frequency background superpixels.",
        storageClass="ExposureF",
        dimensions=["tract", "band"],
    )
    lf_diagnostic_image = cT.Output(
        "deep_coadd_lf_background_image",
        doc="Tract-level image of the remeasured low-frequency background superpixels.",
        storageClass="ExposureF",
        dimensions=["tract", "band"],
    )

    def __init__(self, *, config=None):
        assert isinstance(config, MeasureTractBackgroundConfig)
        if not config.write_diagnostic_images:
            del self.hf_diagnostic_image
            del self.lf_diagnostic_image

    # TODO: guarantee input/output data ID consistency via adjustQuantum.


class MeasureTractBackgroundConfig(PipelineTaskConfig, pipelineConnections=MeasureTractBackgroundConnections):
    background = ConfigurableField(
        "Configuration for the task used to subtract the background on the tract-level image of superpixels.",
        target=SubtractBackgroundTask,
    )
    mask_grow_input_planes = ListField(
        "Mask planes to union, grow by 'mask_grow_radius', and set as 'mask_grow_output_plane'. "
        "Superpixel values that are NaN are masked as NO_DATA before this operation.",
        dtype=str,
        default=["NO_DATA"],
    )
    mask_grow_radius = Field(
        "Radius to grow the background subtraction mask by (in tract-level superpixels, "
        "i.e. bins used in the high-frequency backgrounds).",
        dtype=int,
        default=8,
    )
    mask_grow_output_plane = Field(
        "Name of the mask plane that will hold the grown union of 'mask_grow_input_planes'.",
        dtype=str,
        default="BG_IGNORE",
    )
    write_diagnostic_images = Field(
        "Whether to write diagnostic image outputs.",
        dtype=bool,
        default=True,
    )

    def setDefaults(self):
        super().setDefaults()
        self.background.binSize = 8
        self.background.useApprox = False
        self.background.ignoredPixelMask.append("BG_IGNORE")

    def validate(self):
        super().validate()
        if self.mask_grow_output_plane not in self.background.ignoredPixelMask:
            raise ValueError(
                f"mask_grow_output_plane={self.mask_grow_output_plane} "
                "is not used in 'background.ignoredMaskPixels'."
            )


class MeasureTractBackgroundTask(PipelineTask):
    _DefaultName: ClassVar[str] = "measureTractBackground"
    ConfigClass: ClassVar[type[MeasureTractBackgroundConfig]] = MeasureTractBackgroundConfig
    config: MeasureTractBackgroundConfig

    def __init__(self, *, config=None, log=None, initInputs=None, **kwargs):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        self.makeSubtask("background")

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ):
        sky_map = butlerQC.get(inputRefs.sky_map)
        tract_info = sky_map[butlerQC.quantum.dataId["tract"]]
        hf_backgrounds = {ref.dataId["patch"]: butlerQC.get(ref) for ref in inputRefs.hf_backgrounds}
        results = self.run(hf_backgrounds=hf_backgrounds, tract_info=tract_info)
        for ref in outputRefs.lf_backgrounds:
            butlerQC.put(results.lf_backgrounds[ref.dataId["patch"]], ref)
        if self.config.write_diagnostic_images:
            results.hf_diagnostic_image.setFilter(FilterLabel.fromBand(butlerQC.quantum.dataId["band"]))
            results.lf_diagnostic_image.setFilter(FilterLabel.fromBand(butlerQC.quantum.dataId["band"]))
            butlerQC.put(results.hf_diagnostic_image, outputRefs.hf_diagnostic_image)
            butlerQC.put(results.lf_diagnostic_image, outputRefs.lf_diagnostic_image)

    def run(self, *, hf_backgrounds: Mapping[int, BackgroundList], tract_info: TractInfo) -> Struct:
        geometry = self.compute_geometry(hf_backgrounds=hf_backgrounds, tract_info=tract_info)
        hf_bg_exposure = self.make_hf_background_exposure(geometry=geometry, hf_backgrounds=hf_backgrounds)
        hf_bg_exposure.mask.addMaskPlane(self.config.mask_grow_output_plane)
        grow_spans = (
            SpanSet.fromMask(
                hf_bg_exposure.mask, hf_bg_exposure.mask.getPlaneBitMask(self.config.mask_grow_input_planes)
            )
            .dilated(self.config.mask_grow_radius)
            .clippedTo(geometry.binned_bbox)
        )
        grow_spans.setMask(
            hf_bg_exposure.mask, hf_bg_exposure.mask.getPlaneBitMask(self.config.mask_grow_output_plane)
        )
        lf_bg_list = self.background.run(hf_bg_exposure).background
        lf_bg_exposure = hf_bg_exposure.clone()
        lf_bg_exposure.image = lf_bg_list.getImage()
        lf_backgrounds = self.make_lf_background_lists(
            lf_bg_image=lf_bg_exposure.image, geometry=geometry, hf_backgrounds=hf_backgrounds
        )
        return Struct(
            lf_backgrounds=lf_backgrounds,
            hf_diagnostic_image=hf_bg_exposure,
            lf_diagnostic_image=lf_bg_exposure,
            geometry=geometry,
        )

    def compute_geometry(
        self, *, hf_backgrounds: Mapping[int, BackgroundList], tract_info: TractInfo
    ) -> BinnedTractGeometry:
        original_bbox = Box2I()
        superpixel_size: Extent2I | None = None
        for patch_id, hf_bg_list in hf_backgrounds.items():
            patch_info = tract_info[patch_id]
            patch_bbox = patch_info.getInnerBBox()
            original_bbox.include(patch_bbox)
            for hf_bg, *_ in hf_bg_list:
                assert patch_bbox == hf_bg.getImageBBox()
                if superpixel_size is None:
                    stats_image_bbox = hf_bg.getStatsImage().getBBox()
                    superpixel_size = Extent2I(
                        patch_bbox.width // stats_image_bbox.width,
                        patch_bbox.height // stats_image_bbox.height,
                    )
                    assert superpixel_size.x * stats_image_bbox.width == patch_bbox.width
                    assert superpixel_size.y * stats_image_bbox.height == patch_bbox.height
        binned_bbox = Box2I(
            Point2I(original_bbox.x.begin // superpixel_size.x, original_bbox.y.begin // superpixel_size.y),
            Extent2I(original_bbox.width // superpixel_size.x, original_bbox.height // superpixel_size.y),
        )
        return BinnedTractGeometry(
            tract_info=tract_info,
            original_bbox=original_bbox,
            superpixel_size=superpixel_size,
            binned_bbox=binned_bbox,
            binned_wcs=make_binned_wcs(tract_info.getWcs(), superpixel_size.x, superpixel_size.y),
        )

    def make_hf_background_exposure(
        self, *, geometry: BinnedTractGeometry, hf_backgrounds: Mapping[int, BackgroundList]
    ) -> ExposureF:
        result = ExposureF(geometry.binned_bbox)
        result.setWcs(geometry.binned_wcs)
        no_data_bitmask = result.mask.getPlaneBitMask("NO_DATA")
        result.image.array[...] = np.nan
        result.mask.array[...] = no_data_bitmask
        result.variance.array[...] = 0.0
        for patch_id, hf_bg_list in hf_backgrounds.items():
            slices = geometry.slices(patch_id)
            result.image.array[slices] = 0.0
            result.mask.array[slices] = 0
            for hf_bg, *_ in hf_bg_list:
                stats_image = hf_bg.getStatsImage()
                result.image.array[slices] += stats_image.image.array
                result.mask.array[slices] |= stats_image.mask.array
                result.mask.array[slices] |= no_data_bitmask * np.isnan(stats_image.image.array)
                result.variance.array[slices] += stats_image.variance.array
        return result

    def make_lf_background_lists(
        self,
        *,
        lf_bg_image: ImageF,
        geometry: BinnedTractGeometry,
        hf_backgrounds: Mapping[int, BackgroundList],
    ) -> dict[int, BackgroundList]:
        result = {}
        for patch_id, hf_bg_list in hf_backgrounds.items():
            lf_bg_list = BackgroundList()
            lf_bg_list.append(hf_bg_list[0])
            lf_bg, *_ = lf_bg_list[0]
            stats_image = lf_bg.getStatsImage()
            stats_image.image.array[...] = lf_bg_image.array[geometry.slices(patch_id)]
            stats_image.mask.array[...] = 0
            stats_image.variance.array[...] = 1.0
            result[patch_id] = lf_bg_list
        return result


class SubtractTractBackgroundConnections(PipelineTaskConnections, dimensions=["patch", "band"]):
    input_coadd = cT.Input(
        "deep_coadd_predetection",
        doc="Original coadd with only large-scale visit-level background subtraction.",
        storageClass="ExposureF",
        dimensions=["patch", "band"],
    )
    lf_background = cT.Input(
        "deep_coadd_lf_background",
        doc="Remeasured low-frequency background, covering inner patch regions only.",
        storageClass="Background",
        dimensions=["patch", "band"],
    )
    output_coadd = cT.Output(
        "deep_coadd_lf_subtracted",
        doc="Output coadd with tract-level backgrounds subtracted.",
        storageClass="ExposureF",
        dimensions=["patch", "band"],
    )
    input_coadd_binned = cT.Output(
        "deep_coadd_predetection_binned",
        doc="Binned version of 'input_coadd`, with bin size set to the superpixel size.",
        storageClass="ExposureF",
        dimensions=["patch", "band"],
    )
    output_coadd_binned = cT.Output(
        "deep_coadd_lf_subtracted_binned",
        doc="Binned version of 'output_coadd`, with bin size set to the superpixel size.",
        storageClass="ExposureF",
        dimensions=["patch", "band"],
    )

    def __init__(self, *, config=None):
        assert isinstance(config, SubtractTractBackgroundConfig)
        if not config.write_binned_images:
            del self.input_coadd_binned
            del self.output_coadd_binned


class SubtractTractBackgroundConfig(
    PipelineTaskConfig, pipelineConnections=SubtractTractBackgroundConnections
):
    write_binned_images = Field(
        "Whether to write out a binned versions of the input and output coadds.",
        dtype=bool,
        default=True,
    )


class SubtractTractBackgroundTask(PipelineTask):
    _DefaultName: ClassVar[str] = "subtractTractBackground"
    ConfigClass: ClassVar[type[SubtractTractBackgroundConfig]] = SubtractTractBackgroundConfig
    config: SubtractTractBackgroundConfig

    def run(self, *, input_coadd: ExposureF, lf_background: BackgroundList) -> Struct:
        for bg, *_ in lf_background:
            bg_bbox = bg.getImageBBox()
            bg_binned_bbox = bg.getStatsImage().getBBox()
            bin_size_x = bg_bbox.width // bg_binned_bbox.width
            bin_size_y = bg_bbox.height // bg_binned_bbox.height
            break
        else:
            raise AssertionError("No background in background list.")
        input_coadd = input_coadd[bg_bbox]
        result = Struct()
        if self.config.write_binned_images:
            result.input_coadd_binned = ExposureF(binImage(input_coadd.maskedImage, bin_size_x, bin_size_y))
            result.input_coadd_binned.setXY0(
                Point2I(input_coadd.getX0() // bin_size_x, input_coadd.getY0() // bin_size_y)
            )
            result.input_coadd_binned.setWcs(make_binned_wcs(input_coadd.getWcs(), bin_size_x, bin_size_y))
            result.input_coadd_binned.setFilter(input_coadd.getFilter())
        bg_image = lf_background.getImage()
        result.output_coadd = input_coadd.clone()
        result.output_coadd.image -= bg_image
        if self.config.write_binned_images:
            result.output_coadd_binned = result.input_coadd_binned.clone()
            result.output_coadd_binned.image = binImage(result.output_coadd.image, bin_size_x, bin_size_y)
        return result


def make_binned_wcs(original: SkyWcs, bin_size_x: int, bin_size_y: int) -> SkyWcs:
    binned_to_original = makeTransform(
        AffineTransform(LinearTransform.makeScaling(bin_size_x, bin_size_y))
    )
    return makeModifiedWcs(binned_to_original, original, False)
