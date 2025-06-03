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
    "BinnedTractGeometry",
    "MeasureTractBackgroundConfig",
    "MeasureTractBackgroundConnections",
    "MeasureTractBackgroundTask",
    "SubtractTractBackgroundConfig",
    "SubtractTractBackgroundConnections",
    "SubtractTractBackgroundTask",
)

import dataclasses
from collections.abc import Iterable, Mapping
from typing import ClassVar

import numpy as np

from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs, makeTransform
from lsst.afw.image import ExposureF, FilterLabel, MaskX
from lsst.afw.math import (
    BackgroundList,
    binImage,
)
from lsst.geom import AffineTransform, Box2I, Extent2I, LinearTransform, Point2I
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.pex.config import ConfigurableField, Field, ListField
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
from lsst.skymap import BaseSkyMap, TractInfo


@dataclasses.dataclass
class BinnedTractGeometry:
    """Geometry for binned tract-level images."""

    tract_info: TractInfo
    """Definition of the tract."""

    original_bbox: Box2I
    """The original bounding box of the tract level image in true pixel
    coordinates.

    Since this is constructed from a union of patch-level bounding boxes that
    are in general between the patch inner and outer boundin boxes, it is not
    the same as the tract's usual bounding box.
    """

    superpixel_size: Extent2I
    """The number of true pixels in each superpixel in each dimension."""

    binned_bbox: Box2I
    """The bounding box of the tract-level image in superpixel coordinates."""

    binned_wcs: SkyWcs
    """A WCS that maps the superpixel image to the sky."""

    def slices(self, child_bbox: Box2I) -> tuple[slice, slice]:
        """Return the array slices that select the given pixel-coordinate
        bounding box within the superpixel tract image.

        Parameters
        ----------
        child_bbox : `lsst.geom.Box2I`
            True-pixel bounding box to select.

        Returns
        -------
        y_slice : `slice`
            Superpixel image slice in the y dimension.
        x_slice : `slice`
            Superpixel image slice in the x dimension.
        """
        assert child_bbox.y.begin % self.superpixel_size.y == 0, "superpixel size evenly divides bboxes"
        assert child_bbox.x.begin % self.superpixel_size.x == 0, "superpixel size evenly divides bboxes"
        assert child_bbox.y.end % self.superpixel_size.y == 0, "superpixel size evenly divides bboxes"
        assert child_bbox.x.end % self.superpixel_size.x == 0, "superpixel size evenly divides bboxes"
        return (
            slice(
                (child_bbox.y.begin - self.original_bbox.y.begin) // self.superpixel_size.y,
                (child_bbox.y.end - self.original_bbox.y.begin) // self.superpixel_size.y,
            ),
            slice(
                (child_bbox.x.begin - self.original_bbox.x.begin) // self.superpixel_size.x,
                (child_bbox.x.end - self.original_bbox.x.begin) // self.superpixel_size.x,
            ),
        )


class MeasureTractBackgroundConnections(PipelineTaskConnections, dimensions=["tract"]):
    sky_map = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    hf_backgrounds = cT.Input(
        "deep_coadd_hf_background",
        doc=(
            "Original high-frequency background.  Bins must exactly divide the inner patch region. "
            "This input can be produced by running DetectCoaddSourcesTask with forceExactBinning "
            "and writeEmptyBackgrounds both set to True."
            ""
        ),
        storageClass="Background",
        multiple=True,
        dimensions=["patch", "band"],
    )
    lf_backgrounds = cT.Output(
        "deep_coadd_lf_background",
        doc=(
            "Remeasured low-frequency background, covering the same patches and area within those patches "
            "as the input high-frequency backgrounds."
        ),
        storageClass="Background",
        multiple=True,
        dimensions=["patch", "band"],
    )
    hf_diagnostic_images = cT.Output(
        "deep_coadd_hf_background_image",
        doc="Tract-level image of the original high-frequency background superpixels.",
        storageClass="ExposureF",
        multiple=True,
        dimensions=["tract", "band"],
    )
    lf_diagnostic_images = cT.Output(
        "deep_coadd_lf_background_image",
        doc="Tract-level image of the remeasured low-frequency background superpixels.",
        storageClass="ExposureF",
        multiple=True,
        dimensions=["tract", "band"],
    )

    def __init__(self, *, config=None):
        assert isinstance(config, MeasureTractBackgroundConfig)
        if not config.write_diagnostic_images:
            del self.hf_diagnostic_images
            del self.lf_diagnostic_images

    def adjustQuantum(self, inputs, outputs, label, data_id):
        # Make sure we have an input for every output, and vice-versa.
        hf_connection, hf_refs = inputs["hf_backgrounds"]
        lf_connection, lf_refs = outputs["lf_backgrounds"]
        hf_refs_by_data_id = {ref.dataId: ref for ref in hf_refs}
        lf_refs_by_data_id = {ref.dataId: ref for ref in lf_refs}
        common = sorted(hf_refs_by_data_id.keys() & lf_refs_by_data_id.keys())
        inputs["hf_backgrounds"] = (hf_connection, [hf_refs_by_data_id[d] for d in common])
        outputs["lf_backgrounds"] = (lf_connection, [lf_refs_by_data_id[d] for d in common])
        return super().adjustQuantum(inputs, outputs, label, data_id)


class MeasureTractBackgroundConfig(PipelineTaskConfig, pipelineConnections=MeasureTractBackgroundConnections):
    background = ConfigurableField(
        "Configuration for the task used to subtract the background on the tract-level "
        "image of superpixels. The bin sizes in this configuration are in superpixel units, "
        "and the interpolation/approximation parameter control how the superpixel background is "
        "interpolated back onto the input high-frequency background bins. ",
        target=SubtractBackgroundTask,
    )
    mask_grow_input_planes = ListField(
        "Mask planes to union across bands, grow by 'mask_grow_radius', and set as "
        "'mask_grow_output_plane'. Superpixel values that are NaN are masked as NO_DATA before this "
        "operation.",
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
    """A task that fits a (relatively) low-frequency background to tract-level
    superpixels, using high-frequency patch-level backgrounds as inputs.

    Notes
    -----
    This task starts from an initial round of high-frequency background
    estimation done in a previous task (e.g. DetectCoaddSourcesTask), combining
    the per-bin background estimates into a tract-level image in which each
    superpixel corresponds to a single bin from an input background object.
    The input background's configuration for how to interpolate or approximate
    those bin values is ignored at this stage; we use backgrounds as inputs
    rather than simple binned images because each background bin value is
    already a robust mean of non-detected pixels, which does a passable job
    at avoiding contamination from sources.

    For tracts with very large objects, it is expected that the tract-level
    image will have many missing superpixels (i.e. with NaN values).  These
    are used to form a mask, which is then unioned across all bands and grown
    by a configurable amount.  That mask is then input to a round of background
    subtraction on the superpixel image to form the "low frequency" output
    background models.  This background is propagated to patch-level output
    `lsst.afw.math.BackgroundList` objects by replacing the bin values in the
    input objects with the interpolated low-frequency model.  The original
    input interpolation/approximation configuration is then used in the
    outputs.

    This task runs over multiple bands at once only to enable that cross-band
    union of masks, which is nevertheless very important (at least in the
    context of making RGB images from the background-subtracted coadds, but
    avoiding unphysical colors is good for science productions, too).
    """

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
        hf_backgrounds = {
            (ref.dataId["band"], ref.dataId["patch"]): butlerQC.get(ref) for ref in inputRefs.hf_backgrounds
        }
        results = self.run(hf_backgrounds=hf_backgrounds, tract_info=tract_info)
        for ref in outputRefs.lf_backgrounds:
            butlerQC.put(results.lf_backgrounds[ref.dataId["band"], ref.dataId["patch"]], ref)
        if self.config.write_diagnostic_images:
            for ref in outputRefs.hf_diagnostic_images:
                butlerQC.put(results.hf_diagnostic_images[ref.dataId["band"]], ref)
            for ref in outputRefs.lf_diagnostic_images:
                butlerQC.put(results.lf_diagnostic_images[ref.dataId["band"]], ref)

    def run(
        self, *, hf_backgrounds: Mapping[tuple[str, int], BackgroundList], tract_info: TractInfo
    ) -> Struct:
        """Estimate a tract-level background model.

        Parameters
        ----------
        hf_backgrounds : `~collections.abc.Mapping`
            Mapping of input `lsst.afw.math.BackgroundList` instances, keyed
            by tuples of ``(band, patch)``.  Each background must have a single
            layer with bins that exactly divide the patch inner region (but may
            have bins of the same size outside the patch inner region).
        tract_info : `lsst.skymap.TractInfo`
            Geometry information for the tract.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct with the following attributes:

            - ``lf_backgrounds``: mapping of output
              `lsst.afw.math.BackgroundList` instances, with the same keys as
              ``hf_backgrounds``.
            - ``hf_diagnostic_images``: mapping of tract-level superpixel
              images with the input backgrounds, keyed by band.
            - ``lf_diagnostic_images``: mapping of tract-level superpixel
              images with the output backgrounds, keyed by band.
            - ``geometry``: `BinnedTractGeometry`: struct with the geometry
              of the bins.
        """
        geometry = self.compute_geometry(hf_backgrounds=hf_backgrounds.values(), tract_info=tract_info)
        hf_bg_exposures = self.make_hf_background_exposures(geometry=geometry, hf_backgrounds=hf_backgrounds)
        common_mask = MaskX(geometry.binned_bbox)
        for hf_bg_exposure in hf_bg_exposures.values():
            common_mask |= hf_bg_exposure.mask
        grow_spans = (
            SpanSet.fromMask(common_mask, common_mask.getPlaneBitMask(self.config.mask_grow_input_planes))
            .dilated(self.config.mask_grow_radius)
            .clippedTo(geometry.binned_bbox)
        )
        grow_spans.setMask(common_mask, common_mask.getPlaneBitMask(self.config.mask_grow_output_plane))
        lf_bg_exposures = {}
        for band, hf_bg_exposure in hf_bg_exposures.items():
            hf_bg_exposure.mask = common_mask
            lf_bg_list = self.background.run(hf_bg_exposure).background
            lf_bg_exposure = hf_bg_exposure.clone()
            lf_bg_exposure.image = lf_bg_list.getImage()
            lf_bg_exposures[band] = lf_bg_exposure
        lf_backgrounds = self.make_lf_background_lists(
            lf_bg_exposures=lf_bg_exposures, geometry=geometry, hf_backgrounds=hf_backgrounds
        )
        return Struct(
            lf_backgrounds=lf_backgrounds,
            hf_diagnostic_images=hf_bg_exposures,
            lf_diagnostic_images=lf_bg_exposures,
            geometry=geometry,
        )

    def compute_geometry(
        self, *, hf_backgrounds: Iterable[BackgroundList], tract_info: TractInfo
    ) -> BinnedTractGeometry:
        """Work out the geometry of a binned tract-level superpixel image.

        Parameters
        ----------
        hf_backgrounds : `~collections.abc.Iterable`
            Iterable of all input `lsst.afw.math.BackgroundList` instances that
            cover the tract.
        tract_info : `lsst.skymap.TractInfo`
            Geometry information for the tract.

        Returns
        -------
        geometry: `BinnedTractGeometry`
            A struct describing the bounding boxes and binning of the
            tract-level superpixel image.
        """
        original_bbox = Box2I()
        superpixel_size: Extent2I | None = None
        for hf_bg_list in hf_backgrounds:
            patch_bbox: Box2I | None = None
            for hf_bg, *_ in hf_bg_list:
                if patch_bbox is None:
                    patch_bbox = hf_bg.getImageBBox()
                else:
                    assert patch_bbox == hf_bg.getImageBBox()
                stats_image_bbox = hf_bg.getStatsImage().getBBox()
                if superpixel_size is None:
                    superpixel_size = Extent2I(
                        patch_bbox.width // stats_image_bbox.width,
                        patch_bbox.height // stats_image_bbox.height,
                    )
                assert superpixel_size.x * stats_image_bbox.width == patch_bbox.width
                assert superpixel_size.y * stats_image_bbox.height == patch_bbox.height
            original_bbox.include(patch_bbox)
        binned_bbox = Box2I(
            Point2I(original_bbox.x.begin // superpixel_size.x, original_bbox.y.begin // superpixel_size.y),
            Extent2I(original_bbox.width // superpixel_size.x, original_bbox.height // superpixel_size.y),
        )
        return BinnedTractGeometry(
            tract_info=tract_info,
            original_bbox=original_bbox,
            superpixel_size=superpixel_size,
            binned_bbox=binned_bbox,
            binned_wcs=_make_binned_wcs(tract_info.getWcs(), superpixel_size.x, superpixel_size.y),
        )

    def make_hf_background_exposures(
        self, *, geometry: BinnedTractGeometry, hf_backgrounds: Mapping[tuple[str, int], BackgroundList]
    ) -> dict[str, ExposureF]:
        """Build tract-level superpixel images from the bin values in input
        backgrounds.

        Parameters
        ----------
        hf_backgrounds : `~collections.abc.Mapping`
            Mapping of input `lsst.afw.math.BackgroundList` instances, keyed
            by tuples of ``(band, patch)``.  Each background must have a single
            layer with bins that exactly divide the patch inner region (but may
            have bins of the same size outside the patch inner region).
        geometry: `BinnedTractGeometry`
            A struct describing the bounding boxes and binning of the
            tract-level superpixel image.

        Returns
        -------
        hf_images : `dict`
            Mapping of tract-level superpixel images
            (`lsst.afw.image.ExposureF` objects) with the input background bin
            values, keyed by band.
        """
        result = {}
        no_data_bitmask = MaskX.getPlaneBitMask("NO_DATA")
        for (band, _), hf_bg_list in hf_backgrounds.items():
            if (exposure := result.get(band)) is None:
                exposure = ExposureF(geometry.binned_bbox)
                exposure.setWcs(geometry.binned_wcs)
                exposure.setFilter(FilterLabel.fromBand(band))
                exposure.image.array[...] = np.nan
                exposure.mask.array[...] = no_data_bitmask
                exposure.variance.array[...] = 0.0
                exposure.mask.addMaskPlane(self.config.mask_grow_output_plane)
                result[band] = exposure
            if len(hf_bg_list) != 1:
                raise RuntimeError("Input BackgroundList must have exactly one layer.")
            hf_bg = hf_bg_list[0][0]
            tract_slices = geometry.slices(hf_bg.getImageBBox())
            stats_image = hf_bg.getStatsImage()
            exposure.image.array[tract_slices] = stats_image.image.array
            exposure.mask.array[tract_slices] = stats_image.mask.array
            exposure.mask.array[tract_slices] |= no_data_bitmask * np.isnan(stats_image.image.array)
            exposure.variance.array[tract_slices] = stats_image.variance.array
        return result

    def make_lf_background_lists(
        self,
        *,
        lf_bg_exposures: Mapping[str, ExposureF],
        geometry: BinnedTractGeometry,
        hf_backgrounds: Mapping[tuple[str, int], BackgroundList],
    ) -> dict[tuple[str, int], BackgroundList]:
        """Make output `lsst.afw.math.BackgroundList` objects for each patch
        and band.

        Parameters
        ----------
        lf_bg_exposures : `~collections.abc.Mapping`
            Mapping from band to binned tract-level superpixel images
            (`lsst.afw.image.ExposureF` instance) that hold the low-frequency
            background model fit to the tract-level superpixel image for that
            band.
        geometry: `BinnedTractGeometry`
            A struct describing the bounding boxes and binning of the
            tract-level superpixel image.
        hf_backgrounds : `~collections.abc.Mapping`
            Mapping of input `lsst.afw.math.BackgroundList` instances, keyed
            by tuples of ``(band, patch)``.  THESE ARE MODIFIED IN PLACE AND
            SHOULD BE CONSIDERED CONSUMED.

        Returns
        -------
        lf_backgrounds : `dict`
            A dictionary of output low-frequency `lsst.afw.math.BackgroundList`
            instances, keyed by tuples of ``(band, patch)``.  These use the
            same interpolation/approximation parameters as the input
            high-frequency background objects, but have their bin values
            replaced by evaluating the low-frequency background model for the
            tract.
        """
        result = {}
        for (band, patch_id), hf_bg_list in hf_backgrounds.items():
            lf_bg_list = BackgroundList()
            lf_bg_list.append(hf_bg_list[0])
            lf_bg, *_ = lf_bg_list[0]
            stats_image = lf_bg.getStatsImage()
            slices = geometry.slices(lf_bg.getImageBBox())
            stats_image.image.array[...] = lf_bg_exposures[band].image.array[slices]
            stats_image.mask.array[...] = 0
            stats_image.variance.array[...] = 1.0
            result[band, patch_id] = lf_bg_list
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
    """A task that subtracts a previously-fit background model from a coadd.

    Notes
    -----
    This task is designed to apply the background models fit by
    `MeasureTractBackgroundTask`.  It also produces diagnostic outputs that are
    useful for developing and configuring that task.
    """

    _DefaultName: ClassVar[str] = "subtractTractBackground"
    ConfigClass: ClassVar[type[SubtractTractBackgroundConfig]] = SubtractTractBackgroundConfig
    config: SubtractTractBackgroundConfig

    def run(self, *, input_coadd: ExposureF, lf_background: BackgroundList) -> Struct:
        """Subtract a background model from a coadd and optionally write
        binned versions of the original and background-subtracted images.

        Parameters
        ----------
        input_coadd : `lsst.afw.image.ExposureF`
            Input coadd to subtract a background from.
        lf_background : `lsst.afw.math.BackgroundList`
            Background model to subtract.  May only cover a subset of the given
            coadd.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Output struct containing:

            - ``output_coadd`` (`lsst.afw.image.ExposureF`):
                The output background-subtracted coadd.  If ``lf_background``
                only covers a subset of ``input_coadd``, this will be cropped
                to just that subset.
            - ``input_coadd_binned`` (``lsst.afw.image.ExposureF``):
                A binned version of the input coadd.  Only produced if
                `~SubtractTractBackgroundConfig.write_binned_images` is `True`.
            - ``output_coadd_binned`` (``lsst.afw.image.ExposureF``):
                A binned version of the output coadd.  Only produced if
                `~SubtractTractBackgroundConfig.write_binned_images` is `True`.
        """
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
            result.input_coadd_binned.setWcs(_make_binned_wcs(input_coadd.getWcs(), bin_size_x, bin_size_y))
            result.input_coadd_binned.setFilter(input_coadd.getFilter())
        bg_image = lf_background.getImage()
        result.output_coadd = input_coadd.clone()
        result.output_coadd.image -= bg_image
        if self.config.write_binned_images:
            result.output_coadd_binned = result.input_coadd_binned.clone()
            result.output_coadd_binned.image = binImage(result.output_coadd.image, bin_size_x, bin_size_y)
        return result


def _make_binned_wcs(original: SkyWcs, bin_size_x: int, bin_size_y: int) -> SkyWcs:
    """Make a WCS appropriate for a binned version of an image.

    Parameters
    ----------
    original : `lsst.afw.geom.SkyWcs`
        Original WCS for the unbinned image.
    bin_size_x : `int`
        Size of each bin in the X dimension.
    bin_size_y : `int`
        Size of each bin in the Y dimension.

    Returns
    -------
    binned_wcs : `lsst.afw.geom.SkyWcs`
        WCS for the binned image.

    Notes
    -----
    This function assumes that the original image's XY0 has been divided by
    the bin size in each dimension (or is ``(0, 0)`` so this is irrelevant).
    """
    binned_to_original = makeTransform(AffineTransform(LinearTransform.makeScaling(bin_size_x, bin_size_y)))
    return makeModifiedWcs(binned_to_original, original, False)
