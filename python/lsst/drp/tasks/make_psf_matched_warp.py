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
    "MakePsfMatchedWarpConfig",
    "MakePsfMatchedWarpConnections",
    "MakePsfMatchedWarpTask",
)

import warnings
from typing import TYPE_CHECKING

import numpy as np

import lsst.geom as geom
from lsst.afw.geom import Polygon, SinglePolygonException, makeWcsPairTransform
from lsst.coadd.utils import copyGoodPixels
from lsst.ip.diffim import ModelPsfMatchTask, PsfComputeShapeError, WarpedPsfTransformTooBigError
from lsst.meas.algorithms import GaussianPsfFactory, WarpedPsf
from lsst.pex.config import ConfigurableField
from lsst.pipe.base import (
    AnnotatedPartialOutputsError,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base.connectionTypes import Input, Output
from lsst.pipe.tasks.coaddBase import growValidPolygons, makeSkyInfo
from lsst.skymap import BaseSkyMap
from lsst.utils.timer import timeMethod

if TYPE_CHECKING:
    from lsst.afw.image import Exposure


class MakePsfMatchedWarpConnections(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    defaultTemplates={
        "coaddName": "deep",
    },
):
    """Connections for MakePsfMatchedWarpTask"""

    sky_map = Input(
        doc="Input definition of geometry/bbox and projection/wcs for warps.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )

    direct_warp = Input(
        doc="Direct warped exposure produced by resampling calexps onto the skyMap patch geometry",
        name="{coaddName}Coadd_directWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "instrument", "visit"),
    )

    psf_matched_warp = Output(
        doc=(
            "Output PSF-Matched warped exposure, produced by resampling ",
            "calexps onto the skyMap patch geometry and PSF-matching to a model PSF.",
        ),
        name="{coaddName}Coadd_psfMatchedWarp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
    )


class MakePsfMatchedWarpConfig(
    PipelineTaskConfig,
    pipelineConnections=MakePsfMatchedWarpConnections,
):
    """Config for MakePsfMatchedWarpTask."""

    modelPsf = GaussianPsfFactory.makeField(doc="Model Psf factory")
    psfMatch = ConfigurableField(
        target=ModelPsfMatchTask,
        doc="Task to warp and PSF-match calexp",
    )

    def setDefaults(self):
        super().setDefaults()
        self.psfMatch.kernel["AL"].alardSigGauss = [1.0, 2.0, 4.5]
        self.modelPsf.defaultFwhm = 7.7


class MakePsfMatchedWarpTask(PipelineTask):
    ConfigClass = MakePsfMatchedWarpConfig
    _DefaultName = "makePsfMatchedWarp"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("psfMatch")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Docstring inherited.

        # Read in all inputs.
        inputs = butlerQC.get(inputRefs)

        sky_map = inputs.pop("sky_map")

        quantumDataId = butlerQC.quantum.dataId
        sky_info = makeSkyInfo(
            sky_map,
            tractId=quantumDataId["tract"],
            patchId=quantumDataId["patch"],
        )

        results = self.run(inputs["direct_warp"], sky_info.bbox)
        butlerQC.put(results, outputRefs)

    @timeMethod
    def run(self, direct_warp: Exposure, bbox: geom.Box2I):
        """Make a PSF-matched warp from a direct warp.

        Each individual detector from the direct warp is isolated, one at a
        time, and PSF-matched to the same model PSF. The PSF-matched images are
        then added back together to form the final PSF-matched warp. The bulk
        of the work is done by the `psfMatchTask`.

        Notes
        -----
        Pixels that receive no inputs are set to NaN, for e.g, chip gaps. This
        violates LSST algorithms group policy.

        Parameters
        ----------
        direct_warp : `lsst.afw.image.Exposure`
            Direct warp to be PSF-matched.

        Returns
        -------
        struct : `lsst.pipe.base.Struct`
            Struct containing the PSF-matched warp under the attribute
            `psf_matched_warp`.
        """
        modelPsf = self.config.modelPsf.apply()

        bit_mask = direct_warp.mask.getPlaneBitMask("NO_DATA")
        # Prepare the output exposure. We clone the input image to keep the
        # metadata, but reset the image, mask and variance planes.
        exposure_psf_matched = direct_warp[bbox].clone()
        exposure_psf_matched.image.array[:, :] = np.nan
        exposure_psf_matched.mask.array[:, :] = bit_mask
        exposure_psf_matched.variance.array[:, :] = np.inf
        exposure_psf_matched.setPsf(modelPsf)

        total_good_pixels = 0  # Total number of pixels copied to output.

        for row in direct_warp.info.getCoaddInputs().ccds:
            transform = makeWcsPairTransform(row.wcs, direct_warp.wcs)
            warp_psf = WarpedPsf(row.getPsf(), transform)

            if (src_polygon := row.validPolygon) is None:
                # Calculate the polygon for this detector.
                src_polygon = Polygon(geom.Box2D(row.getBBox()))
                self.log.debug("Polygon for detector=%d is calculated as %s", row["ccd"], src_polygon)
            else:
                self.log.debug(
                    "Polygon for detector=%d is read from the input calexp as %s", row["ccd"], src_polygon
                )

            try:
                destination_polygon = src_polygon.transform(transform).intersectionSingle(
                    geom.Box2D(direct_warp.getBBox())
                )
            except SinglePolygonException:
                self.log.info(
                    "Skipping CCD %d as its polygon does not intersect the direct warp",
                    row["ccd"],
                )
                continue

            # Compute the minimum possible bounding box that overlaps the CCD.
            # First find the intersection polygon between the per-detector warp
            # and the warp bounding box.
            bbox = geom.Box2I()
            for corner in destination_polygon.getVertices():
                bbox.include(geom.Point2I(corner))
            bbox.clip(direct_warp.getBBox())  # Additional safeguard

            # Because the direct warps are larger, it is possible that after
            # clipping, `bbox` lies outside PSF-matched warp's bounding box.
            if not bbox.overlaps(exposure_psf_matched.getBBox()):
                self.log.debug(
                    "Skipping CCD %d as its bbox %s does not overlap the PSF-matched warp",
                    row["ccd"],
                    bbox,
                )
                continue

            self.log.debug("PSF-matching CCD %d with bbox %s", row["ccd"], bbox)

            ccd_mask_array = ~(destination_polygon.createImage(bbox).array <= 0)

            # Clone the subimage, set the PSF to the model and reset the planes
            # outside the detector.
            temp_warp = direct_warp[bbox].clone()
            temp_warp.setPsf(warp_psf)
            temp_warp.image.array *= ccd_mask_array
            temp_warp.mask.array |= (~ccd_mask_array) * bit_mask
            # We intend to divide by zero outside the detector to set the
            # per-pixel variance values to infinity. Suppress the warning.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero", category=RuntimeWarning)
                temp_warp.variance.array /= ccd_mask_array

            try:
                temp_psf_matched = self.psfMatch.run(temp_warp, modelPsf).psfMatchedExposure
            except (WarpedPsfTransformTooBigError, PsfComputeShapeError) as e:
                error = AnnotatedPartialOutputsError(
                    e,
                    self,
                    temp_warp,
                    log=self.log,
                )
                raise error from e
            del temp_warp

            # Set pixels outside the intersection polygon to NO_DATA.
            temp_psf_matched.mask[bbox].array |= (~ccd_mask_array) * bit_mask

            # Clip the bbox to the PSF-matched warp bounding box.
            bbox.clip(exposure_psf_matched.getBBox())

            num_good_pixels = copyGoodPixels(
                exposure_psf_matched.maskedImage[bbox],
                temp_psf_matched.maskedImage[bbox],
                bit_mask,
            )

            del temp_psf_matched

            self.log.info(
                "Copied %d pixels from CCD %d to exposure_psf_matched",
                num_good_pixels,
                row["ccd"],
            )
            total_good_pixels += num_good_pixels

        self.log.info("Total number of good pixels = %d", total_good_pixels)

        if total_good_pixels > 0:
            growValidPolygons(
                exposure_psf_matched.info.getCoaddInputs(),
                -self.config.psfMatch.kernel.active.kernelSize // 2,
            )

            return Struct(psf_matched_warp=exposure_psf_matched)
        else:
            return Struct(psf_matched_warp=None)
