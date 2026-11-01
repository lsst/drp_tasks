# This file is part of pipe_tasks.
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

__all__ = [
    "BackgroundMatchConnections",
    "BackgroundMatchConfig",
    "BackgroundMatchTask",
]


from lsst.pex.config import Field
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
from lsst.pipe.base.connectionTypes import Input


class BackgroundMatchConnections(
    PipelineTaskConnections,
    dimensions=("skymap", "tract", "band"),
    defaultTemplates={
        "warpType": "psf_matched",
    },
):
    warps = Input(
        doc=("Warps used to construct a list of matched backgrounds."),
        name="{warpType}_warp",
        storageClass="ExposureF",
        dimensions=("skymap", "tract", "patch", "visit"),
        multiple=True,
        deferLoad=True,
    )
    background_to_photometric_ratios = Input(
        doc="Ratios of a background-flattened image to a photometric-flattened image. ",
        name="background_to_photometric_ratio",
        storageClass="Image",
        dimensions=("visit", "detector"),
        multiple=True,
        deferLoad=True,
    )


class BackgroundMatchConfig(PipelineTaskConfig, pipelineConnections=BackgroundMatchConnections):
    binSize = Field[int](
        doc="Size of superpixel bins.",
        default=128,
    )


class BackgroundMatchTask(PipelineTask):
    """Background matching via null-space projection."""

    ConfigClass = BackgroundMatchConfig
    config: BackgroundMatchConfig
    _DefaultName = "backgroundMatch"

    def run(self, warps, background_to_photometric_ratio_list):
        """Background matching via null-space projection.

        Parameters
        ----------
        warps : `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of warped exposures (of type `~lsst.afw.image.ExposureF`).
            This is ordered by patch ID, then by visit ID
        background_to_photometric_ratios :
                `list`[`~lsst.daf.butler.DeferredDatasetHandle`]
            List of background to photometric ratio images
            (of type `~lsst.afw.image.Image`).
        """
        pass
