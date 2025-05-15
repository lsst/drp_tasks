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

from typing import ClassVar

from lsst.afw.geom import makeTransform
from lsst.afw.image import ExposureF
from lsst.afw.math import WarpingControl, warpImage
from lsst.geom import AffineTransform, Box2I, LinearTransform
from lsst.meas.algorithms import SubtractBackgroundTask
from lsst.pex.config import ConfigurableField
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


class MeasureSkyFrameBackgroundConnections(
    PipelineTaskConnections, dimensions=["detector", "physical_filter"]
):
    camera = cT.PrerequisiteInput(
        "camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    sky_frame = cT.Input(
        "sky",
        doc="Calibration sky frames.",
        storageClass="ExposureF",
        dimensions=["detector", "physical_filter"],
        isCalibration=True,
    )
    sky_frame_background = cT.Output(
        "sky_frame_background",
        doc="Binned background models fit to the sky frame.",
        storageClass="Background",
        dimensions=["detector", "physical_filter"],
        isCalibration=True,
    )


class MeasureSkyFrameBackgroundConfig(
    PipelineTaskConfig, pipelineConnections=MeasureSkyFrameBackgroundConnections
):
    background = ConfigurableField(
        target=SubtractBackgroundTask,
        doc="Task to perform background subtraction.",
    )

    def setDefaults(self):
        super().setDefaults()
        self.background.statisticsProperty = "MEAN"
        self.background.useApprox = False


class MeasureSkyFrameBackgroundTask(PipelineTask):
    """A task that measures the background on sky frames, effectively binning
    them and allowing them to be used as basis functions in
    `FitVisitBackgroundTask`.
    """

    _DefaultName: ClassVar[str] = "measureSkyFrameBackground"
    ConfigClass: ClassVar[type[MeasureSkyFrameBackgroundConfig]] = MeasureSkyFrameBackgroundConfig
    config: MeasureSkyFrameBackgroundConfig

    def __init__(self, *, config=None, log=None, initInputs=None, **kwargs):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        self.makeSubtask("background")

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        camera = butlerQC.get(inputRefs.camera)
        bbox = camera[butlerQC.quantum.dataId["detector"]].getBBox()
        sky_frame = butlerQC.get(inputRefs.sky_frame)
        results = self.run(bbox=bbox, sky_frame=sky_frame)
        butlerQC.put(results, outputRefs)

    def run(self, *, bbox: Box2I, sky_frame: ExposureF) -> Struct:
        """Subtract the background from a sky frame.

        Parameter
        ---------
        bbox : `lsst.geom.Box2I`
            Bounding box of the full detector the [binned] sky frame
            corresponds to.
        sky_frame : `lsst.afw.geom.ExposureF`
            Sky frame image.  Will be subtracted in place.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Result struct with a single ``sky_frame_backround`` attribute
            (an `lsst.afw.math.BackgroundList`).
        """
        # In order to measure the sky frame with the exact same bins used for
        # background subtraction in calibrateImage (as will be required by
        # fitVisitBackground), we scale the sky frame back up to full size.
        # That's pretty silly from an efficiency standpoint, since we're
        # ultimately going to bin it back down, but it sidesteps any
        # inconsistencies on how to deal with bin sizes that don't evenly
        # divide the image, by running the exact same background estimation
        # code on images with the exact same dimensions.
        linear = LinearTransform.makeScaling(bbox.width / sky_frame.width, bbox.height / sky_frame.height)
        transform = makeTransform(AffineTransform(linear))
        full_exposure = ExposureF(bbox)
        warpImage(full_exposure.maskedImage, sky_frame.maskedImage, transform, WarpingControl("bilinear"))
        full_exposure.maskedImage *= linear.computeDeterminant()
        background = self.background.run(full_exposure).background
        return Struct(sky_frame_background=background)
