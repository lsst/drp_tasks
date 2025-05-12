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
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import ClassVar

import numpy as np
import pydantic

from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, Camera, Detector
from lsst.afw.image import MaskedImageF, Mask
import lsst.afw.math as afwMath
from  lsst.daf.butler import DatasetNotFoundError
from lsst.afw.math import (
    ApproximateControl,
    BackgroundList,
    BackgroundMI,
    ChebyshevBoundedField,
    ChebyshevBoundedFieldControl,
    Interpolate,
    StatisticsControl,
    UndersampleStyle,
)
from lsst.geom import Box2D, Box2I, Point2I
from lsst.pex.config import Field, RangeField, ListField, Config, ConfigField, ChoiceField
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
)
from lsst.pipe.base import connectionTypes as cT
from lsst.skymap import BaseSkyMap
import pandas as pd

class MakeBackgroundMatchArrays(
    PipelineTaskConnections,
    dimensions=("tract", "patch", "band", "skymap"),
    defaultTemplates={
        "inputCoaddName": "deep",
        "outputCoaddName": "deep",
        "warpType": "direct",
        "warpTypeSuffix": "",
    },
):
    inputWarps = cT.Input(
        doc=(
            "Input list of warps to be assemebled i.e. stacked."
            "WarpType (e.g. direct, psfMatched) is controlled by the "
            "warpType config parameter"
        ),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    skyMap = cT.Input(
        doc="Input definition of geometry/bbox and projection/wcs for coadded " "exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    selectedVisits = cT.Input(
        doc="Selected visits to be coadded.",
        name="{outputCoaddName}Visits",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "tract", "patch", "skymap", "band"),
    )
    visit_summary = cT.Input(
        doc="Input visit-summary catalog with updated calibration objects.",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
    )
    output = cT.Output(
        doc="Output image of number of input images per pixel",
        name="{outputCoaddName}Coadd_backgroundMatchArrays",
        storageClass="ArrowAstropy"
        dimensions=("tract", "patch", "skymap", "band"),
    )

class MakeBackgroundMatchArraysConfig(
    PipelineTaskConfig, pipelineConnections=PipelineTaskConnections):
    badMaskPlanes = ListField(
        doc = "Names of mask planes to ignore while estimating the background",
        dtype = str, default = ["EDGE", "DETECTED", "DETECTED_NEGATIVE","SAT","BAD","INTRP","CR"],
        itemCheck = lambda x: x in Mask().getMaskPlaneDict().keys(),
    )
    gridStatistic = ChoiceField(
        dtype = str,
        doc = "Type of statistic to estimate pixel value for the grid points",
        default = "MEDIAN",
        allowed = {
            "MEAN": "mean",
            "MEDIAN": "median",
            "MEANCLIP": "clipped mean"
            }
    )
    undersampleStyle = ChoiceField(
        doc = "Behaviour if there are too few points in grid for requested interpolation style. " \
        "Note: INCREASE_NXNYSAMPLE only allowed for usePolynomial=True.",
        dtype = str,
        default = "REDUCE_INTERP_ORDER",
        allowed = {
            "THROW_EXCEPTION": "throw an exception if there are too few points",
            "REDUCE_INTERP_ORDER": "use an interpolation style with a lower order.",
            "INCREASE_NXNYSAMPLE": "Increase the number of samples used to make the interpolation grid.",
            }
    )
    binSize = Field(
        doc = "Bin size for gridding the difference image and fitting a spatial model",
        dtype=int,
        default=300
    )
    interpStyle = ChoiceField(
        dtype = str,
        doc = "Algorithm to interpolate the background values; ignored if usePolynomial is True" \
              "Maps to an enum; see afw.math.Background",
        default = "AKIMA_SPLINE",
        allowed={
             "CONSTANT" : "Use a single constant value",
             "LINEAR" : "Use linear interpolation",
             "NATURAL_SPLINE" : "cubic spline with zero second derivative at endpoints",
             "AKIMA_SPLINE": "higher-level nonlinear spline that is more robust to outliers",
             "NONE": "No background estimation is to be attempted",
             }
    )
    numSigmaClip = Field(
        dtype = int,
        doc = "Sigma for outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default = 3
    )
    numIter = Field(
        dtype = int,
        doc = "Number of iterations of outlier rejection; ignored if gridStatistic != 'MEANCLIP'.",
        default = 2
    )


class MakeBackgroundMatchArrays(PipelineTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.statsFlag = getattr(afwMath, self.config.gridStatistic)
        self.sctrl = StatisticsControl()
        self.sctrl.setAndMask(Mask.getPlaneBitMask(self.config.badMaskPlanes))
        self.sctrl.setNanSafe(True)

    def run(self, butlerQC, inputRefs, outputRefs):
        outputDF = pd.DataFrame()
        for i, warpRef in enumerate(warpRefList):
            # hold at least one in memory until we come up with something better
            warp1 = warpRef.get()
            for j, warpRef2 in enumerate(warpRefList[i + 1]):
                try:
                    to_concat = self.getDiffArrays(warp1, visitId2=warpRef2, queryResults=results)
                    outputDF = pd.concat([outputDF, to_concat], axis=1)
                except DatasetNotFoundError as e:
                    continue

    def getDiffArrays(self, warp1, warpRef2, tractId, patchId, band, queryResults):
        warp2 =  warpRef2.get()
        visitId1 = warp1.getVisitInfo().getVisit()
        visitId2 = warpRef2.getDataId()["visit"]
        mi = warp2.maskedImage
        mi -= warp1.maskedImage
        diffMI = mi
        width = diffMI.getWidth()
        height = diffMI.getHeight()
        nx = width // self.config.binSize
        if width %  self.config.binSize != 0:
            nx += 1
        ny = height // self.config.binSize
        if height %  self.config.binSize != 0:
            ny += 1

        bctrl = afwMath.BackgroundControl(nx, ny, self.sctrl, self.statsFlag)
        bctrl.setUndersampleStyle("REDUCE_INTERP_ORDER")
        bctrl.setInterpStyle("AKIMA_SPLINE")

        bkgd = afwMath.makeBackground(diffMI, bctrl)
        statsImage = bkgd.getStatsImage()
        z = statsImage.image.array.ravel().astype(float)
        w = statsImage.variance.array.ravel().astype(float) ** (-0.5)
        xpix, ypix = np.meshgrid(bkgd.getBinCentersX(), bkgd.getBinCentersY())
        ra, decl = warp1.wcs.pixelToSkyArray(xpix.ravel(), ypix.ravel(), degrees=True)
        firstDataId = [r for r in queryResults if r["visit"] == visitId1][0]
        xy_fp = getFPCoord(firstDataId, ra, decl)
        df = pd.DataFrame()
        df["fp1X"] = xy_fp[0]
        df["fp1Y"] = xy_fp[1]
        firstDataId = [r for r in queryResults if r["visit"] == visitId2][0]
        xy_fp = getFPCoord(firstDataId, ra, decl)
        df["fp2X"] = xy_fp[0]
        df["fp2Y"] = xy_fp[1]

        df["tractId"] = tractId
        df["patchId"] = patchId
        df["visitId1"] = visitId1
        df["visitId2"] = visitId2
        df["ra"] = ra
        df["decl"] = decl
        df["z"] = z
        df["w"] = w
        return df

def getFPCoord(dataId, ra, decl):
    wcs = butler.get("preliminary_visit_image.wcs", dataId = result)
    detector = butler.get("preliminary_visit_image.detector", dataId = result)
    pix_to_fp = detector.getTransform(PIXELS, FOCAL_PLANE)
    x,y = wcs.skyToPixelArray(ra,decl, degrees=True)
    return  pix_to_fp.getMapping().applyForward(np.vstack((x.ravel(), y.ravel())))