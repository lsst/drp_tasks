# This file is part of drp_tasks
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
import os.path
import re
import unittest

import astropy.units as u
import numpy as np
import yaml
from scipy.linalg import lstsq
from scipy.optimize import minimize

import lsst.drp.tasks
import lsst.drp.tasks.gbdesAstrometricFit
import lsst.utils
from lsst.afw.cameraGeom.testUtils import FIELD_ANGLE, PIXELS
from lsst.afw.table import ExposureCatalog
from lsst.drp.tasks.build_camera import (
    BuildCameraFromAstrometryConfig,
    BuildCameraFromAstrometryTask,
    _z_function,
)
from lsst.obs.subaru import HyperSuprimeCam

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestBuildCameraFromAstrometry(lsst.utils.tests.TestCase):

    def setUp(self):

        rng = np.random.default_rng(seed=54321)

        # The tests will be done with data simulated to look like HSC data.
        self.camera = HyperSuprimeCam().getCamera()
        self.detectorList = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.visitList = [1176, 17900, 17930, 17934]
        self.rotationAngle = (270 * u.degree).to(u.radian).value
        self.task = BuildCameraFromAstrometryTask()

        datadir = os.path.join(TESTDIR, "data")
        with open(os.path.join(datadir, "sample_wcs.yaml"), "r") as f:
            modelParams = yaml.load(f, Loader=yaml.Loader)
        self.mapParams = {}
        for key, value in modelParams.items():
            if isinstance(key, str) and re.fullmatch(".+/poly", key):
                if value["Type"] == "Identity":
                    continue
                polyCoefficients = np.concatenate(
                    [value["XPoly"]["Coefficients"], value["YPoly"]["Coefficients"]]
                )
                self.mapParams[key] = polyCoefficients

        # Make semi-random visit maps:
        for visit in self.visitList:
            positionOffset = rng.random(2)
            angleOffset = rng.random() * (2 * np.pi)
            params = np.array(
                [
                    positionOffset[0],
                    np.cos(angleOffset),
                    -np.sin(angleOffset),
                    positionOffset[1],
                    np.sin(angleOffset),
                    np.cos(angleOffset),
                ]
            )
            self.mapParams[f"{visit}/poly"] = params

        visitSummary = ExposureCatalog.readFits(os.path.join(datadir, "visitSummary_1176.fits"))
        gbdesTask = lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask()
        _, self.mapTemplate = gbdesTask.make_yaml(visitSummary)
        self.mapTemplate["BAND/DEVICE/poly"]["XMax"] = self.camera[10].getBBox().endX
        self.mapTemplate["BAND/DEVICE/poly"]["YMax"] = self.camera[10].getBBox().endY

        recenteredModel = self.task._recenter_model(
            self.mapParams, self.mapTemplate, self.detectorList, self.camera
        )
        self.originalFieldAngleX = (recenteredModel.newTPx * u.degree).to(u.radian).value
        self.originalFieldAngleY = (recenteredModel.newTPy * u.degree).to(u.radian).value

        bbox = self.camera[self.detectorList[0]].getBBox()
        self.xPix = (self.task.x + 1) * bbox.endX / 2
        self.yPix = (self.task.y + 1) * bbox.endY / 2

    def test_recenter_model(self):
        deviceParams = []
        for element in self.mapTemplate["BAND/DEVICE"]["Elements"]:
            for detector in self.detectorList:
                detectorTemplate = element.replace("DEVICE", str(detector))
                detectorTemplate = detectorTemplate.replace("BAND", ".+")
                for k, params in self.mapParams.items():
                    if re.fullmatch(detectorTemplate, k):
                        deviceParams.append(params)
        deviceArray = np.vstack(deviceParams)

        visitParams = []
        for element in self.mapTemplate["EXPOSURE"]["Elements"]:
            for visit in self.visitList:
                visitTemplate = element.replace("EXPOSURE", str(visit))
                for k, params in self.mapParams.items():
                    if re.fullmatch(visitTemplate, k):
                        visitParams.append(params)
        identityVisitParams = np.array([0, 1, 0, 0, 0, 1])
        visitParams.append(identityVisitParams)
        expoArray = np.vstack(visitParams)

        # Get the tangent plane positions from the original device maps:
        origTpX = []
        origTpY = []
        for deviceMap in deviceArray:
            nCoeffsDev = len(deviceMap) // 2
            deviceDegree = lsst.drp.tasks.gbdesAstrometricFit._degreeFromNCoeffs(nCoeffsDev)
            intX = _z_function(deviceMap[:nCoeffsDev], self.task.x, self.task.y, order=deviceDegree)
            intY = _z_function(deviceMap[nCoeffsDev:], self.task.x, self.task.y, order=deviceDegree)
            origTpX.append(intX)
            origTpY.append(intY)

        origTpX = np.array(origTpX).ravel()
        origTpY = np.array(origTpY).ravel()

        for method in ["affine", "rigid"]:
            # Get the interim positions with the new device maps:
            recenteredModel = self.task._recenter_model(
                self.mapParams,
                self.mapTemplate,
                self.detectorList,
                self.camera,
                transform=method,
            )
            origTpX, origTpY = recenteredModel.inputCoords
            newIntX = recenteredModel.newTPx
            newIntY = recenteredModel.newTPy

            # Now fit the per-visit parameters with the constraint that the new
            # tangent plane positions match the old tangent plane positions,
            # and then check they are sufficiently close to the old values.
            newExpoArrayEmp = np.zeros(expoArray.shape)
            for e, expo in enumerate(expoArray):
                origMapX = expo[0] + origTpX * expo[1] + origTpY * expo[2]
                origMapY = expo[3] + origTpX * expo[4] + origTpY * expo[5]

                def min_function(params):
                    tpX = params[0] + params[1] * newIntX + params[2] * newIntY
                    tpY = params[3] + params[4] * newIntX + params[5] * newIntY

                    diff = ((origMapX - tpX)) ** 2 + ((origMapY - tpY)) ** 2
                    return 1000 * diff.sum()

                def jac(params):
                    tpX = params[0] + params[1] * newIntX + params[2] * newIntY
                    tpY = params[3] + params[4] * newIntX + params[5] * newIntY
                    dX = 2 * 1000 * (origMapX - tpX)
                    dY = 2 * 1000 * (origMapY - tpY)
                    jacobian = [
                        -dX.sum(),
                        -(dX * newIntX).sum(),
                        -(dX * newIntY).sum(),
                        -dY.sum(),
                        -(dY * newIntX).sum(),
                        -(dY * newIntY).sum(),
                    ]
                    return jacobian

                res = minimize(min_function, expo, method="Newton-CG", jac=jac, options={"xtol": 1e-14})
                newExpoArrayEmp[e] = res.x

                newMapX = (
                    newExpoArrayEmp[e][0] + newIntX * newExpoArrayEmp[e][1] + newIntY * newExpoArrayEmp[e][2]
                )
                newMapY = (
                    newExpoArrayEmp[e][3] + newIntX * newExpoArrayEmp[e][4] + newIntY * newExpoArrayEmp[e][5]
                )
                self.assertFloatsAlmostEqual(origMapX, newMapX, atol=1e-10)
                self.assertFloatsAlmostEqual(origMapY, newMapY, atol=1e-10)

            baselineCameraTPx, baselineCameraTPy = recenteredModel.targetCoords

            # Check that the average offset between the input camera and the
            # new camera is close to zero.
            dX = newIntX - baselineCameraTPx
            dY = newIntY - baselineCameraTPy

            self.assertAlmostEqual(dX.mean(), 0)
            self.assertAlmostEqual(dY.mean(), 0)

            # Check that the average rotation the input camera and the new
            # camera is close to zero.
            norm = (baselineCameraTPx**2 + baselineCameraTPy**2) ** 0.5 * (newIntX**2 + newIntY**2) ** 0.5
            cosTheta = (baselineCameraTPx * newIntX + baselineCameraTPy * newIntY) / norm
            self.assertAlmostEqual(cosTheta.mean(), 1, places=4)

            if method == "affine":
                # Check that there is no slope on average between input camera
                # and new camera.
                A = np.column_stack([np.ones_like(baselineCameraTPx), baselineCameraTPx, baselineCameraTPy])
                xSolution, *_ = lstsq(A, newIntX)
                ySolution, *_ = lstsq(A, newIntY)
                self.assertFloatsAlmostEqual(xSolution, np.array([0, 1, 0]), atol=1e-6)
                self.assertFloatsAlmostEqual(ySolution, np.array([0, 0, 1]), atol=1e-6)

    def test_run_with_basic_model(self):

        config = BuildCameraFromAstrometryConfig()
        config.modelSplitting = "basic"
        task = BuildCameraFromAstrometryTask(config=config)

        camera = task.run(
            self.mapParams,
            self.mapTemplate,
            self.detectorList,
            self.visitList,
            self.camera,
        )
        testX, testY = [], []
        for dev in camera:
            faX, faY = (
                dev.getTransform(PIXELS, FIELD_ANGLE)
                .getMapping()
                .applyForward(np.array([self.xPix, self.yPix]))
            )
            testX.append(faX)
            testY.append(faY)
        testX = np.concatenate(testX)
        testY = np.concatenate(testY)

        self.assertFloatsAlmostEqual(self.originalFieldAngleX, testX, atol=1e-12)
        self.assertFloatsAlmostEqual(self.originalFieldAngleY, testY, atol=1e-12)

    def test_run_with_splitModel(self):

        config = BuildCameraFromAstrometryConfig()
        config.modelSplitting = "physical"
        config.modelSplittingTolerance = 1e-6
        task = BuildCameraFromAstrometryTask(config=config)
        camera = task.run(
            self.mapParams,
            self.mapTemplate,
            self.detectorList,
            self.visitList,
            self.camera,
        )

        testX, testY = [], []
        for dev in camera:
            faX, faY = (
                dev.getTransform(PIXELS, FIELD_ANGLE)
                .getMapping()
                .applyForward(np.array([self.xPix, self.yPix]))
            )
            testX.append(faX)
            testY.append(faY)
        testX = np.concatenate(testX)
        testY = np.concatenate(testY)

        # The "physical" model splitting is not expected to
        # reconstruct the input field angles perfectly.
        self.assertFloatsAlmostEqual(self.originalFieldAngleX, testX, atol=1e-4)
        self.assertFloatsAlmostEqual(self.originalFieldAngleY, testY, atol=1e-4)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
