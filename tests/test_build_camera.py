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
from scipy.optimize import minimize

import lsst.drp.tasks
import lsst.drp.tasks.gbdesAstrometricFit
import lsst.utils
from lsst.afw.cameraGeom.testUtils import FIELD_ANGLE, PIXELS, CameraWrapper
from lsst.afw.table import ExposureCatalog
from lsst.drp.tasks.build_camera import (
    BuildCameraFromAstrometryConfig,
    BuildCameraFromAstrometryTask,
    _z_function,
)

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestBuildCameraFromAstrometry(lsst.utils.tests.TestCase):

    def setUp(self):

        # The tests will be done with data simulated to look like HSC data.
        self.camera = CameraWrapper(isLsstLike=False).camera
        self.detectorList = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.visitList = np.arange(10)
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

        visitSummary = ExposureCatalog.readFits(os.path.join(datadir, "visitSummary_1176.fits"))
        gbdesTask = lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask()
        _, self.mapTemplate = gbdesTask.make_yaml(visitSummary)

        _, tangentPlaneX, tangentPlaneY = self.task._normalize_model(
            self.mapParams, self.mapTemplate, self.detectorList, self.visitList
        )
        # There is a rotation and flip between the gbdes camera model and the
        # afw camera model, which is applied by-hand here.
        self.originalFieldAngleX = (-tangentPlaneY * u.degree).to(u.radian).value
        self.originalFieldAngleY = (-tangentPlaneX * u.degree).to(u.radian).value

        bbox = self.camera[self.detectorList[0]].getBBox()
        self.xPix = (self.task.x + 1) * bbox.endX / 2
        self.yPix = (self.task.y + 1) * bbox.endY / 2

    def test_normalize_model(self):

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

        # Get the interim positions with the new device maps:
        _, newIntX, newIntY = self.task._normalize_model(
            self.mapParams, self.mapTemplate, self.detectorList, self.visitList
        )

        # Now fit the per-visit parameters with the constraint that the new
        # tangent plane positions match the old tangent plane positions, and
        # then check they are sufficiently close to the old values.
        newExpoArrayEmp = np.zeros(expoArray.shape)
        for e, expo in enumerate(expoArray):
            origMapX = expo[0] + origTpX * expo[1] + origTpY * expo[2]
            origMapY = expo[3] + origTpX * expo[4] + origTpY * expo[5]

            def min_function(params):
                tpX = params[0] + params[1] * newIntX + params[2] * newIntY
                tpY = params[3] + params[4] * newIntX + params[5] * newIntY

                diff = ((origMapX - tpX)) ** 2 + ((origMapY - tpY)) ** 2
                return diff.sum()

            def jac(params):
                tpX = params[0] + params[1] * newIntX + params[2] * newIntY
                tpY = params[3] + params[4] * newIntX + params[5] * newIntY
                dX = 2 * (origMapX - tpX)
                dY = 2 * (origMapY - tpY)
                jacobian = [
                    -dX.sum(),
                    -(dX * newIntX).sum(),
                    -(dX * newIntY).sum(),
                    -dY.sum(),
                    -(dY * newIntX).sum(),
                    -(dY * newIntY).sum(),
                ]
                return jacobian

            res = minimize(min_function, expo, method="Newton-CG", jac=jac, options={"xtol": 1e-7})
            newExpoArrayEmp[e] = res.x

            newMapX = (
                newExpoArrayEmp[e][0] + newIntX * newExpoArrayEmp[e][1] + newIntY * newExpoArrayEmp[e][2]
            )
            newMapY = (
                newExpoArrayEmp[e][3] + newIntX * newExpoArrayEmp[e][4] + newIntY * newExpoArrayEmp[e][5]
            )

            self.assertFloatsAlmostEqual(origMapX, newMapX, atol=1e-12)
            self.assertFloatsAlmostEqual(origMapY, newMapY, atol=1e-12)

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
            self.rotationAngle,
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
            self.rotationAngle,
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
