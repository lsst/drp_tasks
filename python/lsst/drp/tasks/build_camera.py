# This file is part of drp_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
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
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
import re
from collections import defaultdict

import astshim as ast
import numpy as np
import scipy.optimize

import lsst.pex.config
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, Camera
from lsst.afw.geom import TransformPoint2ToPoint2
from lsst.geom import degrees
from lsst.pipe.base import Task

__all__ = [
    "BuildCameraFromAstrometryConfig",
    "BuildCameraFromAstrometryTask",
]

# Set up global variable to use in minimization callback.
nFunctionEval = 0


def _z_function(params, x, y, order=4):
    """Convenience function for calculating the value of a 2-d polynomial with
    coefficients given by an array of parameters.

    Parameters
    ----------
    params : `np.ndarray`
        Coefficientss for polynomial with terms ordered
        [1, x, y, x^2, xy, y^2, ...].
    x, y : `np.ndarray`
        x and y values at which to evaluate the 2-d polynomial.
    order : `int`, optional
        Order of the given polynomial.

    Returns
    -------
    z : `np.ndarray`
        Value of polynomial evaluated at (x, y).
    """
    coeffs = np.zeros((order + 1, order + 1))
    c = 0
    for i in range(order + 1):
        for j in range(i + 1):
            coeffs[j, i - j] = params[c]
            c += 1
    z = np.polynomial.polynomial.polyval2d(x, y, coeffs.T)
    return z


def _z_function_dx(params, x, y, order=4):
    """Convenience function for calculating the derivative with respect to x of
    a 2-d polynomial with coefficients given by an array of parameters.

    Parameters
    ----------
    params : `np.ndarray`
        Coefficientss for polynomial with terms ordered
        [1, x, y, x^2, xy, y^2, ...].
    x, y : `np.ndarray`
        x and y values at which to evaluate the 2-d polynomial.
    order : `int`, optional
        Order of the given polynomial.

    Returns
    -------
    z : `np.ndarray`
        Derivative of polynomial evaluated at (x, y).
    """
    coeffs = np.zeros((order + 1, order + 1))
    c = 0
    for i in range(order + 1):
        for j in range(i + 1):
            coeffs[j, i - j] = params[c]
            c += 1
    derivCoeffs = np.zeros(coeffs.shape)
    derivCoeffs[:, :-1] = coeffs[:, 1:]
    derivCoeffs[:, :-1] *= np.arange(1, order + 1)

    z = np.polynomial.polynomial.polyval2d(x, y, derivCoeffs.T)
    return z


def _z_function_dy(params, x, y, order=4):
    """Convenience function for calculating the derivative with respect to y of
    a 2-d polynomial with coefficients given by an array of parameters.

    Parameters
    ----------
    params : `np.ndarray`
        Coefficientss for polynomial with terms ordered
        [1, x, y, x^2, xy, y^2, ...].
    x, y : `np.ndarray`
        x and y values at which to evaluate the 2-d polynomial.
    order : `int`, optional
        Order of the given polynomial.

    Returns
    -------
    z : `np.ndarray`
        Derivative of polynomial evaluated at (x, y).
    """
    coeffs = np.zeros((order + 1, order + 1))
    c = 0
    for i in range(order + 1):
        for j in range(i + 1):
            coeffs[j, i - j] = params[c]
            c += 1
    derivCoeffs = np.zeros(coeffs.shape)
    derivCoeffs[:-1] = coeffs[1:]
    derivCoeffs[:-1] *= np.arange(1, order + 1).reshape(-1, 1)

    z = np.polynomial.polynomial.polyval2d(x, y, derivCoeffs.T)
    return z


class BuildCameraFromAstrometryConfig(lsst.pex.config.Config):
    """Configuration for BuildCameraTask."""

    tangentPlaneDegree = lsst.pex.config.Field(
        dtype=int,
        doc=(
            "Order of polynomial mapping between the focal plane and the tangent plane. Only used if "
            "modelSplitting='physical'."
        ),
        default=9,
    )
    focalPlaneDegree = lsst.pex.config.Field(
        dtype=int,
        doc=(
            "Order of polynomial mapping between the pixels and the focal plane. Only used if "
            "modelSplitting='physical'."
        ),
        default=3,
    )
    modelSplitting = lsst.pex.config.ChoiceField(
        dtype=str,
        doc="How to split the camera model into pixel to focal plane and focal plane to tangent plane parts.",
        default="basic",
        allowed={
            "basic": "Put all the mapping except scaling into the pixel to focal plane part",
            "physical": (
                "Fit a more physical mapping where as much as possible of the model variability goes"
                "into the focal plane to tangent plane part of the model, to imitate the effects of "
                "the telescope optics"
            ),
        },
    )
    plateScale = lsst.pex.config.Field(
        dtype=float,
        doc=("Scaling between camera coordinates in mm and angle on the sky in" " arcsec."),
        default=20.005867576692737,
    )
    astInversionTolerance = lsst.pex.config.Field(
        dtype=float,
        doc="Tolerance for AST map inversion.",
        default=0.005,
    )
    astInversionMaxIter = lsst.pex.config.Field(
        dtype=int, doc="Maximum iterations fro AST map inversion.", default=10
    )
    modelSplittingTolerance = lsst.pex.config.Field(
        dtype=float,
        doc="Average error in model splitting minimization acceptable for convergence.",
        default=1e-8,
    )


class BuildCameraFromAstrometryTask(Task):
    """Build an `lsst.afw.cameraGeom.Camera` object out of the `gbdes`
    polynomials mapping from pixels to the tangent plane.

    Parameters
    ----------

    """

    ConfigClass = BuildCameraFromAstrometryConfig
    _DefaultName = "buildCameraFromAstrometry"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # The gbdes model normalizes the pixel positions to the range -1 - +1.
        X = np.arange(-1, 1, 0.1)
        Y = np.arange(-1, 1, 0.1)
        x, y = np.meshgrid(X, Y)
        self.x = x.ravel()
        self.y = y.ravel()

    def run(self, mapParams, mapTemplate, detectorList, visitList, inputCamera, rotationAngle):
        """Convert the model parameters into a Camera object.

        Parameters
        ----------
        mapParams : `dict` [`float`]
            Parameters that describe the per-detector and per-visit parts of
            the astrometric model.
        mapTemplate : `dict`
            Dictionary describing the format of the astrometric model,
            following the convention in `gbdes`.
        detectorList : `list` [`int`]
            List of detector ids.
        visitList : `list` [`int`]
            List of ids for visits that were used to train the input model.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object from which to take pupil function, name, and other
            properties.
        rotationAngle : `float`
            Value in radians of the average rotation angle of the input visits.

        Returns
        -------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object with transformations set by the input mapParams.
        """

        # Normalize the model.
        newParams, newTPx, newTPy = self._prep_model(mapParams, mapTemplate, detectorList, visitList)

        if self.config.modelSplitting == "basic":
            # Put all of the camera distortion into the pixels->focal plane
            # part of the distortion model, with the focal plane->tangent plane
            # part only used for scaling between the focal plane and sky.
            pixToFocalPlane, focalPlaneToTangentPlane = self._basic_model(newParams, detectorList)

        else:
            # Fit two polynomials, such that the first describes the pixel->
            # focal plane part of the model, and the second describes the focal
            # plane->tangent plane part of the model, with the goal of
            # generating a more physically-motivated distortion model.
            pixToFocalPlane, focalPlaneToTangentPlane = self._split_model(newTPx, newTPy, detectorList)

        xScale = int(mapTemplate["BAND/DEVICE/poly"]["XMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["XMin"])
        yScale = int(mapTemplate["BAND/DEVICE/poly"]["YMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["YMin"])

        # Turn the mappings into a Camera object.
        camera = self._translate_to_afw(
            pixToFocalPlane, focalPlaneToTangentPlane, detectorList, inputCamera, rotationAngle,
            xScale, yScale,
        )

        return camera

    def _prep_model(self, mapParams, mapTemplate, detectorList, visitList, normalize=False):
        """Normalize and recenter the camera mappings, such that they
        correspond to the average visit.

        In order to break degeneracies, `gbdes` sets one visit to have the
        identity for the per-visit part of its model. This visit is thus
        imprinted in the per-detector part of the model. This function
        normalizes and recenters the model such that the parameters correspond
        instead to the average per-visit behavior. This step assumes that the
        per-visit part of the model is a first-order polynomial, in which case
        this conversion can be done without loss of information. If `normalize`
        is false, the model is just recentered so that the average offset is
        zero.

        Parameters
        ----------
        mapParams : `dict` [`float`]
            Parameters that describe the per-detector and per-visit parts of
            the astrometric model.
        mapTemplate : `dict`
            Dictionary describing the format of the astrometric model,
            following the convention in `gbdes`.
        detectorList : `list` [`int`]
            List of detector ids.
        visitList : `list` [`int`]
            List of ids for visits that were used to train the input model.

        Returns
        -------
        newDeviceArray : `np.ndarray`
            Array of NxM, where N is the number of detectors, and M is the
            number of coefficients for each per-detector mapping.
        newTPx, newTPy : `np.ndarray`
            Projection of `self.x` and `self.y` onto the tangent plane, given
            the normalized mapping.
        """

        # Get the per-device and per-visit parts of the model.
        deviceParams = []
        visitParams = []
        for element in mapTemplate["BAND/DEVICE"]["Elements"]:
            for detector in detectorList:
                detectorTemplate = element.replace("DEVICE", str(detector))
                detectorTemplate = detectorTemplate.replace("BAND", ".+")
                for k, params in mapParams.items():
                    if re.fullmatch(detectorTemplate, k):
                        deviceParams.append(params)
        deviceArray = np.vstack(deviceParams)

        for element in mapTemplate["EXPOSURE"]["Elements"]:
            for visit in visitList:
                visitTemplate = element.replace("EXPOSURE", str(visit))
                for k, params in mapParams.items():
                    if re.fullmatch(visitTemplate, k):
                        visitParams.append(params)
        identityVisitParams = np.array([0, 1, 0, 0, 0, 1])
        visitParams.append(identityVisitParams)
        expoArray = np.vstack(visitParams)

        expoMean = expoArray.mean(axis=0)
        if normalize:
            # Shift the per-device part of the model to correspond with the mean
            # per-visit behavior.
            newDeviceArray = np.zeros(deviceArray.shape)
            nCoeffsDev = deviceArray.shape[1] // 2
            newDeviceArray[:, :nCoeffsDev] = (
                deviceArray[:, :nCoeffsDev] * expoMean[1] + deviceArray[:, nCoeffsDev:] * expoMean[2]
            )
            newDeviceArray[:, nCoeffsDev:] = (
                deviceArray[:, :nCoeffsDev] * expoMean[4] + deviceArray[:, nCoeffsDev:] * expoMean[5]
            )
            newDeviceArray[:, 0] += expoMean[0]
            newDeviceArray[:, nCoeffsDev] += expoMean[3]
        else:
            # Just recenter the model.
            newDeviceArray = np.copy(deviceArray)
            newDeviceArray[:, 0] += expoMean[0]
            newDeviceArray[:, nCoeffsDev] += expoMean[3]


        # Then get the tangent plane positions from the new device model:
        newTPx = []
        newTPy = []
        for deviceMap in newDeviceArray:
            nCoeffsDev = len(deviceMap) // 2
            deviceDegree = int(-1.5 + 0.5 * (1 + 8 * nCoeffsDev) ** 0.5)

            intX = _z_function(deviceMap[:nCoeffsDev], self.x, self.y, order=deviceDegree)
            intY = _z_function(deviceMap[nCoeffsDev:], self.x, self.y, order=deviceDegree)
            newTPx.append(intX)
            newTPy.append(intY)

        newTPx = np.array(newTPx).ravel()
        newTPy = np.array(newTPy).ravel()

        return newDeviceArray, newTPx, newTPy

    def _basic_model(self, modelParameters, detectorList):
        """This will just convert the pix->fp parameters into the right format,
        and return an identity mapping for the fp->tp part.

        Parameters
        ----------
        modelParameters : `np.ndarray`
            Array of NxM, where N is the number of detectors, and M is the
            number of coefficients for each per-detector mapping.
        detectorList : `list` [`int`]
            List of detector ids.

        Returns
        -------
        pixelsToFocalPlane : `dict` [`int`, [`string`, `float`]]
            Dictionary giving the per-detector pixel to focal plane mapping,
            with keys for each detector giving the x and y transformation
            polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Dictionary giving the focal plane to tangent plane mapping.
        """

        nCoeffsFp = modelParameters.shape[1] // 2
        pixelsToFocalPlane = defaultdict(dict)
        for d, det in enumerate(detectorList):
            pixelsToFocalPlane[det]["x"] = modelParameters[d][:nCoeffsFp]
            pixelsToFocalPlane[det]["y"] = modelParameters[d][nCoeffsFp:]

        focalPlaneToTangentPlane = {"x": [0, 1, 0], "y": [0, 0, 1]}

        return pixelsToFocalPlane, focalPlaneToTangentPlane

    def _split_model(self, targetX, targetY, detectorList, pixToFpGuess=None, fpToTpGuess=None):
        """Fit a two-step model, with one polynomial per detector fitting the
        pixels to focal plane part, followed by a polynomial fitting the focal
        plane to tangent plane part.

        The linear parameters of the focal plane to tangent plane model are
        fixed to be an identity map to remove degeneracies.

        Parameters
        ----------
        targetX, targetY : `np.ndarray`
            Target x and y values in the tangent plane.
        detectorList : `list` [`int`]
            List of detector ids.
        pixToFpGuess : `dict` [`int`, [`string`, `float`]]
            Initial guess for the pixels to focal plane mapping to use in the
            model fit, in the form of a dictionary giving the per-detector
            pixel to focal plane mapping, with keys for each detector giving
            the x and y transformation polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Initial guess for the focal plane to tangent plane mapping to use
            in the model fit, in the form of a dictionary giving the focal
            plane to tangent plane mapping.

        Returns
        -------
        pixelsToFocalPlane : `dict` [`int`, [`string`, `float`]]
            Dictionary giving the per-detector pixel to focal plane mapping,
            with keys for each detector giving the x and y transformation
            polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Dictionary giving the focal plane to tangent plane mapping.
        """

        tpDegree = self.config.tangentPlaneDegree
        fpDegree = self.config.focalPlaneDegree

        nCoeffsFP = int((fpDegree + 2) * (fpDegree + 1) / 2)
        nCoeffsFPTot = len(detectorList) * nCoeffsFP * 2

        nCoeffsTP = int((tpDegree + 2) * (tpDegree + 1) / 2)
        # The constant and linear terms will be fixed to remove degeneracy with
        # the pix->fp part of the model
        nCoeffsFixed = 3
        nCoeffsFree = nCoeffsTP - nCoeffsFixed
        fixedParams = np.zeros(nCoeffsTP * 2)
        fixedParams[1] = 1
        fixedParams[nCoeffsTP + 2] = 1

        nX = len(self.x)
        # We need an array of the form [1, x, y, x^2, xy, y^2, ...] to use when
        # calculating derivatives.
        xyArray = np.zeros((nCoeffsFP, nX))
        c = 0
        for ii in range(fpDegree + 1):
            for jj in range(ii + 1):
                coeffs = np.zeros((fpDegree + 1, fpDegree + 1))
                coeffs[jj, ii - jj] = 1
                xyArray[c] = np.polynomial.polynomial.polyval2d(self.x, self.y, coeffs.T)
                c += 1

        def two_part_function(params):
            # The function giving the split model.
            fpXAll = []
            fpYAll = []
            for i in range(len(detectorList)):
                fpX = _z_function(
                    params[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                fpY = _z_function(
                    params[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)], self.x, self.y, order=fpDegree
                )
                fpXAll.append(fpX)
                fpYAll.append(fpY)
            fpXAll = np.array(fpXAll).ravel()
            fpYAll = np.array(fpYAll).ravel()

            tpParams = fixedParams.copy()
            tpParams[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFPTot : nCoeffsFPTot + nCoeffsFree]
            tpParams[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFPTot + nCoeffsFree :]

            tpX = _z_function(tpParams[:nCoeffsTP], fpXAll, fpYAll, order=tpDegree)
            tpY = _z_function(tpParams[nCoeffsTP:], fpXAll, fpYAll, order=tpDegree)
            return tpX, tpY

        def min_function(params):
            # The least-squares function to be minimized.
            tpX, tpY = two_part_function(params)
            diff = ((targetX - tpX)) ** 2 + ((targetY - tpY)) ** 2
            result = diff.sum()
            return result

        def jac(params):
            # The Jacobian of min_function, which is needed for the minimizer.
            tpX, tpY = two_part_function(params)
            dX = -2 * (targetX - tpX)
            dY = -2 * (targetY - tpY)
            jacobian = np.zeros(len(params))
            fpParams = params[:nCoeffsFPTot]
            tpParams = fixedParams.copy()
            tpParams[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFPTot : nCoeffsFPTot + nCoeffsFree]
            tpParams[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFPTot + nCoeffsFree :]
            fpXAll = []
            fpYAll = []
            # Fill in the derivatives for the pix->fp terms.
            for i in range(len(detectorList)):
                dX_i = dX[nX * i : nX * (i + 1)]
                dY_i = dY[nX * i : nX * (i + 1)]
                fpX = _z_function(
                    fpParams[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                fpY = _z_function(
                    fpParams[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
                    self.x,
                    self.y,
                    order=fpDegree,
                )
                fpXAll.append(fpX)
                fpYAll.append(fpY)

                dTpX_dFpX = _z_function_dx(tpParams[:nCoeffsTP], fpX, fpY, order=tpDegree)
                dTpX_dFpY = _z_function_dy(tpParams[:nCoeffsTP], fpX, fpY, order=tpDegree)
                dTpY_dFpX = _z_function_dx(tpParams[nCoeffsTP:], fpX, fpY, order=tpDegree)
                dTpY_dFpY = _z_function_dy(tpParams[nCoeffsTP:], fpX, fpY, order=tpDegree)

                dTpX_part = np.concatenate([xyArray * dTpX_dFpX, xyArray * dTpX_dFpY])
                dTpY_part = np.concatenate([xyArray * dTpY_dFpX, xyArray * dTpY_dFpY])
                jacobian_i = (dX_i * dTpX_part + dY_i * dTpY_part).sum(axis=1)
                jacobian[2 * nCoeffsFP * i : 2 * nCoeffsFP * (i + 1)] = jacobian_i

            fpXAll = np.array(fpXAll).ravel()
            fpYAll = np.array(fpYAll).ravel()

            # Fill in the derivatives for the fp->tp terms.
            for j in range(nCoeffsFree):
                tParams = np.zeros(nCoeffsTP)
                tParams[nCoeffsFixed + j] = 1
                tmpZ = _z_function(tParams, fpXAll, fpYAll, order=tpDegree)
                jacobian[nCoeffsFPTot + j] = (dX * tmpZ).sum()
                jacobian[nCoeffsFPTot + nCoeffsFree + j] = (dY * tmpZ).sum()
            return jacobian

        def hessian(params):
            # The Hessian of min_function, which is needed for the minimizer.
            hessian = np.zeros((len(params), len(params)))

            fpParams = params[:nCoeffsFPTot]
            tpParams = fixedParams.copy()
            tpParams[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFPTot : nCoeffsFPTot + nCoeffsFree]
            tpParams[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFPTot + nCoeffsFree :]
            fpXAll = []
            fpYAll = []

            # Loop over the detectors to fill in the d(pix->fp)**2 and
            # d(pix->fp)d(fp->tp) cross terms.
            for i in range(len(detectorList)):
                fpX = _z_function(
                    fpParams[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                fpY = _z_function(
                    fpParams[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
                    self.x,
                    self.y,
                    order=fpDegree,
                )
                fpXAll.append(fpX)
                fpYAll.append(fpY)
                dTpX_dFpX = _z_function_dx(tpParams[:nCoeffsTP], fpX, fpY, order=tpDegree)
                dTpX_dFpY = _z_function_dy(tpParams[:nCoeffsTP], fpX, fpY, order=tpDegree)
                dTpY_dFpX = _z_function_dx(tpParams[nCoeffsTP:], fpX, fpY, order=tpDegree)
                dTpY_dFpY = _z_function_dy(tpParams[nCoeffsTP:], fpX, fpY, order=tpDegree)

                dTpX_part = np.concatenate([xyArray * dTpX_dFpX, xyArray * dTpX_dFpY])
                dTpY_part = np.concatenate([xyArray * dTpY_dFpX, xyArray * dTpY_dFpY])

                for k in range(2 * nCoeffsFP):
                    for m in range(2 * nCoeffsFP):
                        hessian[2 * nCoeffsFP * i + k, 2 * nCoeffsFP * i + m] = (
                            2 * (dTpX_part[k] * dTpX_part[m] + dTpY_part[k] * dTpY_part[m]).sum()
                        )

                    for j in range(nCoeffsFree):
                        tmpParams = np.zeros(nCoeffsTP)
                        tmpParams[nCoeffsFixed + j] = 1
                        tmpZ = _z_function(tmpParams, fpX, fpY, order=tpDegree)

                        hessX = 2 * (dTpX_part[k] * tmpZ).sum()
                        hessY = 2 * (dTpY_part[k] * tmpZ).sum()
                        # dTP_x part
                        hessian[2 * nCoeffsFP * i + k, nCoeffsFPTot + j] = hessX
                        hessian[nCoeffsFPTot + j, 2 * nCoeffsFP * i + k] = hessX
                        # dTP_y part
                        hessian[2 * nCoeffsFP * i + k, nCoeffsFPTot + nCoeffsFree + j] = hessY
                        hessian[nCoeffsFPTot + nCoeffsFree + j, 2 * nCoeffsFP * i + k] = hessY

            fpXAll = np.array(fpXAll).ravel()
            fpYAll = np.array(fpYAll).ravel()
            tmpZArray = np.zeros((nCoeffsFree, nX * len(detectorList)))
            # Finally, get the d(fp->tp)**2 terms
            for j in range(nCoeffsFree):
                tParams = np.zeros(nCoeffsTP)
                tParams[nCoeffsFixed + j] = 1
                tmpZArray[j] = _z_function(tParams, fpXAll, fpYAll, order=tpDegree)
            for j in range(nCoeffsFree):
                for m in range(nCoeffsFree):
                    # X-Y cross terms are zero
                    hess = 2 * (tmpZArray[j] * tmpZArray[m]).sum()
                    hessian[nCoeffsFPTot + j, nCoeffsFPTot + m] = hess
                    hessian[nCoeffsFPTot + nCoeffsFree + j, nCoeffsFPTot + nCoeffsFree + m] = hess
            return hessian

        global nFunctionEval
        nFunctionEval = 0

        def callbackMF(params):
            global nFunctionEval
            self.log.info(f"Iteration {nFunctionEval}, function value {min_function(params)}")
            nFunctionEval += 1

        initGuess = np.zeros(nCoeffsFPTot + 2 * nCoeffsFree)
        if pixToFpGuess:
            useVar = min(nCoeffsFP, len(pixToFpGuess[0]["x"]))
            for i, det in enumerate(detectorList):
                initGuess[2 * nCoeffsFP * i : 2 * nCoeffsFP * i + useVar] = pixToFpGuess[det]["x"][:useVar]
                initGuess[(2 * i + 1) * nCoeffsFP : (2 * i + 1) * nCoeffsFP + useVar] = pixToFpGuess[det][
                    "y"
                ][:useVar]
        if fpToTpGuess:
            useVar = min(nCoeffsTP, len(fpToTpGuess["x"]))
            initGuess[nCoeffsFPTot : nCoeffsFPTot + useVar - nCoeffsFixed] = fpToTpGuess["x"][
                nCoeffsFixed:useVar
            ]
            initGuess[nCoeffsFPTot + nCoeffsFree : nCoeffsFPTot + nCoeffsFree + useVar - nCoeffsFixed] = (
                fpToTpGuess["y"][nCoeffsFixed:useVar]
            )

        self.log.info(f"Initial value of least squares function: {min_function(initGuess)}")
        res = scipy.optimize.minimize(
            min_function,
            initGuess,
            callback=callbackMF,
            method="Newton-CG",
            jac=jac,
            hess=hessian,
            options={"xtol": self.config.modelSplittingTolerance},
        )

        # Convert parameters to a dictionary.
        pixToFP = {}
        for i, det in enumerate(detectorList):
            pixToFP[det] = {
                "x": res.x[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP],
                "y": res.x[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
            }

        fpToTpAll = fixedParams.copy()
        fpToTpAll[nCoeffsFixed:nCoeffsTP] = res.x[nCoeffsFPTot : nCoeffsFPTot + nCoeffsFree]
        fpToTpAll[nCoeffsFixed + nCoeffsTP :] = res.x[nCoeffsFPTot + nCoeffsFree :]
        fpToTp = {"x": fpToTpAll[:nCoeffsTP], "y": fpToTpAll[nCoeffsTP:]}

        return pixToFP, fpToTp

    def make_ast_polymap_coeffs(self, order, xCoeffs, yCoeffs):
        """Convert polynomial coefficients in gbdes format to AST PolyMap input
        format.

        Paramaters
        ----------
        order: `int`
            Polynomial order
        xCoeffs, yCoeffs: `list` of `float`
            Forward or inverse polynomial coefficients for the x and y axes
            of output, in this order:
                x0y0, x1y0, x0y1, x2y0, x1y1, x0y2, ...xNy0, xN-1y1, ...x0yN
            where N is the polynomial order.
        Returns
        -------
        polyArray : `np.ndarray`
            Array formatted for AST PolyMap input.
        """
        N = len(xCoeffs)

        polyArray = np.zeros((2 * N, 4))

        for i in range(order + 1):
            for j in range(order + 1):
                if (i + j) > order:
                    continue
                vectorIndex = int(((i + j) * (i + j + 1)) / 2 + j)
                polyArray[vectorIndex] = [xCoeffs[vectorIndex], 1, i, j]
                polyArray[vectorIndex + N] = [yCoeffs[vectorIndex], 2, i, j]

        return polyArray

    def _translate_to_afw(
        self, pixToFocalPlane, focalPlaneToTangentPlane, detectorList, inputCamera, rotationAngle,
        xScale, yScale
    ):
        """Convert the model parameters to a Camera object.

        Parameters
        ----------
        pixelsToFocalPlane : `dict` [`int`, [`string`, `float`]]
            Dictionary giving the per-detector pixel to focal plane mapping,
            with keys for each detector giving the x and y transformation
            polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Dictionary giving the focal plane to tangent plane mapping.
        rotationAngle : `float`
            Value in radians of the average rotation angle of the input visits.

        Returns
        -------
        output_camera : `lsst.afw.cameraGeom.Camera`
            Camera object containing the pix->fp and fp->tp transforms.
        """
        if self.config.modelSplitting == "basic":
            tpDegree = 1
            nCoeffsFP = len(pixToFocalPlane[detectorList[0]]["x"])
            fpDegree = int(-1.5 + 0.5 * (1 + 8 * nCoeffsFP) ** 0.5)
        else:
            tpDegree = self.config.tangentPlaneDegree
            fpDegree = self.config.focalPlaneDegree

        scaleConvert = (1 * degrees).asArcseconds() / self.config.plateScale

        cameraBuilder = Camera.Builder(inputCamera.getName())
        cameraBuilder.setPupilFactoryName(inputCamera.getPupilFactoryName())

        # Convert fp->tp to AST format:
        forwardCoeffs = self.make_ast_polymap_coeffs(
            tpDegree, focalPlaneToTangentPlane["x"], focalPlaneToTangentPlane["y"]
        )
        # Reverse rotation from input visits and flip x-axis.
        cosRot = np.cos(rotationAngle)
        sinRot = np.sin(rotationAngle)
        rotateAndFlipCoeffs = self.make_ast_polymap_coeffs(1, [0, -cosRot, sinRot], [0, sinRot, cosRot])

        ccdZoom = ast.ZoomMap(2, 1 / scaleConvert)
        ccdToSky = ast.PolyMap(
            forwardCoeffs,
            2,
            "IterInverse=1, TolInverse=%s, NIterInverse=%s"
            % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
        )
        rotateAndFlip = ast.PolyMap(
            rotateAndFlipCoeffs,
            2,
            "IterInverse=1, TolInverse=%s, NIterInverse=%s"
            % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
        )
        toRadians = ast.ZoomMap(2, (1 * degrees).asRadians())

        fullMapping = ccdZoom.then(rotateAndFlip).then(ccdToSky).then(rotateAndFlip).then(toRadians)
        focalPlaneToTPMapping = TransformPoint2ToPoint2(fullMapping)
        cameraBuilder.setTransformFromFocalPlaneTo(FIELD_ANGLE, focalPlaneToTPMapping)

        # Convert pix->fp to AST format:
        for detector in inputCamera:
            d = detector.getId()
            if d not in pixToFocalPlane:
                # This camera will not include detectors that were not used in
                # astrometric fit.
                continue

            detectorBuilder = cameraBuilder.add(detector.getName(), detector.getId())
            detectorBuilder.setBBox(detector.getBBox())
            detectorBuilder.setPixelSize(detector.getPixelSize())
            for amp in detector.getAmplifiers():
                detectorBuilder.append(amp.rebuild())

            normCoeffs = self.make_ast_polymap_coeffs(
                #1, [-1.0, 2 / detector.getBBox().getWidth(), 0], [-1.0, 0, 2 / detector.getBBox().getHeight()]
                1, [-1.0, 2 / xScale, 0], [-1.0, 0, 2 / yScale]
            )
            normMap = ast.PolyMap(
                normCoeffs,
                2,
                "IterInverse=1, TolInverse=%s, NIterInverse=%s"
                % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
            )
            forwardDetCoeffs = self.make_ast_polymap_coeffs(
                fpDegree, pixToFocalPlane[d]["x"], pixToFocalPlane[d]["y"]
            )
            ccdToFP = ast.PolyMap(
                forwardDetCoeffs,
                2,
                "IterInverse=1, TolInverse=%s, NIterInverse=%s"
                % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
            )
            zoomMap = ast.ZoomMap(2, scaleConvert)
            fullDetMap = normMap.then(ccdToFP).then(rotateAndFlip).then(zoomMap)
            ccdToFPMapping = TransformPoint2ToPoint2(fullDetMap)
            detectorBuilder.setTransformFromPixelsTo(FOCAL_PLANE, ccdToFPMapping)

        outputCamera = cameraBuilder.finish()

        return outputCamera
