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
import time
from collections import defaultdict

import astshim as ast
import lsst.pex.config
import numpy as np
import scipy.optimize
from lsst.afw.cameraGeom import (
    ACTUAL_PIXELS,
    FIELD_ANGLE,
    FOCAL_PLANE,
    PIXELS,
    TAN_PIXELS,
    Camera,
    CameraSys,
    DetectorConfig,
)
from lsst.afw.geom import TransformPoint2ToPoint2, transformConfig
from lsst.geom import degrees
from lsst.pipe.base import Task

from .gbdesAstrometricFit import _degreeFromNCoeffs, _nCoeffsFromDegree

__all__ = [
    "BuildCameraConfig",
    "BuildCameraTask",
]

cameraSysList = [FIELD_ANGLE, FOCAL_PLANE, PIXELS, TAN_PIXELS, ACTUAL_PIXELS]
cameraSysMap = dict((sys.getSysName(), sys) for sys in cameraSysList)

Nfeval = 0


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
    deriv_coeffs = np.zeros(coeffs.shape)
    deriv_coeffs[:, :-1] = coeffs[:, 1:]
    deriv_coeffs[:, :-1] *= np.arange(1, order + 1)

    z = np.polynomial.polynomial.polyval2d(x, y, deriv_coeffs.T)
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
    deriv_coeffs = np.zeros(coeffs.shape)
    deriv_coeffs[:-1] = coeffs[1:]
    deriv_coeffs[:-1] *= np.arange(1, order + 1).reshape(-1, 1)

    z = np.polynomial.polynomial.polyval2d(x, y, deriv_coeffs.T)
    return z


class BuildCameraConfig(lsst.pex.config.Config):
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
        default=1.0,
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


class BuildCameraTask(Task):
    """Build an `lsst.afw.cameraGeom.Camera` object out of the `gbdes`
    polynomials mapping from pixels to the tangent plane.

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera object from which to take pupil function, name, and other
        properties.
    detectorList : `list` [`int`]
        List of detector ids.
    visitList : `list` [`int`]
        List of ids for visits that were used to train the input model.
    """

    ConfigClass = BuildCameraConfig
    _DefaultName = "buildCamera"

    def __init__(self, camera, detectorList, visitList, **kwargs):
        super().__init__(**kwargs)
        self.camera = camera
        self.detectorList = detectorList
        self.visitList = visitList

        # The gbdes model normalizes the pixel positions to the range -1 - +1.
        X = np.arange(-1, 1, 0.1)
        Y = np.arange(-1, 1, 0.1)
        x, y = np.meshgrid(X, Y)
        self.x = x.ravel()
        self.y = y.ravel()

    def run(self, mapParams, mapTemplate):
        """Convert the model parameters into a Camera object.

        Parameters
        ----------
        mapParams : `dict` [`float`]
            Parameters that describe the per-detector and per-visit parts of
            the astrometric model.
        mapTemplate : `dict`
            Dictionary describing the format of the astrometric model,
            following the convention in `gbdes`.

        Returns
        -------
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object with transformations set by the input mapParams.
        """

        # Normalize the model.
        newParams, newIntX, newIntY = self._normalizeModel(mapParams, mapTemplate)

        if self.config.modelSplitting == "basic":
            # Put all of the camera distortion into the pixels->focal plane
            # part of the distortion model, with the focal plane->tangent plane
            # part only used for scaling between the focal plane and sky.
            pixToFocalPlane, focalPlaneToTangentPlane = self._basicModel(newParams)

        else:
            # Fit two polynomials, such that the first describes the pixel->
            # focal plane part of the model, and the second describes the focal
            # plane->tangent plane part of the model, with the goal of
            # generating a more physically-motivated distortion model.
            pixToFocalPlane, focalPlaneToTangentPlane = self._splitModel(newIntX, newIntY)

        # Turn the mappings into a Camera object.
        camera = self._translateToAfw(pixToFocalPlane, focalPlaneToTangentPlane)

        return camera

    def _normalizeModel(self, mapParams, mapTemplate):
        """Normalize the camera mappings, such that they correspond to the
        average visit.

        In order to break degeneracies, `gbdes` sets one visit to have the
        identity for the per-visit part of its model. This visit is thus
        imprinted in the per-detector part of the model. This function
        normalizes the model such that the parameters correspond instead to the
        average per-visit behavior. This step assumes that the per-visit part
        of the model is a first-order polynomial, in which case this conversion
        can be done without loss of information.

        Parameters
        ----------
        mapParams : `dict` [`float`]
            Parameters that describe the per-detector and per-visit parts of
            the astrometric model.
        mapTemplate : `dict`
            Dictionary describing the format of the astrometric model,
            following the convention in `gbdes`.

        Returns
        -------
        newDeviceArray : `np.ndarray`
            Array of NxM, where N is the number of detectors, and M is the
            number of coefficients for each per-detector mapping.
        newIntX, newIntY : `np.ndarray`
            Projection of `self.x` and `self.y` onto the tangent plane, given
            the normalized mapping.
        """

        # Get the per-device and per-visit parts of the model.
        deviceParams = []
        visitParams = []
        for element in mapTemplate["BAND/DEVICE"]["Elements"]:
            for detector in self.detectorList:
                detectorTemplate = element.replace("DEVICE", str(detector))
                detectorTemplate = detectorTemplate.replace("BAND", ".+")
                for k, params in mapParams.items():
                    if re.fullmatch(detectorTemplate, k):
                        deviceParams.append(params)
        deviceArray = np.vstack(deviceParams)

        for element in mapTemplate["EXPOSURE"]["Elements"]:
            for visit in self.visitList:
                visitTemplate = element.replace("EXPOSURE", str(visit))
                for k, params in mapParams.items():
                    if re.fullmatch(visitTemplate, k):
                        visitParams.append(params)
        identityVisitParams = np.array([0, 1, 0, 0, 0, 1])
        visitParams.append(identityVisitParams)
        expoArray = np.vstack(visitParams)

        expoMean = expoArray.mean(axis=0)

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

        # Then get the interim positions from the new device model:
        newIntX = []
        newIntY = []
        for deviceMap in newDeviceArray:
            nCoeffsDev = len(deviceMap) // 2
            deviceDegree = _degreeFromNCoeffs(nCoeffsDev)

            intX = _z_function(deviceMap[:nCoeffsDev], self.x, self.y, order=deviceDegree)
            intY = _z_function(deviceMap[nCoeffsDev:], self.x, self.y, order=deviceDegree)
            newIntX.append(intX)
            newIntY.append(intY)

        newIntX = np.array(newIntX).ravel()
        newIntY = np.array(newIntY).ravel()

        return newDeviceArray, newIntX, newIntY

    def _basicModel(self, modelParameters):
        """This will just convert the pix->fp parameters into the right format,
        and return an identity mapping for the fp->tp part.

        Parameters
        ----------
        modelParameters : `np.ndarray`
            Array of NxM, where N is the number of detectors, and M is the
            number of coefficients for each per-detector mapping.

        Returns
        -------
        pixelsToFocalPlane : `dict` [`int`, [`string`, `float`]]
            Dictionary giving the per-detector pixel to focal plane mapping,
            with keys for each detector giving the x and y transformation
            polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Dictionary giving the focal plane to tangent plane mapping.
        """

        nCoeffsFP = modelParameters.shape[1] // 2
        pixelsToFocalPlane = defaultdict(dict)
        for d, det in enumerate(self.detectorList):
            pixelsToFocalPlane[det]["x"] = modelParameters[d][:nCoeffsFP]
            pixelsToFocalPlane[det]["y"] = modelParameters[d][nCoeffsFP:]

        focalPlaneToTangentPlane = {"x": [0, 1, 0], "y": [0, 0, 1]}

        return pixelsToFocalPlane, focalPlaneToTangentPlane

    def _splitModel(self, targetX, targetY, pixToFPGuess=None, fpToTpGuess=None):
        """Fit a two-step model, with one polynomial per detector fitting the
        pixels to focal plane part, followed by a polynomial fitting the focal
        plane to tangent plane part.

        The linear parameters of the focal plane to tangent plane model are
        fixed to be an identity map to remove degeneracies.

        Parameters
        ----------
        targetX, targetY : `np.ndarray`
            Target x and y values in the tangent plane.
        pixToFPGuess : `dict` [`int`, [`string`, `float`]]
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

        nCoeffsFP = _nCoeffsFromDegree(fpDegree)
        nCoeffsFP_tot = len(self.detectorList) * nCoeffsFP * 2

        nCoeffsTP = _nCoeffsFromDegree(tpDegree)
        # The constant and linear terms will be fixed to remove degeneracy with
        # the pix->fp part of the model
        nCoeffsFixed = 3
        nCoeffsFree = nCoeffsTP - nCoeffsFixed
        fx_params = np.zeros(nCoeffsTP * 2)
        fx_params[1] = 1
        fx_params[nCoeffsTP + 2] = 1

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
            intX_tot = []
            intY_tot = []
            for i in range(len(self.detectorList)):
                intX = _z_function(
                    params[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                intY = _z_function(
                    params[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)], self.x, self.y, order=fpDegree
                )
                intX_tot.append(intX)
                intY_tot.append(intY)
            intX_tot = np.array(intX_tot).ravel()
            intY_tot = np.array(intY_tot).ravel()

            tpParams = fx_params.copy()
            tpParams[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFP_tot : nCoeffsFP_tot + nCoeffsFree]
            tpParams[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFP_tot + nCoeffsFree :]

            tpX = _z_function(tpParams[:nCoeffsTP], intX_tot, intY_tot, order=tpDegree)
            tpY = _z_function(tpParams[nCoeffsTP:], intX_tot, intY_tot, order=tpDegree)
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
            fp_params = params[:nCoeffsFP_tot]
            tp_params = fx_params.copy()
            tp_params[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFP_tot : nCoeffsFP_tot + nCoeffsFree]
            tp_params[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFP_tot + nCoeffsFree :]
            intX_tot = []
            intY_tot = []
            # Fill in the derivatives for the pix->fp terms.
            for i in range(len(self.detectorList)):
                dX_i = dX[nX * i : nX * (i + 1)]
                dY_i = dY[nX * i : nX * (i + 1)]
                intX = _z_function(
                    fp_params[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                intY = _z_function(
                    fp_params[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
                    self.x,
                    self.y,
                    order=fpDegree,
                )
                intX_tot.append(intX)
                intY_tot.append(intY)

                dTpX_dfpX = _z_function_dx(tp_params[:nCoeffsTP], intX, intY, order=tpDegree)
                dTpX_dfpY = _z_function_dy(tp_params[:nCoeffsTP], intX, intY, order=tpDegree)
                dTpY_dfpX = _z_function_dx(tp_params[nCoeffsTP:], intX, intY, order=tpDegree)
                dTpY_dfpY = _z_function_dy(tp_params[nCoeffsTP:], intX, intY, order=tpDegree)

                dTpX_part = np.concatenate([xyArray * dTpX_dfpX, xyArray * dTpX_dfpY])
                dTpY_part = np.concatenate([xyArray * dTpY_dfpX, xyArray * dTpY_dfpY])
                jacobian_i = (dX_i * dTpX_part + dY_i * dTpY_part).sum(axis=1)
                jacobian[2 * nCoeffsFP * i : 2 * nCoeffsFP * (i + 1)] = jacobian_i

            intX_tot = np.array(intX_tot).ravel()
            intY_tot = np.array(intY_tot).ravel()

            # Fill in the derivatives for the fp->tp terms.
            for j in range(nCoeffsFree):
                tParams = np.zeros(nCoeffsTP)
                tParams[nCoeffsFixed + j] = 1
                tmpZ = _z_function(tParams, intX_tot, intY_tot, order=tpDegree)
                jacobian[nCoeffsFP_tot + j] = (dX * tmpZ).sum()
                jacobian[nCoeffsFP_tot + nCoeffsFree + j] = (dY * tmpZ).sum()
            return jacobian

        def hessian(params):
            # The Hessian of min_function, which is needed for the minimizer.
            hessian = np.zeros((len(params), len(params)))

            fp_params = params[:nCoeffsFP_tot]
            tp_params = fx_params.copy()
            tp_params[nCoeffsFixed:nCoeffsTP] = params[nCoeffsFP_tot : nCoeffsFP_tot + nCoeffsFree]
            tp_params[nCoeffsFixed + nCoeffsTP :] = params[nCoeffsFP_tot + nCoeffsFree :]
            intX_tot = []
            intY_tot = []

            # Loop over the detectors to fill in the d(pix->fp)**2 and
            # d(pix->fp)d(fp->tp) cross terms.
            for i in range(len(self.detectorList)):
                intX = _z_function(
                    fp_params[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP], self.x, self.y, order=fpDegree
                )
                intY = _z_function(
                    fp_params[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
                    self.x,
                    self.y,
                    order=fpDegree,
                )
                intX_tot.append(intX)
                intY_tot.append(intY)
                dTpX_dfpX = _z_function_dx(tp_params[:nCoeffsTP], intX, intY, order=tpDegree)
                dTpX_dfpY = _z_function_dy(tp_params[:nCoeffsTP], intX, intY, order=tpDegree)
                dTpY_dfpX = _z_function_dx(tp_params[nCoeffsTP:], intX, intY, order=tpDegree)
                dTpY_dfpY = _z_function_dy(tp_params[nCoeffsTP:], intX, intY, order=tpDegree)

                dTpX_part = np.concatenate([xyArray * dTpX_dfpX, xyArray * dTpX_dfpY])
                dTpY_part = np.concatenate([xyArray * dTpY_dfpX, xyArray * dTpY_dfpY])

                for l in range(2 * nCoeffsFP):
                    for m in range(2 * nCoeffsFP):
                        hessian[2 * nCoeffsFP * i + l, 2 * nCoeffsFP * i + m] = (
                            2 * (dTpX_part[l] * dTpX_part[m] + dTpY_part[l] * dTpY_part[m]).sum()
                        )

                    for j in range(nCoeffsFree):
                        tParams = np.zeros(nCoeffsTP)
                        tParams[nCoeffsFixed + j] = 1
                        tmpZ = _z_function(tParams, intX, intY, order=tpDegree)

                        hessX = 2 * (dTpX_part[l] * tmpZ).sum()
                        hessY = 2 * (dTpY_part[l] * tmpZ).sum()
                        # dTP_x part
                        hessian[2 * nCoeffsFP * i + l, nCoeffsFP_tot + j] = hessX
                        hessian[nCoeffsFP_tot + j, 2 * nCoeffsFP * i + l] = hessX
                        # dTP_y part
                        hessian[2 * nCoeffsFP * i + l, nCoeffsFP_tot + nCoeffsFree + j] = hessY
                        hessian[nCoeffsFP_tot + nCoeffsFree + j, 2 * nCoeffsFP * i + l] = hessY

            intX_tot = np.array(intX_tot).ravel()
            intY_tot = np.array(intY_tot).ravel()
            tmpZ_array = np.zeros((nCoeffsFree, nX * len(self.detectorList)))
            # Finally, get the d(fp->tp)**2 terms
            for j in range(nCoeffsFree):
                tParams = np.zeros(nCoeffsTP)
                tParams[nCoeffsFixed + j] = 1
                tmpZ_array[j] = _z_function(tParams, intX_tot, intY_tot, order=tpDegree)
            for j in range(nCoeffsFree):
                for m in range(nCoeffsFree):
                    # X-Y cross terms are zero
                    hess = 2 * (tmpZ_array[j] * tmpZ_array[m]).sum()
                    hessian[nCoeffsFP_tot + j, nCoeffsFP_tot + m] = hess
                    hessian[nCoeffsFP_tot + nCoeffsFree + j, nCoeffsFP_tot + nCoeffsFree + m] = hess
            return hessian

        global Nfeval
        Nfeval = 0

        def callbackMF(params):
            global Nfeval
            self.log.info(f"Iteration {Nfeval}, function value {min_function(params)}")
            Nfeval += 1

        initGuess = np.zeros(nCoeffsFP_tot + 2 * nCoeffsFree)
        if pixToFPGuess:
            useVar = min(nCoeffsFP, len(pixToFPGuess[0]["x"]))
            for i, det in enumerate(self.detectorList):
                initGuess[2 * nCoeffsFP * i : 2 * nCoeffsFP * i + useVar] = pixToFPGuess[det]["x"][:useVar]
                initGuess[(2 * i + 1) * nCoeffsFP : (2 * i + 1) * nCoeffsFP + useVar] = pixToFPGuess[det][
                    "y"
                ][:useVar]
        if fpToTpGuess:
            useVar = min(nCoeffsTP, len(fpToTpGuess["x"]))
            initGuess[nCoeffsFP_tot : nCoeffsFP_tot + useVar - nCoeffsFixed] = fpToTpGuess["x"][
                nCoeffsFixed:useVar
            ]
            initGuess[nCoeffsFP_tot + nCoeffsFree : nCoeffsFP_tot + nCoeffsFree + useVar - nCoeffsFixed] = (
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
        for i, det in enumerate(self.detectorList):
            pixToFP[det] = {
                "x": res.x[2 * nCoeffsFP * i : (2 * i + 1) * nCoeffsFP],
                "y": res.x[(2 * i + 1) * nCoeffsFP : 2 * nCoeffsFP * (1 + i)],
            }

        fpToTpAll = fx_params.copy()
        fpToTpAll[nCoeffsFixed:nCoeffsTP] = res.x[nCoeffsFP_tot : nCoeffsFP_tot + nCoeffsFree]
        fpToTpAll[nCoeffsFixed + nCoeffsTP :] = res.x[nCoeffsFP_tot + nCoeffsFree :]
        fpToTp = {"x": fpToTpAll[:nCoeffsTP], "y": fpToTpAll[nCoeffsTP:]}

        return pixToFP, fpToTp

    def makeAstPolyMapCoeffs(self, order, xCoeffs, yCoeffs):
        """Convert polynomial coefficients in gbdes format to AST PolyMap
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
        focalPlaneToPupil: `lsst.afw.geom.TransformPoint2ToPoint2`
            Transform from focal plane to field angle coordinates
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

    def _translateToAfw(self, pixToFocalPlane, focalPlaneToTangentPlane):
        """Convert the model parameters to a Camera object.

        Parameters
        ----------
        pixelsToFocalPlane : `dict` [`int`, [`string`, `float`]]
            Dictionary giving the per-detector pixel to focal plane mapping,
            with keys for each detector giving the x and y transformation
            polynomials.
        focalPlaneToTangentPlane : `dict` [`string`, `float`]
            Dictionary giving the focal plane to tangent plane mapping.

        Returns
        -------
        output_camera : `lsst.afw.cameraGeom.Camera`
            Camera object containing the pix->fp and fp->tp transforms.
        """
        if self.config.modelSplitting == "basic":
            tpDegree = 1
            fpDegree = _degreeFromNCoeffs(len(pixToFocalPlane[self.detectorList[0]]["x"]))
        else:
            tpDegree = self.config.tangentPlaneDegree
            fpDegree = self.config.focalPlaneDegree

        scaleConvert = (1 * degrees).asArcseconds() / self.config.plateScale

        cameraBuilder = Camera.Builder(self.camera.getName())
        cameraBuilder.setPupilFactoryName(self.camera.getPupilFactoryName())

        # Convert fp->tp to AST format:
        forwardCoeffs = self.makeAstPolyMapCoeffs(
            tpDegree, focalPlaneToTangentPlane["x"], focalPlaneToTangentPlane["y"]
        )
        rotateAndFlipCoeffs = self.makeAstPolyMapCoeffs(1, [0, 0, -1], [0, -1, 0])

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
        for detector in self.camera:
            d = detector.getId()
            if d not in pixToFocalPlane:
                # TODO: just grab detector map from the obs_ package.
                continue

            detectorBuilder = cameraBuilder.add(detector.getName(), detector.getId())
            # TODO: Add all the detector details like pixel size, etc.
            for amp in detector.getAmplifiers():
                detectorBuilder.append(amp.rebuild())

            normCoeffs = self.makeAstPolyMapCoeffs(
                1, [-1.0, 2 / detector.getBBox().getWidth(), 0], [-1.0, 0, 2 / detector.getBBox().getHeight()]
            )
            normMap = ast.PolyMap(
                normCoeffs,
                2,
                "IterInverse=1, TolInverse=%s, NIterInverse=%s"
                % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
            )
            forwardDetCoeffs = self.makeAstPolyMapCoeffs(
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

        output_camera = cameraBuilder.finish()

        return output_camera
