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
from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS, Camera
from lsst.afw.geom import TransformPoint2ToPoint2
from lsst.geom import degrees
import lsst.pipe.base as pipeBase
from collections import defaultdict
from astropy.table import Table, vstack


__all__ = [
    "BuildCameraFromAstrometryConfig",
    "BuildCameraFromAstrometryTask",
    "AstrometryCameraInputSelectionConfig",
    "AstrometryCameraInputSelectionConnections",
    "AstrometryCameraInputSelectionTask",
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


class AstrometryCameraInputSelectionConnections(pipeBase.PipelineTaskConnections, dimensions=("instrument",)):
    visit_detector_table = pipeBase.connectionTypes.Input(
        doc="Table with estimate of seeing for each detector.",
        name="preliminary_visit_detector_table",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
        deferLoad=True,
    )
    isolated_star_association = pipeBase.connectionTypes.Input(
        doc="Catalog of isolated stars",
        name="isolated_star_association",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "tract", "skymap"),
        deferLoad=True,
        multiple=True,
    )
    isolated_star = pipeBase.connectionTypes.Input(
        doc="Catalog of individual sources for isolated stars",
        name="isolated_star",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "tract", "skymap"),
        deferLoad=True,
        multiple=True,
    )
    selectedVisits = pipeBase.connectionTypes.Output(
        doc="Selected input visits for building the astrometry camera.",
        name="astrometry_camera_input",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "physical_filter"),
        multiple=True,
    )

class AstrometryCameraInputSelectionConfig(pipeBase.PipelineTaskConfig,
                                           pipelineConnections=AstrometryCameraInputSelectionConnections):
    """Configuration for AstrometryCameraInputSelectionTask."""

    nStarPerTract = lsst.pex.config.Field(
        doc=('Threshold for number of stars per tract. Default is approximately 100 stars per LSSTCam'
             'detector.'),
        dtype=int,
        default=4000,
    )
    seeingThreshold = lsst.pex.config.Field(
        doc="Median seeing threshold above which visits will be rejected.",
        dtype=float,
        default=1.2,
    )
    nVisits = lsst.pex.config.Field(
        doc=("Minimum number of overlapping visits required to include a given field."),
        dtype=int,
        default=20,
    )
    nDetectors = lsst.pex.config.Field(
        doc="Minimum number of detectors per visit with sources that can be included in fit.",
        dtype=int,
        default=130,
    )
    nSourceColumn = lsst.pex.config.Field(
        doc="TODO",
        dtype=str,
        default="nsource_",
    )


class AstrometryCameraInputSelectionTask(pipeBase.PipelineTask):
    """Select input visits for fitting the astrometry_camera."""
    ConfigClass = AstrometryCameraInputSelectionConfig
    _DefaultName = 'astrometryCameraInputSelection'

    def runQuantum(self, butlerQC, inputRefs, outputRefs):

        inputs = butlerQC.get(inputRefs)

        isolated_star_association = inputs.pop("isolated_star_association")
        input_tracts_isa = np.array([inputCat.dataId["tract"] for inputCat in isolated_star_association])
        isolated_star_association = [isolated_star_association[t] for t in input_tracts_isa.argsort()]
        inputs['isolated_star_association'] = isolated_star_association

        isolated_star = inputs.pop("isolated_star")
        input_tracts_is = np.array([inputCat.dataId["tract"] for inputCat in isolated_star])
        isolated_star = [isolated_star[t] for t in input_tracts_is.argsort()]
        inputs['isolated_star'] = isolated_star
        if not (input_tracts_isa == input_tracts_is).all():
            raise RuntimeError("these should match")

        output = self.run(**inputs)
        missingFilters = []
        import ipdb; ipdb.set_trace()
        outputRefDict = {outputRef.dataId['physical_filter']: outputRef for outputRef in outputRefs}
        for outputFilter, outputTable in output:
            if len(output[outputFilter]) > 0:
                butlerQC.put(outputTable, outputRefDict[outputFilter])
            else:
                missingFilters.append(outputFilter)

        if missingFilters:
            e = RuntimeError(f"Filter(s) {','.join(missingFilters)} have an insufficient number of visits meeting the quality cuts")
            error = pipeBase.AnnotatedPartialOutputsError.annotate(e, self, log=self.log)
            raise error from e

    def run(self, visit_detector_table, isolated_star_association, isolated_star):

        all_association_columns = isolated_star_association[0].get().colnames
        needed_columns = [col for col in all_association_columns if self.config.nSourceColumn in col]
        needed_columns.append('isolated_star_id')

        candidateVisitDetectors = defaultdict(list)
        for c, catalog_ref in enumerate(isolated_star_association):
            tract = catalog_ref.dataId['tract']
            if tract % 50 == 0:
                print(f"processing tract {tract}")
            # First, check if this tract has enough sources to be useful:
            preliminary_catalog = catalog_ref.get(parameters={"columns": needed_columns})
            if len(preliminary_catalog) < self.config.nStarPerTract:
                self.log.debug(f"Tract {tract} has too few stars: {len(preliminary_catalog)}")
                continue
            perBandMax = np.array([preliminary_catalog[col].max() for col in preliminary_catalog.columns])
            if perBandMax.max() < self.config.nVisits:
                # No band has enough visits to be useful.
                self.log.debug(f"Tract {tract} has too few visits per star: {perBandMax.max()}")
                continue

            # If the tract passes cuts, load the sources:
            isolated_star_cat = isolated_star[c].get(parameters={"columns": ['visit', 'detector', 'obj_index', 'band', 'physical_filter']})
            isolated_star_cat['isolated_star_id'] = preliminary_catalog[isolated_star_cat['obj_index']]['isolated_star_id']
            for band in np.unique(isolated_star_cat['physical_filter']):
                bandInds = isolated_star_cat['physical_filter'] == band
                visitDetectorSourceCounts = preliminary_catalog[isolated_star_cat[bandInds]['obj_index']][self.config.nSourceColumn + band]
                goodSources = visitDetectorSourceCounts > self.config.nVisits
                candidateVisitDetectors[band].append(isolated_star_cat[bandInds][goodSources][['visit', 'detector', 'isolated_star_id']])
    
        visDetTable = visit_detector_table.get(parameters={"columns": ['visitId', 'detectorId', 'seeing']})
        visDetTable.add_index(['visitId', 'detectorId'])

        outputTables = {}
        for band, sourceList in candidateVisitDetectors.items():
            bandSources = vstack(sourceList)
            bandVisitDetectors = np.unique(bandSources[['visit', 'detector']])
            self.log.info(f"Processing {band} band with {len(np.unique(bandVisitDetectors['visit']))} initial visits.")

            bandVisDetInfo = visDetTable.loc.with_index("visitId", "detectorId")[list(map(tuple, bandVisitDetectors))]
            goodSeeingInds = bandVisDetInfo['seeing'] < self.config.seeingThreshold
            self.log.info(f"Keeping {goodSeeingInds.sum()} out of {len(goodSeeingInds)} detectors with seeing below {self.config.seeingThreshold} asec.")
            
            visits, detectorInverse, detectorCounts = np.unique(bandVisitDetectors[goodSeeingInds]['visit'], return_inverse=True, return_counts=True, sorted=True)
            meanSeeing = np.zeros(len(visits))
            np.add.at(meanSeeing, detectorInverse, bandVisDetInfo[goodSeeingInds]['seeing'])
            meanSeeing /= detectorCounts
            
            selectedVisits = visits[detectorCounts > self.config.nDetectors]
            self.log.info(f"Keeping {len(selectedVisits)} out of {len(visits)} visits with data from at least {self.config.nDetectors} detectors per visit.")

            # Each visit needs to share stars with lots of other visits. Find the average number of times
            # that a visit's stars have been observed, and iteratively cut any that do not meet the 
            # config threshold.
            nCutVisits = 1
            interimSelectedVisits = selectedVisits.copy()
            while nCutVisits > 0:
                # Calculate how many times each star is observed.
                selectedBandSources = bandSources[np.isin(bandSources['visit'], interimSelectedVisits)]
                sources, sourceInverse, sourceCounts = np.unique(selectedBandSources['isolated_star_id'], return_inverse=True, return_counts=True)
                bandSourceCounts = sourceCounts[sourceInverse]

                # Get the mean for each visit.
                reorderedSelectedVisits, visitId = np.unique(selectedBandSources['visit'], return_inverse=True)
                perVisitMeanCounts = np.zeros(len(interimSelectedVisits))
                nSources = np.zeros(len(interimSelectedVisits))
                np.add.at(perVisitMeanCounts, visitId, bandSourceCounts)
                np.add.at(nSources, visitId, 1)
                perVisitMeanCounts /= nSources

                # Cut visits whose stars are not in enough other visits.
                tooFewVisits = perVisitMeanCounts < self.config.nVisits
                nCutVisits = tooFewVisits.sum()
                if nCutVisits > 0:
                    try:
                        self.log.info("Cutting visits " + ",".join(reorderedSelectedVisits[tooFewVisits].value.astype(str)))
                    except:
                        import ipdb; ipdb.set_trace()
                interimSelectedVisits = reorderedSelectedVisits[~tooFewVisits]
            
            finalVisits = interimSelectedVisits
            if len(finalVisits) < self.config.nVisits:
                self.log.warning(f"Band {band} does not have enough visits meeting the required thresholds.")
            self.log.info(f"Keeping {len(finalVisits)} out of {len(selectedVisits)} with an average of at least {self.config.nVisits} matches per source.")
            selectedInds = np.searchsorted(visits, finalVisits, side='left')
            finalDetectorCounts = detectorCounts[selectedInds]
            finalSeeing = meanSeeing[selectedInds]

            bandTable = Table({'visit': finalVisits,
                               'nDetectors': finalDetectorCounts,
                               'meanSeeing': finalSeeing,
                               'meanOverlap': perVisitMeanCounts})
            outputTables[band] = bandTable
        return outputTables





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


class BuildCameraFromAstrometryTask(pipeBase.Task):
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
        self.X = np.linspace(-1, 1, 10)
        self.Y = np.linspace(-1, 1, 10)
        x, y = np.meshgrid(self.X, self.Y)
        self.x = x.ravel()
        self.y = y.ravel()

    def run(self, mapParams, mapTemplate, detectorList, visitList, inputCamera):
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

        # Align the model with the obs_* model camera.
        recenteredParams = self._recenter_model(mapParams, mapTemplate, detectorList, inputCamera)

        if self.config.modelSplitting == "basic":
            # Put all of the camera distortion into the pixels->focal plane
            # part of the distortion model, with the focal plane->tangent plane
            # part only used for scaling between the focal plane and sky.
            pixToFocalPlane, focalPlaneToTangentPlane = self._basic_model(
                recenteredParams.newDeviceArray, detectorList
            )

        else:
            # Fit two polynomials, such that the first describes the pixel->
            # focal plane part of the model, and the second describes the focal
            # plane->tangent plane part of the model, with the goal of
            # generating a more physically-motivated distortion model.
            pixToFocalPlane, focalPlaneToTangentPlane = self._split_model(
                recenteredParams.newTPx, recenteredParams.newTPy, detectorList
            )

        xScale = int(mapTemplate["BAND/DEVICE/poly"]["XMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["XMin"])
        yScale = int(mapTemplate["BAND/DEVICE/poly"]["YMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["YMin"])

        # Turn the mappings into a Camera object.
        camera = self._translate_to_afw(
            pixToFocalPlane,
            focalPlaneToTangentPlane,
            detectorList,
            inputCamera,
            xScale,
            yScale,
        )

        return camera

    def _recenter_model(self, mapParams, mapTemplate, detectorList, camera, transform="affine"):
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
        camera: `lsst.afw.cameraGeom.Camera`
            Existing camera model ontoo which the new model will be normalized.
        transform: `str`
            Method for recentering the camera.
                - ``"affine"``: The camera parameters are shifted by a linear
                function to minimize the difference between this camera model
                and a preexisting model.
                - ``"rigid"`: The camera parameters are shifted by a
                translation and rotation to minimize the difference between
                this camera model and a preexisting model. TODO: DM-56786 This
                needs to be combined with a model for other causes of shifts,
                like refraction or camera flexure, in order to be
                scientifically useful.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            A struct object with the following attributes:
            ``newDeviceArray`` : `np.ndarray`
                Array of NxM, where N is the number of detectors, and M is the
                number of coefficients for each per-detector mapping.
            ``newTPx``, ``newTPy`` : `np.ndarray`
                Projection of `self.x` and `self.y` onto the tangent plane for
                each detector, given the normalized mapping.
            ``inputCoords`` : `np.ndarray`
                Projection of `self.x` and `self.y` onto the tangent plane for
                each detector using the input camera.
            ``targetCoords`` : `np.ndarray`
                Projection of `self.x` and `self.y` onto the tangent plane for
                each detector using the target preexisting camera.
        """
        # Get the per-device part of the model.
        deviceParams = []
        for element in mapTemplate["BAND/DEVICE"]["Elements"]:
            for detector in detectorList:
                detectorTemplate = element.replace("DEVICE", str(detector))
                detectorTemplate = detectorTemplate.replace("BAND", ".+")
                for k, params in mapParams.items():
                    if re.fullmatch(detectorTemplate, k):
                        deviceParams.append(params)
        deviceArray = np.vstack(deviceParams)

        inputTPx = []
        inputTPy = []
        targetTPx = []
        targetTPy = []
        xMax = max([det.getBBox().endX for det in camera]) - 1
        yMax = max([det.getBBox().endY for det in camera]) - 1

        for d, deviceMap in enumerate(deviceArray):
            detector = detectorList[d]
            obsDet = camera[detector]
            detBBox = obsDet.getBBox()

            nCoeffsDev = len(deviceMap) // 2
            deviceDegree = int(-1.5 + 0.5 * (1 + 8 * nCoeffsDev) ** 0.5)

            x = ((detBBox.endX - 1) / xMax) * (self.x + 1) - 1
            y = ((detBBox.endY - 1) / yMax) * (self.y + 1) - 1

            intX = _z_function(deviceMap[:nCoeffsDev], x, y, order=deviceDegree)
            intY = _z_function(deviceMap[nCoeffsDev:], x, y, order=deviceDegree)
            inputTPx.append(intX)
            inputTPy.append(intY)

            # Get the tangent plane mapping for the same points from the target
            # camera model.
            dX = np.linspace(detBBox.beginX, detBBox.endX - 1, len(self.X))
            dY = np.linspace(detBBox.beginY, detBBox.endY - 1, len(self.Y))
            detX, detY = np.meshgrid(dX, dY)
            detX = detX.ravel()
            detY = detY.ravel()
            mapping = obsDet.getTransform(PIXELS, FIELD_ANGLE).getMapping()
            faX, faY = mapping.applyForward(np.array([detX, detY]))
            targetTPx.append(np.degrees(faX))
            targetTPy.append(np.degrees(faY))

        inputCoords = np.array([np.concatenate(inputTPx), np.concatenate(inputTPy)])
        targetCoords = np.array([np.concatenate(targetTPx), np.concatenate(targetTPy)])

        if transform == "rigid":
            # Find the best rigid transformation matching the new model to the
            # input camera model using the method described in
            # https://igl.ethz.ch/projects/ARAP/svd_rot.pdf.
            inputAve = inputCoords.mean(axis=1)
            targetAve = targetCoords.mean(axis=1)

            xVec = inputCoords - inputAve.reshape(-1, 1)
            yVec = targetCoords - targetAve.reshape(-1, 1)

            # Now do SVD:
            SMatrix = xVec.dot(yVec.T)
            U, S, VTrans = np.linalg.svd(SMatrix)

            rotationMatrix = (VTrans.T).dot(U.T)
            translation = targetAve.reshape(-1, 1) - rotationMatrix.dot(inputAve.reshape(-1, 1))

            # Convert the tangent plane coordinates already calculated.
            matchedTPCoords = rotationMatrix.dot(inputCoords) + translation

            # Convert the device map coefficients.
            nCoeffsDev = deviceArray.shape[1] // 2
            newDeviceArray = np.zeros(deviceArray.shape)
            newDeviceArray[:, :nCoeffsDev] = (
                deviceArray[:, :nCoeffsDev] * rotationMatrix[0, 0]
                + deviceArray[:, nCoeffsDev:] * rotationMatrix[0, 1]
            )
            newDeviceArray[:, 0] += translation[0]
            newDeviceArray[:, nCoeffsDev:] = (
                deviceArray[:, :nCoeffsDev] * rotationMatrix[1, 0]
                + deviceArray[:, nCoeffsDev:] * rotationMatrix[1, 1]
            )
            newDeviceArray[:, nCoeffsDev] += translation[1]

        elif transform == "affine":
            # Find the transformation parameters by minimizing
            # (target coords - transforms X input coords)**2.
            x = np.vstack([np.ones_like(inputCoords[0]), inputCoords])
            A = x.dot(x.T)
            AInv = np.linalg.inv(A)
            bx = (x * targetCoords[0]).sum(axis=1)
            by = (x * targetCoords[1]).sum(axis=1)
            transformX = AInv.dot(bx)
            transformY = AInv.dot(by)

            matchedTPCoords = np.array([transformX, transformY]).dot(x)

            # Convert the device map coefficients.
            nCoeffsDev = deviceArray.shape[1] // 2
            newDeviceArray = np.zeros(deviceArray.shape)
            newDeviceArray[:, :nCoeffsDev] = (
                deviceArray[:, :nCoeffsDev] * transformX[1] + deviceArray[:, nCoeffsDev:] * transformX[2]
            )
            newDeviceArray[:, 0] += transformX[0]
            newDeviceArray[:, nCoeffsDev:] = (
                deviceArray[:, :nCoeffsDev] * transformY[1] + deviceArray[:, nCoeffsDev:] * transformY[2]
            )
            newDeviceArray[:, nCoeffsDev] += transformY[0]

        else:
            raise RuntimeError("Transform '%s' is not supported", transform)

        newTPx, newTPy = matchedTPCoords

        return pipeBase.Struct(
            newDeviceArray=newDeviceArray,
            newTPx=newTPx,
            newTPy=newTPy,
            inputCoords=inputCoords,
            targetCoords=targetCoords,
        )

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
        self, pixToFocalPlane, focalPlaneToTangentPlane, detectorList, inputCamera, xScale, yScale
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

        ccdZoom = ast.ZoomMap(2, 1 / scaleConvert)
        ccdToSky = ast.PolyMap(
            forwardCoeffs,
            2,
            "IterInverse=1, TolInverse=%s, NIterInverse=%s"
            % (self.config.astInversionTolerance / 2.0, self.config.astInversionMaxIter),
        )
        toRadians = ast.ZoomMap(2, (1 * degrees).asRadians())

        fullMapping = ccdZoom.then(ccdToSky).then(toRadians)
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
                1,
                [-1.0, 2 / xScale, 0],
                [-1.0, 0, 2 / yScale],
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
            fullDetMap = normMap.then(ccdToFP).then(zoomMap)
            ccdToFPMapping = TransformPoint2ToPoint2(fullDetMap)
            detectorBuilder.setTransformFromPixelsTo(FOCAL_PLANE, ccdToFPMapping)

        outputCamera = cameraBuilder.finish()

        return outputCamera
