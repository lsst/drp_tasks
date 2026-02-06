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
import dataclasses
import re

import astropy.coordinates
import astropy.time
import astropy.units as u
import astshim
import numpy as np
import wcsfit
import yaml
from astropy.table import vstack
from sklearn.cluster import AgglomerativeClustering
from smatch.matcher import Matcher

import lsst.afw.geom as afwgeom
import lsst.afw.table
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.sphgeom
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    ReferenceSourceSelectorTask,
)
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry

from .build_camera import BuildCameraFromAstrometryTask

__all__ = [
    "calculate_apparent_motion",
    "GbdesAstrometricFitConnections",
    "GbdesAstrometricFitConfig",
    "GbdesAstrometricFitTask",
    "GbdesAstrometricMultibandFitConnections",
    "GbdesAstrometricMultibandFitTask",
    "GbdesGlobalAstrometricFitConnections",
    "GbdesGlobalAstrometricFitConfig",
    "GbdesGlobalAstrometricFitTask",
    "GbdesGlobalAstrometricMultibandFitConnections",
    "GbdesGlobalAstrometricMultibandFitTask",
]


def calculate_apparent_motion(catalog, refEpoch):
    """Calculate shift from reference epoch to the apparent observed position
    at another date.

    This function calculates the shift due to proper motion combined with the
    apparent motion due to parallax. This is not used in the
    `GbdesAstrometricFitTask` or related child tasks, but is useful for
    assessing results.

    Parameters
    ----------
    catalog : `astropy.table.Table`
        Table containing position, proper motion, parallax, and epoch for each
        source, labeled by columns 'ra', 'dec', 'pmRA', 'pmDec', 'parallax',
        and 'MJD'.
    refEpoch : `astropy.time.Time`
        Epoch of the reference position.

    Returns
    -------
    apparentMotionRA : `np.ndarray` [`astropy.units.Quantity`]
        RA shift in degrees.
    apparentMotionDec : `np.ndarray` [`astropy.units.Quantity`]
        Dec shift in degrees.
    """
    ra_rad = catalog["ra"].to(u.rad)
    dec_rad = catalog["dec"].to(u.rad)

    dt = (catalog["MJD"] - refEpoch).to(u.yr)
    properMotionRA = catalog["pmRA"].to(u.deg / u.yr) * dt
    properMotionDec = catalog["pmDec"].to(u.deg / u.yr) * dt

    # Just do calculations for unique mjds:
    mjds = astropy.time.Time(
        np.unique(catalog["MJD"].mjd), scale=catalog["MJD"][0].scale, format=catalog["MJD"][0].format
    )
    sun = astropy.coordinates.get_body("sun", time=mjds)
    frame = astropy.coordinates.GeocentricTrueEcliptic(equinox=mjds)
    tmpSunLongitudes = sun.transform_to(frame).lon.radian

    # Project back to full table:
    newTable = astropy.table.Table({"MJD": mjds, "SL": tmpSunLongitudes})
    joined = astropy.table.join(catalog[["MJD"]], newTable, keys="MJD", keep_order=True)
    sunLongitudes = joined["SL"]

    # These equations for parallax come from Equations 5.2 in Van de Kamp's
    # book Stellar Paths. They differ from the parallax calculated in gbdes by
    # ~0.01 mas, which is acceptable for QA and plotting purposes.
    parallaxFactorRA = np.cos(wcsfit.EclipticInclination) * np.cos(ra_rad) * np.sin(sunLongitudes) - np.sin(
        ra_rad
    ) * np.cos(sunLongitudes)
    parallaxFactorDec = (
        np.sin(wcsfit.EclipticInclination) * np.cos(dec_rad)
        - np.cos(wcsfit.EclipticInclination) * np.sin(ra_rad) * np.sin(dec_rad)
    ) * np.sin(sunLongitudes) - np.cos(ra_rad) * np.sin(dec_rad) * np.cos(sunLongitudes)
    parallaxDegrees = catalog["parallax"].to(u.degree)
    parallaxCorrectionRA = parallaxDegrees * parallaxFactorRA
    parallaxCorrectionDec = parallaxDegrees * parallaxFactorDec

    apparentMotionRA = properMotionRA + parallaxCorrectionRA
    apparentMotionDec = properMotionDec + parallaxCorrectionDec

    return apparentMotionRA, apparentMotionDec


def _make_ref_covariance_matrix(
    refCat, inputUnit=u.radian, outputCoordUnit=u.marcsec, outputPMUnit=u.marcsec, version=1
):
    """Make a covariance matrix for the reference catalog including proper
    motion and parallax.

    The output is flattened to one dimension to match the format expected by
    `gbdes`.

    Parameters
    ----------
    refCat : `lsst.afw.table.SimpleCatalog`
        Catalog including proper motion and parallax measurements.
    inputUnit : `astropy.unit.core.Unit`
        Units of the input catalog
    outputCoordUnit : `astropy.unit.core.Unit`
        Units required for the coordinates in the covariance matrix. `gbdes`
        expects milliarcseconds.
    outputPMUnit : `astropy.unit.core.Unit`
        Units required for the proper motion/parallax in the covariance matrix.
        `gbdes` expects milliarcseconds.
    version : `int`
        Version of the reference catalog. Version 2 includes covariance
        measurements.
    Returns
    -------
    cov : `list` [`float`]
        Flattened output covariance matrix.
    """
    cov = np.zeros((len(refCat), 25))
    if version == 1:
        # Here is the standard ordering of components in the cov matrix,
        # to match the PM enumeration in C++ code of gbdes package's Match.
        # Each tuple gives: the array holding the 1d error,
        #                   the string in Gaia column names for this
        #                   the ordering in the Gaia catalog
        # and the ordering of the tuples is the order we want in our cov matrix
        raErr = (refCat["coord_raErr"]).to(outputCoordUnit).to_value()
        decErr = (refCat["coord_decErr"]).to(outputCoordUnit).to_value()
        raPMErr = (refCat["pm_raErr"]).to(outputPMUnit).to_value()
        decPMErr = (refCat["pm_decErr"]).to(outputPMUnit).to_value()
        parallaxErr = (refCat["parallaxErr"]).to(outputPMUnit).to_value()
        stdOrder = (
            (raErr, "ra", 0),
            (decErr, "dec", 1),
            (raPMErr, "pmra", 3),
            (decPMErr, "pmdec", 4),
            (parallaxErr, "parallax", 2),
        )

        k = 0
        for i, pr1 in enumerate(stdOrder):
            for j, pr2 in enumerate(stdOrder):
                if pr1[2] < pr2[2]:
                    cov[:, k] = 0
                elif pr1[2] > pr2[2]:
                    cov[:, k] = 0
                else:
                    # diagnonal element
                    cov[:, k] = pr1[0] * pr2[0]
                k = k + 1

    elif version == 2:
        positionParameters = ["coord_ra", "coord_dec", "pm_ra", "pm_dec", "parallax"]
        units = [outputCoordUnit, outputCoordUnit, outputPMUnit, outputPMUnit, outputPMUnit]
        k = 0
        for i, pi in enumerate(positionParameters):
            for j, pj in enumerate(positionParameters):
                if i == j:
                    cov[:, k] = ((refCat[f"{pi}Err"].value) ** 2 * inputUnit**2).to(units[j] * units[j]).value
                elif i > j:
                    cov[:, k] = (refCat[f"{pj}_{pi}_Cov"].value * inputUnit**2).to_value(units[i] * units[j])
                else:
                    cov[:, k] = (refCat[f"{pi}_{pj}_Cov"].value * inputUnit**2).to_value(units[i] * units[j])
                k += 1
    return cov


def _nCoeffsFromDegree(degree):
    """Get the number of coefficients for a polynomial of a certain degree with
    two variables.

    This uses the general formula that the number of coefficients for a
    polynomial of degree d with n variables is (n + d) choose d, where in this
    case n is fixed to 2.

    Parameters
    ----------
    degree : `int`
        Degree of the polynomial in question.

    Returns
    -------
    nCoeffs : `int`
        Number of coefficients for the polynomial in question.
    """
    nCoeffs = int((degree + 2) * (degree + 1) / 2)
    return nCoeffs


def _degreeFromNCoeffs(nCoeffs):
    """Get the degree for a polynomial with two variables and a certain number
    of coefficients.

    This is done by applying the quadratic formula to the
    formula for calculating the number of coefficients of the polynomial.

    Parameters
    ----------
    nCoeffs : `int`
        Number of coefficients for the polynomial in question.

    Returns
    -------
    degree : `int`
        Degree of the polynomial in question.
    """
    degree = int(-1.5 + 0.5 * (1 + 8 * nCoeffs) ** 0.5)
    return degree


def _convert_to_ast_polymap_coefficients(coefficients):
    """Convert vector of polynomial coefficients from the format used in
    `gbdes` into AST format (see Poly2d::vectorIndex(i, j) in
    gbdes/gbutil/src/Poly2d.cpp). This assumes two input and two output
    coordinates.

    Parameters
    ----------
    coefficients : `list`
        Coefficients of the polynomials.
    degree : `int`
        Degree of the polynomial.

    Returns
    -------
    astPoly : `astshim.PolyMap`
        Coefficients in AST polynomial format.
    """
    polyArray = np.zeros((len(coefficients), 4))
    N = len(coefficients) / 2
    degree = _degreeFromNCoeffs(N)

    for outVar in [1, 2]:
        for i in range(degree + 1):
            for j in range(degree + 1):
                if (i + j) > degree:
                    continue
                vectorIndex = int(((i + j) * (i + j + 1)) / 2 + j + N * (outVar - 1))
                polyArray[vectorIndex, 0] = coefficients[vectorIndex]
                polyArray[vectorIndex, 1] = outVar
                polyArray[vectorIndex, 2] = i
                polyArray[vectorIndex, 3] = j

    astPoly = astshim.PolyMap(polyArray, 2, options="IterInverse=1,NIterInverse=10,TolInverse=1e-7")
    return astPoly


def _get_instruments(inputVisitSummaries):
    """Make `wcsfit.Instrument` objects for all of the instruments and filters
    used for the input visits. This also returns the indices to match the
    visits to the instrument/filter used.

    Parameters
    ----------
    inputVisitSummaries: `list` [`lsst.afw.table.ExposureCatalog`]
        List of catalogs with per-detector summary information.

    Returns
    -------
    instruments : `list` [`wcsfit.Instrument`]
        List of instrument objects.
    instrumentIndices : `list` [`int`]
        Indices matching each visit to the instrument/filter used.
    """
    instruments = []
    filters = []
    instrumentIndices = []
    for visitSummary in inputVisitSummaries:
        filter = visitSummary[0]["physical_filter"]
        instrumentName = visitSummary[0].getVisitInfo().instrumentLabel
        if filter not in filters:
            filters.append(filter)
            filter_instrument = wcsfit.Instrument(instrumentName)
            filter_instrument.band = filter
            instruments.append(filter_instrument)
        instrumentIndices.append(filters.index(filter))
    return instruments, instrumentIndices


class CholeskyError(pipeBase.AlgorithmError):
    """Raised if the Cholesky decomposition in the model fit fails."""

    def __init__(self) -> None:
        super().__init__(
            "Cholesky decomposition failed, likely because data is not sufficient to constrain the model."
        )

    @property
    def metadata(self) -> dict:
        """There is no metadata associated with this error."""
        return {}


class GbdesAstrometricFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("skymap", "tract", "instrument", "physical_filter"),
    defaultTemplates={
        "outputName": "gbdesAstrometricFit",
    },
):
    """Middleware input/output connections for task data."""

    inputCatalogRefs = pipeBase.connectionTypes.Input(
        doc="Source table in parquet format, per visit.",
        name="preSourceTable_visit",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        multiple=True,
    )
    inputVisitSummaries = pipeBase.connectionTypes.Input(
        doc=(
            "Per-visit consolidated exposure metadata built from calexps. "
            "These catalogs use detector id for the id and must be sorted for "
            "fast lookups of a detector."
        ),
        name="visitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    referenceCatalog = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The astrometry reference catalog to match to loaded input catalog sources.",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )
    colorCatalog = pipeBase.connectionTypes.Input(
        doc="The catalog of magnitudes to match to input sources.",
        name="fgcm_Cycle4_StandardStars",
        storageClass="SimpleCatalog",
        dimensions=("instrument",),
    )
    inputCameraModel = pipeBase.connectionTypes.PrerequisiteInput(
        doc="Camera parameters to use for 'device' part of model",
        name="gbdesAstrometricFit_cameraModel",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    inputCamera = pipeBase.connectionTypes.PrerequisiteInput(
        doc="Input camera to use when constructing camera from astrometric model.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    outputWcs = pipeBase.connectionTypes.Output(
        doc=(
            "Per-tract, per-visit world coordinate systems derived from the fitted model."
            " These catalogs only contain entries for detectors with an output, and use"
            " the detector id for the catalog id, sorted on id for fast lookups of a detector."
        ),
        name="{outputName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "skymap", "tract"),
        multiple=True,
    )
    outputCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="{outputName}_fitStars",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    starCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of best-fit object positions. Also includes the fit proper motion and parallax if "
            "fitProperMotion is True."
        ),
        name="{outputName}_starCatalog",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    modelParams = pipeBase.connectionTypes.Output(
        doc="WCS parameters and covariance.",
        name="{outputName}_modelParams",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    outputCameraModel = pipeBase.connectionTypes.Output(
        doc="Camera parameters to use for 'device' part of model",
        name="{outputName}_cameraModel",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    camera = pipeBase.connectionTypes.Output(
        doc="Camera object constructed using the per-detector part of the astrometric model",
        name="{outputName}Camera",
        storageClass="Camera",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    dcrCoefficients = pipeBase.connectionTypes.Output(
        doc="Per-visit coefficients for DCR correction.",
        name="{outputName}_dcrCoefficients",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )

    def getSpatialBoundsConnections(self):
        return ("inputVisitSummaries",)

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if self.config.healpix is not None:
            self.dimensions.remove("tract")
            self.dimensions.remove("skymap")
            healpixName = f"healpix{self.config.healpix}"
            self.dimensions.add(healpixName)
            self.outputWcs = dataclasses.replace(
                self.outputWcs, dimensions=("instrument", "visit", healpixName)
            )
            self.outputCatalog = dataclasses.replace(
                self.outputCatalog, dimensions=("instrument", "physical_filter", healpixName)
            )
            self.starCatalog = dataclasses.replace(
                self.starCatalog, dimensions=("instrument", "physical_filter", healpixName)
            )
            self.modelParams = dataclasses.replace(
                self.modelParams, dimensions=("instrument", "physical_filter", healpixName)
            )
            self.outputCameraModel = dataclasses.replace(
                self.outputCameraModel, dimensions=("instrument", "physical_filter", healpixName)
            )
            self.camera = dataclasses.replace(
                self.camera, dimensions=("instrument", "physical_filter", healpixName)
            )
            self.dcrCoefficients = dataclasses.replace(
                self.dcrCoefficients, dimensions=("instrument", "physical_filter", healpixName)
            )

        if not self.config.useColor:
            self.inputs.remove("colorCatalog")
            self.outputs.remove("dcrCoefficients")
        if not self.config.saveModelParams:
            self.outputs.remove("modelParams")
        if not self.config.useInputCameraModel:
            self.prerequisiteInputs.remove("inputCameraModel")
        if not self.config.saveCameraModel:
            self.outputs.remove("outputCameraModel")
        if not self.config.saveCameraObject:
            self.prerequisiteInputs.remove("inputCamera")
            self.outputs.remove("camera")


class GbdesAstrometricFitConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=GbdesAstrometricFitConnections
):
    """Configuration for GbdesAstrometricFitTask"""

    sourceSelector = sourceSelectorRegistry.makeField(
        doc="How to select sources for cross-matching.", default="science"
    )
    referenceSelector = pexConfig.ConfigurableField(
        target=ReferenceSourceSelectorTask,
        doc="How to down-select the loaded astrometry reference catalog.",
    )
    referenceFilter = pexConfig.Field(
        dtype=str,
        doc="Name of filter to load from reference catalog. This is a required argument, although the values"
        "returned are not used.",
        default="phot_g_mean",
    )
    setRefEpoch = pexConfig.Field(
        dtype=float,
        doc="Set the reference epoch to a fixed value in MJD (if None, median observation date is used)",
        default=None,
        optional=True,
    )
    applyRefCatProperMotion = pexConfig.Field(
        dtype=bool,
        doc="Apply proper motion to shift reference catalog to epoch of observations.",
        default=True,
    )
    matchRadius = pexConfig.Field(
        doc="Matching tolerance between associated objects (arcseconds).", dtype=float, default=1.0
    )
    minMatches = pexConfig.Field(
        doc="Number of matches required to keep a source object.", dtype=int, default=2
    )
    allowSelfMatches = pexConfig.Field(
        doc="Allow multiple sources from the same visit to be associated with the same object.",
        dtype=bool,
        default=False,
    )
    sourceFluxType = pexConfig.Field(
        dtype=str,
        doc="Source flux field to use in source selection and to get fluxes from the catalog.",
        default="apFlux_12_0",
    )
    systematicError = pexConfig.Field(
        dtype=float,
        doc=(
            "Systematic error padding added in quadrature for the science catalogs (marcsec). The default"
            "value is equivalent to 0.02 pixels for HSC."
        ),
        default=0.0034,
    )
    referenceSystematicError = pexConfig.Field(
        dtype=float,
        doc="Systematic error padding added in quadrature for the reference catalog (marcsec).",
        default=0.0,
    )
    useColor = pexConfig.Field(
        dtype=bool,
        doc="Use color information to correct for differential chromatic refraction.",
        default=False,
    )
    color = pexConfig.ListField(
        dtype=str,
        doc="The bands to use for calculating color.",
        default=["g", "i"],
        listCheck=(lambda x: (len(x) == 2) and (len(set(x)) == len(x))),
    )
    referenceColor = pexConfig.Field(
        dtype=float,
        doc="The color for which DCR is defined as zero.",
        default=0.61,
    )
    modelComponents = pexConfig.ListField(
        dtype=str,
        doc=(
            "List of mappings to apply to transform from pixels to sky, in order of their application."
            "Supported options are 'INSTRUMENT/DEVICE' and 'EXPOSURE'."
        ),
        default=["INSTRUMENT/DEVICE", "EXPOSURE"],
    )
    deviceModel = pexConfig.ListField(
        dtype=str,
        doc=(
            "List of mappings to apply to transform from detector pixels to intermediate frame. Map names"
            "should match the format 'BAND/DEVICE/<map name>'."
        ),
        default=["BAND/DEVICE/poly"],
    )
    exposureModel = pexConfig.ListField(
        dtype=str,
        doc=(
            "List of mappings to apply to transform from intermediate frame to sky coordinates. Map names"
            "should match the format 'EXPOSURE/<map name>'."
        ),
        default=["EXPOSURE/poly"],
    )
    devicePolyOrder = pexConfig.Field(dtype=int, doc="Order of device polynomial model.", default=4)
    exposurePolyOrder = pexConfig.Field(dtype=int, doc="Order of exposure polynomial model.", default=6)
    fitProperMotion = pexConfig.Field(dtype=bool, doc="Fit the proper motions of the objects.", default=False)
    excludeNonPMObjects = pexConfig.Field(
        dtype=bool, doc="Exclude reference objects without proper motion/parallax information.", default=True
    )
    fitReserveFraction = pexConfig.Field(
        dtype=float, default=0.2, doc="Fraction of objects to reserve from fit for validation."
    )
    fitReserveRandomSeed = pexConfig.Field(
        dtype=int,
        doc="Set the random seed for selecting data points to reserve from the fit for validation.",
        default=1234,
    )
    saveModelParams = pexConfig.Field(
        dtype=bool,
        doc=(
            "Save the parameters and covariance of the WCS model. Default to "
            "false because this can be very large."
        ),
        default=False,
    )
    useInputCameraModel = pexConfig.Field(
        dtype=bool,
        doc=(
            "Use a preexisting model for the 'device' part of the model. When true, the device part of the"
            " model will be held fixed in the fitting process."
        ),
        default=False,
    )
    saveCameraModel = pexConfig.Field(
        dtype=bool,
        doc="Save the 'device' part of the model to be used as input in future runs.",
        default=False,
    )
    buildCamera = pexConfig.ConfigurableField(
        target=BuildCameraFromAstrometryTask, doc="Subtask to build camera from astrometric model."
    )
    saveCameraObject = pexConfig.Field(
        dtype=bool,
        doc="Build and output an lsst.afw.cameraGeom.Camera object using the fit per-detector model.",
        default=False,
    )
    clipThresh = pexConfig.Field(
        dtype=float,
        doc="Threshold for clipping outliers in the fit (in standard deviations)",
        default=5.0,
    )
    clipFraction = pexConfig.Field(
        dtype=float,
        doc="Minimum fraction of clipped sources that triggers a new fit iteration.",
        default=0.0,
    )
    healpix = pexConfig.Field(
        dtype=int,
        doc="Run using all visits overlapping a healpix pixel with this order instead of a tract. Order 3 "
        "corresponds to pixels with angular size of 7.329 degrees.",
        optional=True,
        default=None,
    )
    minDetectorFraction = pexConfig.Field(
        dtype=float,
        doc=(
            "Minimum fraction of detectors with valid WCSs per visit required to include the visit in the "
            "fit."
        ),
        default=0.25,
    )

    def setDefaults(self):
        # Use only stars because aperture fluxes of galaxies are biased and
        # depend on seeing.
        self.sourceSelector["science"].doUnresolved = True
        self.sourceSelector["science"].unresolved.name = "sizeExtendedness"

        # Use only isolated sources.
        self.sourceSelector["science"].doIsolated = True
        self.sourceSelector["science"].isolated.parentName = "parentSourceId"
        self.sourceSelector["science"].isolated.nChildName = "deblend_nChild"
        # Do not use either flux or centroid measurements with flags,
        # chosen from the usual QA flags for stars.
        self.sourceSelector["science"].doFlags = True
        badFlags = [
            "pixelFlags_edge",
            "pixelFlags_saturated",
            "pixelFlags_interpolatedCenter",
            "pixelFlags_interpolated",
            "pixelFlags_crCenter",
            "pixelFlags_bad",
            "hsmPsfMoments_flag",
            f"{self.sourceFluxType}_flag",
        ]
        self.sourceSelector["science"].flags.bad = badFlags

        # Use only primary sources.
        self.sourceSelector["science"].doRequirePrimary = True

        self.sourceSelector["science"].doSignalToNoise = True
        self.sourceSelector["science"].signalToNoise.minimum = 8.0
        self.sourceSelector["science"].signalToNoise.maximum = 1000.0
        self.sourceSelector["science"].signalToNoise.fluxField = self.sourceFluxType + "_instFlux"
        self.sourceSelector["science"].signalToNoise.errField = self.sourceFluxType + "_instFluxErr"

    def validate(self):
        super().validate()

        # Check if all components of the device and exposure models are
        # supported.
        for component in self.deviceModel:
            mapping = component.split("/")[-1]
            if mapping not in ["poly", "identity"]:
                raise pexConfig.FieldValidationError(
                    GbdesAstrometricFitConfig.deviceModel,
                    self,
                    f"deviceModel component {component} is not supported.",
                )

        for component in self.exposureModel:
            mapping = component.split("/")[-1]
            if mapping not in ["poly", "identity", "dcr"]:
                raise pexConfig.FieldValidationError(
                    GbdesAstrometricFitConfig.exposureModel,
                    self,
                    f"exposureModel component {component} is not supported.",
                )

        if self.saveCameraModel and self.useInputCameraModel:
            raise pexConfig.FieldValidationError(
                GbdesAstrometricFitConfig.saveCameraModel,
                self,
                "saveCameraModel and useInputCameraModel cannot both be true.",
            )

        if self.saveCameraObject and (self.exposurePolyOrder != 1):
            raise pexConfig.FieldValidationError(
                GbdesAstrometricFitConfig.saveCameraObject,
                self,
                "If saveCameraObject is True, exposurePolyOrder must be set to 1.",
            )


class GbdesAstrometricFitTask(pipeBase.PipelineTask):
    """Calibrate the WCS across multiple visits of the same field using the
    GBDES package.
    """

    ConfigClass = GbdesAstrometricFitConfig
    _DefaultName = "gbdesAstrometricFit"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sourceSelector")
        self.makeSubtask("referenceSelector")
        if self.config.saveCameraObject:
            self.makeSubtask("buildCamera")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # We override runQuantum to set up the refObjLoaders
        inputs = butlerQC.get(inputRefs)

        instrumentName = butlerQC.quantum.dataId["instrument"]

        # Ensure the inputs are in a consistent and deterministic order
        inputCatVisits = np.array([inputCat.dataId["visit"] for inputCat in inputs["inputCatalogRefs"]])
        inputs["inputCatalogRefs"] = [inputs["inputCatalogRefs"][v] for v in inputCatVisits.argsort()]
        inputSumVisits = np.array([inputSum[0]["visit"] for inputSum in inputs["inputVisitSummaries"]])
        inputs["inputVisitSummaries"] = [inputs["inputVisitSummaries"][v] for v in inputSumVisits.argsort()]
        inputRefHtm7s = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputRefs.referenceCatalog])
        inputRefCatRefs = [inputRefs.referenceCatalog[htm7] for htm7 in inputRefHtm7s.argsort()]
        inputRefCats = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputs["referenceCatalog"]])
        inputs["referenceCatalog"] = [inputs["referenceCatalog"][v] for v in inputRefCats.argsort()]

        refConfig = LoadReferenceObjectsConfig()
        if self.config.applyRefCatProperMotion:
            refConfig.requireProperMotion = True
        refObjectLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefCatRefs],
            refCats=inputs.pop("referenceCatalog"),
            config=refConfig,
            log=self.log,
        )

        nCores = butlerQC.resources.num_cores
        self.log.info("Running with nCores = %d", nCores)

        if self.config.useColor:
            colorCatalog = inputs.pop("colorCatalog")
        else:
            colorCatalog = None
        try:
            output = self.run(
                **inputs,
                instrumentName=instrumentName,
                refObjectLoader=refObjectLoader,
                colorCatalog=colorCatalog,
                nCores=nCores,
            )
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(e, self, log=self.log)
            # No partial outputs for butler to put
            raise error from e

        wcsOutputRefDict = {outWcsRef.dataId["visit"]: outWcsRef for outWcsRef in outputRefs.outputWcs}
        for visit, outputWcs in output.outputWcss.items():
            butlerQC.put(outputWcs, wcsOutputRefDict[visit])
        butlerQC.put(output.outputCatalog, outputRefs.outputCatalog)
        butlerQC.put(output.starCatalog, outputRefs.starCatalog)
        if self.config.saveModelParams:
            butlerQC.put(output.modelParams, outputRefs.modelParams)
        if self.config.saveCameraModel:
            butlerQC.put(output.cameraModelParams, outputRefs.outputCameraModel)
        if self.config.saveCameraObject:
            butlerQC.put(output.camera, outputRefs.camera)
        if self.config.useColor:
            butlerQC.put(output.colorParams, outputRefs.dcrCoefficients)
        if output.partialOutputs:
            e = RuntimeError("Some visits were dropped because data was insufficient to fit model.")
            error = pipeBase.AnnotatedPartialOutputsError.annotate(e, self, log=self.log)
            raise error from e

    def run(
        self,
        inputCatalogRefs,
        inputVisitSummaries,
        instrumentName="",
        refEpoch=None,
        refObjectLoader=None,
        inputCameraModel=None,
        colorCatalog=None,
        inputCamera=None,
        nCores=1,
    ):
        """Run the WCS fit for a given set of visits

        Parameters
        ----------
        inputCatalogRefs : `list` [`DeferredDatasetHandle`]
            List of handles pointing to visit-level source
            tables.
        inputVisitSummaries : `list` [`lsst.afw.table.ExposureCatalog`]
            List of catalogs with per-detector summary information.
        instrumentName : `str`, optional
            Name of the instrument used. This is only used for labelling.
        refEpoch : `float`
            Epoch of the reference objects in MJD.
        refObjectLoader : instance of
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`
            Reference object loader instance.
        inputCameraModel : `dict` [`str`, `np.ndarray`], optional
            Parameters to use for the device part of the model.
        colorCatalog : `lsst.afw.table.SimpleCatalog`
            Catalog containing object coordinates and magnitudes.
        inputCamera : `lsst.afw.cameraGeom.Camera`, optional
            Camera to be used as template when constructing new camera.
        nCores : `int`, optional
            Number of cores to use during WCS fit.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``outputWcss`` : `list` [`lsst.afw.table.ExposureCatalog`]
                List of exposure catalogs (one per visit) with the WCS for each
                detector set by the new fitted WCS.
            ``fitModel`` : `wcsfit.WCSFit`
                Model-fitting object with final model parameters.
            ``outputCatalog`` : `pyarrow.Table`
                Catalog with fit residuals of all sources used.
            ``starCatalog`` : `pyarrow.Table`
                Catalog with best-fit positions of the objects fit.
            ``modelParams`` : `dict`
                Parameters and covariance of the best-fit WCS model.
            ``cameraModelParams`` : `dict` [`str`, `np.ndarray`]
                Parameters of the device part of the model, in the format
                needed as input for future runs.
            ``colorParams`` : `dict` [`int`, `np.ndarray`]
                DCR parameters fit in RA and Dec directions for each visit.
            ``camera`` : `lsst.afw.cameraGeom.Camera`
                Camera object constructed from the per-detector model.
        """
        self.log.info("Gather instrument, exposure, and field info")

        # Get RA, Dec, MJD, etc., for the input visits
        exposureInfo, exposuresHelper, extensionInfo, instruments = self._get_exposure_info(
            inputVisitSummaries
        )

        # Get information about the extent of the input visits
        fields, fieldCenter, fieldRadius = self._prep_sky(inputVisitSummaries, exposureInfo.medianEpoch)
        self.log.info("Field center set at %s with radius %s degrees", fieldCenter, fieldRadius.asDegrees())

        self.log.info("Load catalogs and associate sources")
        # Set up class to associate sources into matches using a
        # friends-of-friends algorithm
        associations = wcsfit.FoFClass(
            fields,
            instruments,
            exposuresHelper,
            [fieldRadius.asDegrees()],
            (self.config.matchRadius * u.arcsec).to(u.degree).value,
        )

        # Add the reference catalog to the associator
        medianEpoch = astropy.time.Time(exposureInfo.medianEpoch, format="jyear").mjd
        refObjects, refCovariance = self._load_refcat(
            refObjectLoader,
            extensionInfo,
            epoch=medianEpoch,
            center=fieldCenter,
            radius=fieldRadius,
            associations=associations,
        )

        # Add the science catalogs and associate new sources as they are added
        sourceIndices, usedColumns = self._load_catalogs_and_associate(
            associations, inputCatalogRefs, extensionInfo
        )
        self._check_degeneracies(associations, extensionInfo)

        self.log.info("Fit the WCSs")
        # Set up a YAML-type string using the config variables and a sample
        # visit
        inputYaml, mapTemplate = self.make_yaml(
            inputVisitSummaries[0],
            inputCameraModel=(inputCameraModel if self.config.useInputCameraModel else None),
        )

        # Set the verbosity level for WCSFit from the task log level.
        # TODO: DM-36850, Add lsst.log to gbdes so that log messages are
        # properly propagated.
        loglevel = self.log.getEffectiveLevel()
        if loglevel >= self.log.WARNING:
            verbose = 0
        elif loglevel == self.log.INFO:
            verbose = 1
        else:
            verbose = 2

        # Set up the WCS-fitting class using the results of the FOF associator
        if self.config.useInputCameraModel:
            fixMaps = ",".join(inputCameraModel.keys())
        else:
            fixMaps = ""
        wcsf = wcsfit.WCSFit(
            fields,
            instruments,
            exposuresHelper,
            extensionInfo.visitIndex,
            extensionInfo.detectorIndex,
            inputYaml,
            extensionInfo.wcs,
            associations.sequence,
            associations.extn,
            associations.obj,
            sysErr=self.config.systematicError,
            refSysErr=self.config.referenceSystematicError,
            usePM=self.config.fitProperMotion,
            verbose=verbose,
            fixMaps=fixMaps,
            num_threads=nCores,
        )
        # Add the science and reference sources
        self._add_objects(wcsf, inputCatalogRefs, sourceIndices, extensionInfo, usedColumns)
        self._add_ref_objects(wcsf, refObjects, refCovariance, extensionInfo)
        if self.config.useColor:
            self._add_color_objects(wcsf, colorCatalog)

        # There must be at least as many sources per visit as the number of
        # free parameters in the per-visit mapping. Set minFitExposures to be
        # the number of free parameters divided by the fraction of non-reserved
        # sources, so that visits with fewer sources are dropped.
        nCoeffVisitModel = _nCoeffsFromDegree(self.config.exposurePolyOrder)
        minFitExposures = int(np.ceil(nCoeffVisitModel / (1 - self.config.fitReserveFraction)))
        # Do the WCS fit
        try:
            wcsf.fit(
                reserveFraction=self.config.fitReserveFraction,
                randomNumberSeed=self.config.fitReserveRandomSeed,
                minFitExposures=minFitExposures,
                clipThresh=self.config.clipThresh,
                clipFraction=self.config.clipFraction,
            )
        except RuntimeError as e:
            if "Cholesky decomposition failed" in str(e):
                raise CholeskyError() from e
            else:
                raise

        self.log.info("WCS fitting done")

        outputWcss, cameraParams, colorParams, camera, partialOutputs = self._make_outputs(
            wcsf,
            inputVisitSummaries,
            exposureInfo,
            mapTemplate,
            extensionInfo,
            inputCameraModel=(inputCameraModel if self.config.useInputCameraModel else None),
            inputCamera=(inputCamera if self.config.buildCamera else None),
        )
        outputCatalog = wcsf.getOutputCatalog()
        outputCatalog["exposureName"] = np.array(outputCatalog["exposureName"])
        outputCatalog["deviceName"] = np.array(outputCatalog["deviceName"])

        starCatalog = wcsf.getStarCatalog()
        modelParams = self._compute_model_params(wcsf) if self.config.saveModelParams else None

        return pipeBase.Struct(
            outputWcss=outputWcss,
            fitModel=wcsf,
            outputCatalog=outputCatalog,
            starCatalog=starCatalog,
            modelParams=modelParams,
            cameraModelParams=cameraParams,
            colorParams=colorParams,
            camera=camera,
            partialOutputs=partialOutputs,
        )

    def _prep_sky(self, inputVisitSummaries, epoch, fieldName="Field"):
        """Get center and radius of the input tract. This assumes that all
        visits will be put into the same `wcsfit.Field` and fit together.

        Paramaters
        ----------
        inputVisitSummaries : `list` [`lsst.afw.table.ExposureCatalog`]
            List of catalogs with per-detector summary information.
        epoch : float
            Reference epoch.
        fieldName : str
            Name of the field, used internally.

        Returns
        -------
        fields : `wcsfit.Fields`
            Object with field information.
        center : `lsst.geom.SpherePoint`
            Center of the field.
        radius : `lsst.sphgeom._sphgeom.Angle`
            Radius of the bounding circle of the tract.
        """
        allDetectorCorners = []
        for visSum in inputVisitSummaries:
            detectorCorners = [
                lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees).getVector()
                for (ra, dec) in zip(visSum["raCorners"].ravel(), visSum["decCorners"].ravel())
                if (np.isfinite(ra) and (np.isfinite(dec)))
            ]
            allDetectorCorners.extend(detectorCorners)
        boundingCircle = lsst.sphgeom.ConvexPolygon.convexHull(allDetectorCorners).getBoundingCircle()
        center = lsst.geom.SpherePoint(boundingCircle.getCenter())
        ra = center.getRa().asDegrees()
        dec = center.getDec().asDegrees()
        radius = boundingCircle.getOpeningAngle()

        # wcsfit.Fields describes a list of fields, but we assume all
        # observations will be fit together in one field.
        fields = wcsfit.Fields([fieldName], [ra], [dec], [epoch])

        return fields, center, radius

    def _get_exposure_info(
        self,
        inputVisitSummaries,
        fieldNumber=0,
        refEpoch=None,
        fieldRegions=None,
    ):
        """Get various information about the input visits to feed to the
        fitting routines.

        Parameters
        ----------
        inputVisitSummaries : `list [`lsst.afw.table.ExposureCatalog`]
            Tables for each visit with information for detectors.
        fieldNumber : `int`, optional
            Index of the field for these visits. Should be zero if all data is
            being fit together. This is ignored if `fieldRegions` is not None.
        refEpoch : `float`, optional
            Epoch of the reference objects in MJD.
        fieldRegions : `dict` [`int`, `lsst.sphgeom.ConvexPolygon`], optional
            Dictionary of regions encompassing each group of input visits
            keyed by an arbitrary index.

        Returns
        -------
        exposureInfo : `lsst.pipe.base.Struct`
            Struct containing general properties for the visits:
            ``visits`` : `list`
                List of visit names.
            ``detectors`` : `list`
                List of all detectors in any visit.
            ``ras`` : `list` [`float`]
                List of boresight RAs for each visit.
            ``decs`` : `list` [`float`]
                List of borseight Decs for each visit.
            ``medianEpoch`` : float
                Median epoch of all visits in decimal-year format.
        exposuresHelper : `wcsfit.ExposuresHelper`
            Object containing information about the input visits.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector):
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        instruments : `list` [`wcsfit.Instrument`]
            List of instrument objects used for the input visits.
        """
        exposureNames = []
        ras = []
        decs = []
        visits = []
        detectors = []
        airmasses = []
        exposureTimes = []
        mjds = []
        observatories = []
        wcss = []
        fieldNumbers = []

        extensionType = []
        extensionVisitIndices = []
        extensionDetectorIndices = []
        extensionVisits = []
        extensionDetectors = []

        instruments, instrumentNumbers = _get_instruments(inputVisitSummaries)

        # Get information for all the science visits
        for v, visitSummary in enumerate(inputVisitSummaries):
            visitInfo = visitSummary[0].getVisitInfo()
            visit = visitSummary[0]["visit"]
            visits.append(visit)
            exposureNames.append(str(visit))
            raDec = visitInfo.getBoresightRaDec()
            ras.append(raDec.getRa().asRadians())
            decs.append(raDec.getDec().asRadians())
            if fieldRegions is not None:
                inField = [r for r, region in fieldRegions.items() if region.contains(raDec.getVector())]
                if len(inField) != 1:
                    raise RuntimeError(
                        f"Visit should be in one and only one field, but {visit} is contained "
                        f"in {len(inField)} fields."
                    )
                fieldNumbers.append(inField[0])
            else:
                fieldNumbers.append(fieldNumber)
            airmasses.append(visitInfo.getBoresightAirmass())
            exposureTimes.append(visitInfo.getExposureTime())
            obsDate = visitInfo.getDate()
            obsMJD = obsDate.get(obsDate.MJD)
            mjds.append(obsMJD)
            # Get the observatory ICRS position for use in fitting parallax
            obsLon = visitInfo.observatory.getLongitude().asDegrees()
            obsLat = visitInfo.observatory.getLatitude().asDegrees()
            obsElev = visitInfo.observatory.getElevation()
            earthLocation = astropy.coordinates.EarthLocation.from_geodetic(obsLon, obsLat, obsElev)
            observatory_gcrs = earthLocation.get_gcrs(astropy.time.Time(obsMJD, format="mjd"))
            observatory_icrs = observatory_gcrs.transform_to(astropy.coordinates.ICRS())
            # We want the position in AU in Cartesian coordinates
            observatories.append(observatory_icrs.cartesian.xyz.to(u.AU).value)

            validDetectors = [row for row in visitSummary if row.wcs is not None]
            nDetectorFraction = len(validDetectors) / len(visitSummary)
            if nDetectorFraction < self.config.minDetectorFraction:
                self.log.warning(
                    "Visit %d has only %d detectors with valid WCSs (%s of total) and will be dropped.",
                    visit,
                    len(validDetectors),
                    nDetectorFraction,
                )
                continue

            for row in visitSummary:
                detector = row["id"]

                wcs = row.getWcs()
                if wcs is None:
                    self.log.warning(
                        "WCS is None for visit %d, detector %d: this extension (visit/detector) will be "
                        "dropped.",
                        visit,
                        detector,
                    )
                    continue
                else:
                    wcsRA = wcs.getSkyOrigin().getRa().asRadians()
                    wcsDec = wcs.getSkyOrigin().getDec().asRadians()
                    tangentPoint = wcsfit.Gnomonic(wcsRA, wcsDec)
                    mapping = wcs.getFrameDict().getMapping("PIXELS", "IWC")
                    gbdes_wcs = wcsfit.Wcs(wcsfit.ASTMap(mapping), tangentPoint)
                    wcss.append(gbdes_wcs)

                if detector not in detectors:
                    detectors.append(detector)
                if not instruments[instrumentNumbers[v]].hasDevice(str(detector)):
                    detectorBounds = wcsfit.Bounds(
                        row["bbox_min_x"], row["bbox_max_x"], row["bbox_min_y"], row["bbox_max_y"]
                    )
                    instruments[instrumentNumbers[v]].addDevice(str(detector), detectorBounds)

                detectorIndex = np.flatnonzero(detector == np.array(detectors))[0]
                extensionVisitIndices.append(v)
                extensionDetectorIndices.append(detectorIndex)
                extensionVisits.append(visit)
                extensionDetectors.append(detector)
                extensionType.append("SCIENCE")

        if len(wcss) == 0:
            raise pipeBase.NoWorkFound("No input extensions have valid WCSs.")

        # Set the reference epoch to be the median of the science visits.
        # The reference catalog will be shifted to this date.
        if self.config.setRefEpoch is None:
            medianMJD = np.median(mjds)
            self.log.info(f"Ref epoch set to median: {medianMJD}")
        else:
            medianMJD = self.config.setRefEpoch
            self.log.info(f"Ref epoch set by user: {medianMJD}")
        medianEpoch = astropy.time.Time(medianMJD, format="mjd").jyear

        # Add information for the reference catalog. Most of the values are
        # not used. There needs to be a separate catalog for each field.
        if fieldRegions is None:
            fieldRegions = {0: None}
        for f in fieldRegions:
            exposureNames.append("REFERENCE")
            # Make the "visit" number the field * -1 to disambiguate it from
            # any potential visit number:
            visits.append(-1 * f)
            fieldNumbers.append(f)
            if self.config.fitProperMotion:
                instrumentNumbers.append(-2)
            else:
                instrumentNumbers.append(-1)
            ras.append(0.0)
            decs.append(0.0)
            airmasses.append(0.0)
            exposureTimes.append(0)
            mjds.append((refEpoch if (refEpoch is not None) else medianMJD))
            observatories.append(np.array([0, 0, 0]))
            identity = wcsfit.IdentityMap()
            icrs = wcsfit.SphericalICRS()
            refWcs = wcsfit.Wcs(identity, icrs, "Identity", np.pi / 180.0)
            wcss.append(refWcs)

            extensionVisitIndices.append(len(exposureNames) - 1)
            extensionDetectorIndices.append(-1)  # REFERENCE device must be -1
            extensionVisits.append(-1 * f)
            extensionDetectors.append(-1)
            extensionType.append("REFERENCE")

        # Make a table of information to use elsewhere in the class
        extensionInfo = pipeBase.Struct(
            visit=np.array(extensionVisits),
            detector=np.array(extensionDetectors),
            visitIndex=np.array(extensionVisitIndices),
            detectorIndex=np.array(extensionDetectorIndices),
            wcs=np.array(wcss),
            extensionType=np.array(extensionType),
        )

        # Make the exposureHelper object to use in the fitting routines
        exposuresHelper = wcsfit.ExposuresHelper(
            exposureNames,
            fieldNumbers,
            instrumentNumbers,
            ras,
            decs,
            airmasses,
            exposureTimes,
            mjds,
            observatories,
        )

        exposureInfo = pipeBase.Struct(
            visits=visits, detectors=detectors, ras=ras, decs=decs, medianEpoch=medianEpoch
        )

        return exposureInfo, exposuresHelper, extensionInfo, instruments

    def _load_refcat(
        self,
        refObjectLoader,
        extensionInfo,
        epoch=None,
        fieldIndex=0,
        associations=None,
        center=None,
        radius=None,
        region=None,
    ):
        """Load the reference catalog and add reference objects to the
        `wcsfit.FoFClass` object.

        Parameters
        ----------
        refObjectLoader :
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`
            Object set up to load reference catalog objects.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector).
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        epoch : `float`, optional
            MJD to which to correct the object positions.
        fieldIndex : `int`, optional
            Index of the field. Should be zero if all the data is fit together.
        associations : `wcsfit.FoFClass`, optional
            Object to which to add the catalog of reference objects.
        center : `lsst.geom.SpherePoint`, optional
            Center of the circle in which to load reference objects. Ignored if
            `region` is set. If used, `radius` must also be set.
        radius : `lsst.sphgeom._sphgeom.Angle`, optional
            Radius of the circle in which to load reference objects. Ignored if
            `region` is set. If used, `center` must also be set.
        region : `lsst.sphgeom.ConvexPolygon`, optional
            Region in which to load reference objects.

        Returns
        -------
        refObjects : `dict`
            Position and error information of reference objects.
        refCovariance : `list` [`float`]
            Flattened output covariance matrix.
        """
        if self.config.applyRefCatProperMotion:
            formattedEpoch = astropy.time.Time(epoch, format="mjd")
        else:
            formattedEpoch = None

        neededColumns = ["coord_ra", "coord_dec", "coord_raErr", "coord_decErr"]
        if self.config.applyRefCatProperMotion:
            neededColumns += [
                "pm_ra",
                "pm_dec",
                "parallax",
                "pm_raErr",
                "pm_decErr",
                "parallaxErr",
            ]
        # Get refcat version from refcat metadata
        refCatMetadata = refObjectLoader.refCats[0].get().getMetadata()
        # DM-47181: Added this to work for LSSTComCam with
        # the_monster_20240904 catalog that does not have this key.
        if "REFCAT_FORMAT_VERSION" not in refCatMetadata:
            refCatVersion = 2
        else:
            refCatVersion = refCatMetadata["REFCAT_FORMAT_VERSION"]
        if refCatVersion == 2:
            neededColumns += [
                "coord_ra_coord_dec_Cov",
                "coord_ra_pm_ra_Cov",
                "coord_ra_pm_dec_Cov",
                "coord_ra_parallax_Cov",
                "coord_dec_pm_ra_Cov",
                "coord_dec_pm_dec_Cov",
                "coord_dec_parallax_Cov",
                "pm_ra_pm_dec_Cov",
                "pm_ra_parallax_Cov",
                "pm_dec_parallax_Cov",
            ]

        # Load each shard of the reference catalog separately to avoid a spike
        # in the memory load.
        refCatShards = []
        for dataId, cat in zip(refObjectLoader.dataIds, refObjectLoader.refCats):
            miniRefObjectLoader = ReferenceObjectLoader(
                dataIds=[dataId],
                refCats=[cat],
                config=refObjectLoader.config,
                log=self.log,
            )
            try:
                if region is not None:
                    skyRegion = miniRefObjectLoader.loadRegion(
                        region, self.config.referenceFilter, epoch=formattedEpoch
                    )
                elif (center is not None) and (radius is not None):
                    skyRegion = miniRefObjectLoader.loadSkyCircle(
                        center, radius, self.config.referenceFilter, epoch=formattedEpoch
                    )
                else:
                    raise RuntimeError("Either `region` or `center` and `radius` must be set.")
            except RuntimeError:
                self.log.debug("Reference catalog shard has no objects inside the region.")
                continue
            selected = self.referenceSelector.run(skyRegion.refCat)
            # Need memory contiguity to get reference filters as a vector.
            if not selected.sourceCat.isContiguous():
                refCatShard = selected.sourceCat.copy(deep=True)
            else:
                refCatShard = selected.sourceCat
            refCatShard = refCatShard.asAstropy()[neededColumns]

            # In Gaia DR3, missing values are denoted by NaNs.
            finiteInd = np.isfinite(refCatShard["coord_ra"]) & np.isfinite(refCatShard["coord_dec"])
            refCatShard = refCatShard[finiteInd]
            refCatShards.append(refCatShard)
        refCat = vstack(refCatShards)

        if self.config.excludeNonPMObjects and self.config.applyRefCatProperMotion:
            # Gaia DR2 has zeros for missing data, while Gaia DR3 has NaNs:
            hasPM = (
                (refCat["pm_raErr"] != 0) & np.isfinite(refCat["pm_raErr"]) & np.isfinite(refCat["pm_decErr"])
            )
            refCat = refCat[hasPM]

        ra = (refCat["coord_ra"]).to(u.degree).to_value().tolist()
        dec = (refCat["coord_dec"]).to(u.degree).to_value().tolist()
        raCov = ((refCat["coord_raErr"]).to(u.degree).to_value() ** 2).tolist()
        decCov = ((refCat["coord_decErr"]).to(u.degree).to_value() ** 2).tolist()

        if refCatVersion == 2:
            raDecCov = (refCat["coord_ra_coord_dec_Cov"]).to(u.degree**2).to_value().tolist()
        else:
            raDecCov = np.zeros(len(ra))

        refObjects = {"ra": ra, "dec": dec, "raCov": raCov, "decCov": decCov, "raDecCov": raDecCov}
        refCovariance = []

        if self.config.fitProperMotion:
            raPM = (refCat["pm_ra"]).to(u.marcsec).to_value().tolist()
            decPM = (refCat["pm_dec"]).to(u.marcsec).to_value().tolist()
            parallax = (refCat["parallax"]).to(u.marcsec).to_value().tolist()
            cov = _make_ref_covariance_matrix(refCat, version=refCatVersion)
            pmDict = {"raPM": raPM, "decPM": decPM, "parallax": parallax}
            refObjects.update(pmDict)
            refCovariance = cov

        if associations is not None:
            extensionIndex = np.flatnonzero(extensionInfo.extensionType == "REFERENCE")[0]
            visitIndex = extensionInfo.visitIndex[extensionIndex]
            detectorIndex = extensionInfo.detectorIndex[extensionIndex]
            instrumentIndex = -1  # -1 indicates the reference catalog
            refWcs = extensionInfo.wcs[extensionIndex]

            associations.addCatalog(
                refWcs,
                "STELLAR",
                visitIndex,
                fieldIndex,
                instrumentIndex,
                detectorIndex,
                extensionIndex,
                np.ones(len(refCat), dtype=bool),
                ra,
                dec,
                np.arange(len(ra)),
            )

        return refObjects, refCovariance

    @staticmethod
    def _find_extension_index(extensionInfo, visit, detector):
        """Find the index for a given extension from its visit and detector
        number.

        If no match is found, None is returned.

        Parameters
        ----------
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension.
        visit : `int`
            Visit number
        detector : `int`
            Detector number

        Returns
        -------
        extensionIndex : `int` or None
            Index of this extension
        """
        findExtension = np.flatnonzero((extensionInfo.visit == visit) & (extensionInfo.detector == detector))
        if len(findExtension) == 0:
            extensionIndex = None
        else:
            extensionIndex = findExtension[0]
        return extensionIndex

    def _load_catalogs_and_associate(
        self, associations, inputCatalogRefs, extensionInfo, fieldIndex=0, instrumentIndex=0
    ):
        """Load the science catalogs and add the sources to the associator
        class `wcsfit.FoFClass`, associating them into matches as you go.

        Parameters
        ----------
        associations : `wcsfit.FoFClass`
            Object to which to add the catalog of source and which performs
            the source association.
        inputCatalogRefs : `list`
            List of DeferredDatasetHandles pointing to visit-level source
            tables.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector).
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        fieldIndex : `int`
            Index of the field for these catalogs. Should be zero assuming all
            data is being fit together.
        instrumentIndex : `int`
            Index of the instrument for these catalogs. Should be zero
            assuming all data comes from the same instrument.

        Returns
        -------
        sourceIndices : `list`
            List of boolean arrays used to select sources.
        columns : `list` [`str`]
            List of columns needed from source tables.
        """
        columns = [
            "detector",
            "sourceId",
            "x",
            "xErr",
            "y",
            "yErr",
            "ixx",
            "iyy",
            "ixy",
            f"{self.config.sourceFluxType}_instFlux",
            f"{self.config.sourceFluxType}_instFluxErr",
        ]
        if self.sourceSelector.config.doFlags:
            columns.extend(self.sourceSelector.config.flags.bad)
        if self.sourceSelector.config.doUnresolved:
            columns.append(self.sourceSelector.config.unresolved.name)
        if self.sourceSelector.config.doIsolated:
            columns.append(self.sourceSelector.config.isolated.parentName)
            columns.append(self.sourceSelector.config.isolated.nChildName)
        if self.sourceSelector.config.doRequirePrimary:
            columns.append(self.sourceSelector.config.requirePrimary.primaryColName)

        sourceIndices = [None] * len(extensionInfo.visit)
        for inputCatalogRef in inputCatalogRefs:
            visit = inputCatalogRef.dataId["visit"]
            inputCatalog = inputCatalogRef.get(parameters={"columns": columns})
            # Get a sorted array of detector names
            detectors = np.unique(inputCatalog["detector"])

            for detector in detectors:
                detectorSources = inputCatalog[inputCatalog["detector"] == detector]
                xCov = detectorSources["xErr"] ** 2
                yCov = detectorSources["yErr"] ** 2
                xyCov = (
                    detectorSources["ixy"] * (xCov + yCov) / (detectorSources["ixx"] + detectorSources["iyy"])
                )
                # Remove sources with bad shape measurements
                goodShapes = xyCov**2 <= (xCov * yCov)
                selected = self.sourceSelector.run(detectorSources)
                goodInds = selected.selected & goodShapes
                isStar = np.ones(goodInds.sum())
                extensionIndex = self._find_extension_index(extensionInfo, visit, detector)
                if extensionIndex is None:
                    # This extension does not have information necessary for
                    # fit. Skip the detections from this detector for this
                    # visit.
                    continue
                detectorIndex = extensionInfo.detectorIndex[extensionIndex]
                visitIndex = extensionInfo.visitIndex[extensionIndex]

                sourceIndices[extensionIndex] = goodInds

                wcs = extensionInfo.wcs[extensionIndex]
                associations.reprojectWCS(wcs, fieldIndex)

                associations.addCatalog(
                    wcs,
                    "STELLAR",
                    visitIndex,
                    fieldIndex,
                    instrumentIndex,
                    detectorIndex,
                    extensionIndex,
                    isStar,
                    detectorSources[goodInds]["x"].to_list(),
                    detectorSources[goodInds]["y"].to_list(),
                    np.arange(goodInds.sum()),
                )

        associations.sortMatches(
            fieldIndex, minMatches=self.config.minMatches, allowSelfMatches=self.config.allowSelfMatches
        )

        return sourceIndices, columns

    def _check_degeneracies(self, associations, extensionInfo):
        """Check that the minimum number of visits and sources needed to
        constrain the model are present.

        This does not guarantee that the Hessian matrix of the chi-square,
        which is used to fit the model, will be positive-definite, but if the
        checks here do not pass, the matrix is certain to not be
        positive-definite and the model cannot be fit.

        Parameters
        ----------
        associations : `wcsfit.FoFClass`
            Object holding the source association information.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector):
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        """
        # As a baseline, need to have more stars per detector than per-detector
        # parameters, and more stars per visit than per-visit parameters.
        whichExtension = np.array(associations.extn)
        whichDetector = np.zeros(len(whichExtension))
        whichVisit = np.zeros(len(whichExtension))

        for extension, (detector, visit) in enumerate(zip(extensionInfo.detector, extensionInfo.visit)):
            ex_ind = whichExtension == extension
            whichDetector[ex_ind] = detector
            whichVisit[ex_ind] = visit

        if (not self.config.useInputCameraModel) and ("BAND/DEVICE/poly" in self.config.deviceModel):
            nCoeffDetectorModel = _nCoeffsFromDegree(self.config.devicePolyOrder)
            unconstrainedDetectors = []
            for detector in np.unique(extensionInfo.detector):
                numSources = (whichDetector == detector).sum()
                if numSources < nCoeffDetectorModel:
                    unconstrainedDetectors.append(str(detector))

            if unconstrainedDetectors:
                raise pipeBase.NoWorkFound(
                    "The model is not constrained. The following detectors do not have enough "
                    f"sources ({nCoeffDetectorModel} required): ",
                    ", ".join(unconstrainedDetectors),
                )

    def make_yaml(self, inputVisitSummary, inputFile=None, inputCameraModel=None):
        """Make a YAML-type object that describes the parameters of the fit
        model.

        Parameters
        ----------
        inputVisitSummary : `lsst.afw.table.ExposureCatalog`
            Catalog with per-detector summary information.
        inputFile : `str`
            Path to a file that contains a basic model.
        inputCameraModel : `dict` [`str`, `np.ndarray`], optional
            Parameters to use for the device part of the model.

        Returns
        -------
        inputYaml : `wcsfit.YAMLCollector`
            YAML object containing the model description.
        inputDict : `dict` [`str`, `str`]
            Dictionary containing the model description.
        """
        if inputFile is not None:
            inputYaml = wcsfit.YAMLCollector(inputFile, "PixelMapCollection")
        else:
            inputYaml = wcsfit.YAMLCollector("", "PixelMapCollection")
        inputDict = {}
        modelComponents = ["BAND/DEVICE", "EXPOSURE"]
        baseMap = {"Type": "Composite", "Elements": modelComponents}
        inputDict["EXPOSURE/DEVICE/base"] = baseMap

        xMin = str(inputVisitSummary["bbox_min_x"].min())
        xMax = str(inputVisitSummary["bbox_max_x"].max())
        yMin = str(inputVisitSummary["bbox_min_y"].min())
        yMax = str(inputVisitSummary["bbox_max_y"].max())

        deviceModel = {"Type": "Composite", "Elements": self.config.deviceModel.list()}
        inputDict["BAND/DEVICE"] = deviceModel
        for component in self.config.deviceModel:
            mapping = component.split("/")[-1]
            if mapping == "poly":
                componentDict = {
                    "Type": "Poly",
                    "XPoly": {"OrderX": self.config.devicePolyOrder, "SumOrder": True},
                    "YPoly": {"OrderX": self.config.devicePolyOrder, "SumOrder": True},
                    "XMin": xMin,
                    "XMax": xMax,
                    "YMin": yMin,
                    "YMax": yMax,
                }
            elif mapping == "identity":
                componentDict = {"Type": "Identity"}

            inputDict[component] = componentDict

        if (inputCameraModel is not None) and self.config.useInputCameraModel:
            # This assumes that the input camera model is a 'poly' model
            nCoeffs = _nCoeffsFromDegree(self.config.devicePolyOrder)
            for key, coeffs in inputCameraModel.items():
                if len(coeffs) != nCoeffs * 2:
                    raise RuntimeError(
                        "Input camera model polynomial order does not match the devicePolyOrder"
                    )
                mapDict = {
                    "Type": "Poly",
                    "XPoly": {
                        "OrderX": self.config.devicePolyOrder,
                        "SumOrder": True,
                        "Coefficients": coeffs[:nCoeffs].tolist(),
                    },
                    "YPoly": {
                        "OrderX": self.config.devicePolyOrder,
                        "SumOrder": True,
                        "Coefficients": coeffs[nCoeffs:].tolist(),
                    },
                    "XMin": xMin,
                    "XMax": xMax,
                    "YMin": yMin,
                    "YMax": yMax,
                }
                inputDict[key] = mapDict

        exposureModelComponents = self.config.exposureModel.list()
        if self.config.useColor:
            exposureModelComponents.append("EXPOSURE/dcr")
        exposureModel = {"Type": "Composite", "Elements": exposureModelComponents}
        inputDict["EXPOSURE"] = exposureModel
        for component in exposureModelComponents:
            mapping = component.split("/")[-1]
            if mapping == "poly":
                componentDict = {
                    "Type": "Poly",
                    "XPoly": {"OrderX": self.config.exposurePolyOrder, "SumOrder": "true"},
                    "YPoly": {"OrderX": self.config.exposurePolyOrder, "SumOrder": "true"},
                }
            elif mapping == "identity":
                componentDict = {"Type": "Identity"}
            elif mapping == "dcr":
                componentDict = {
                    "Type": "Color",
                    "Reference": self.config.referenceColor,
                    "Function": {"Type": "Constant"},
                }

            inputDict[component] = componentDict

        inputYaml.addInput(yaml.dump(inputDict))
        inputYaml.addInput("Identity:\n  Type:  Identity\n")

        return inputYaml, inputDict

    def _add_objects(self, wcsf, inputCatalogRefs, sourceIndices, extensionInfo, columns):
        """Add science sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object.
        inputCatalogRefs : `list`
            List of DeferredDatasetHandles pointing to visit-level source
            tables.
        sourceIndices : `list`
            List of boolean arrays used to select sources.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector):
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        columns : `list` [`str`]
            List of columns needed from source tables.
        """
        for inputCatalogRef in inputCatalogRefs:
            visit = inputCatalogRef.dataId["visit"]
            inputCatalog = inputCatalogRef.get(parameters={"columns": columns})
            detectors = np.unique(inputCatalog["detector"])

            for detector in detectors:
                detectorSources = inputCatalog[inputCatalog["detector"] == detector]

                extensionIndex = self._find_extension_index(extensionInfo, visit, detector)
                if extensionIndex is None:
                    # This extension does not have information necessary for
                    # fit. Skip the detections from this detector for this
                    # visit.
                    continue

                sourceCat = detectorSources[sourceIndices[extensionIndex]]

                xCov = sourceCat["xErr"] ** 2
                yCov = sourceCat["yErr"] ** 2
                xyCov = sourceCat["ixy"] * (xCov + yCov) / (sourceCat["ixx"] + sourceCat["iyy"])
                # TODO: add correct xyErr if DM-7101 is ever done.

                d = {
                    "x": sourceCat["x"].to_numpy(),
                    "y": sourceCat["y"].to_numpy(),
                    "xCov": xCov.to_numpy(),
                    "yCov": yCov.to_numpy(),
                    "xyCov": xyCov.to_numpy(),
                }

                wcsf.setObjects(
                    extensionIndex,
                    d,
                    "x",
                    "y",
                    ["xCov", "yCov", "xyCov"],
                    defaultColor=self.config.referenceColor,
                )

    def _add_ref_objects(self, wcsf, refObjects, refCovariance, extensionInfo, fieldIndex=0):
        """Add reference sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object.
        refObjects : `dict`
            Position and error information of reference objects.
        refCovariance : `list` [`float`]
            Flattened output covariance matrix.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector):
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        fieldIndex : `int`, optional
            Index of the field to which these sources correspond.
        """
        extensionIndex = np.flatnonzero(
            (extensionInfo.extensionType == "REFERENCE") & (extensionInfo.visit == fieldIndex)
        )[0]
        if self.config.fitProperMotion:
            wcsf.setObjects(
                extensionIndex,
                refObjects,
                "ra",
                "dec",
                ["raCov", "decCov", "raDecCov"],
                pmDecKey="decPM",
                pmRaKey="raPM",
                parallaxKey="parallax",
                pmCovKey="fullCov",
                pmCov=refCovariance,
            )
        else:
            wcsf.setObjects(extensionIndex, refObjects, "ra", "dec", ["raCov", "decCov", "raDecCov"])

    def _add_color_objects(self, wcsf, colorCatalog):
        """Associate input matches with objects in color catalog and set their
        color value.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCSFit object, assumed to have fit model.
        colorCatalog : `lsst.afw.table.SimpleCatalog`
            Catalog containing object coordinates and magnitudes.
        """

        # Get current best position for matches
        starCat = wcsf.getStarCatalog()

        # TODO: DM-45650, update how the colors are read in here.
        catalogBands = colorCatalog.metadata.getArray("BANDS")
        colorInd1 = catalogBands.index(self.config.color[0])
        colorInd2 = catalogBands.index(self.config.color[1])
        colors = colorCatalog["mag_std_noabs"][:, colorInd1] - colorCatalog["mag_std_noabs"][:, colorInd2]
        goodInd = (colorCatalog["mag_std_noabs"][:, colorInd1] != 99.0) & (
            colorCatalog["mag_std_noabs"][:, colorInd2] != 99.0
        )

        with Matcher(np.array(starCat["starX"]), np.array(starCat["starY"])) as matcher:
            idx, idx_starCat, idx_colorCat, d = matcher.query_radius(
                (colorCatalog[goodInd]["coord_ra"] * u.radian).to(u.degree).value,
                (colorCatalog[goodInd]["coord_dec"] * u.radian).to(u.degree).value,
                self.config.matchRadius / 3600.0,
                return_indices=True,
            )

        matchesWithColor = starCat["starMatchID"][idx_starCat]
        matchColors = np.ones(len(matchesWithColor)) * self.config.referenceColor
        matchColors = colors[goodInd][idx_colorCat]
        wcsf.setColors(matchesWithColor, matchColors)

    def _make_afw_wcs(self, mapDict, centerRA, centerDec, doNormalizePixels=False, xScale=1, yScale=1):
        """Make an `lsst.afw.geom.SkyWcs` from a dictionary of mappings.

        Parameters
        ----------
        mapDict : `dict`
            Dictionary of mapping parameters.
        centerRA : `lsst.geom.Angle`
            RA of the tangent point.
        centerDec : `lsst.geom.Angle`
            Declination of the tangent point.
        doNormalizePixels : `bool`
            Whether to normalize pixels so that range is [-1,1].
        xScale : `float`
            Factor by which to normalize x-dimension. Corresponds to width of
            detector.
        yScale : `float`
            Factor by which to normalize y-dimension. Corresponds to height of
            detector.

        Returns
        -------
        outWCS : `lsst.afw.geom.SkyWcs`
            WCS constructed from the input mappings
        """
        # Set up pixel frames
        pixelFrame = astshim.Frame(2, "Domain=PIXELS")
        normedPixelFrame = astshim.Frame(2, "Domain=NORMEDPIXELS")

        if doNormalizePixels:
            # Pixels will need to be rescaled before going into the mappings
            normCoefficients = [-1.0, 2.0 / xScale, 0, -1.0, 0, 2.0 / yScale]
            normMap = _convert_to_ast_polymap_coefficients(normCoefficients)
        else:
            normMap = astshim.UnitMap(2)

        # All of the detectors for one visit map to the same tangent plane
        tangentPoint = lsst.geom.SpherePoint(centerRA, centerDec)
        cdMatrix = afwgeom.makeCdMatrix(1.0 * lsst.geom.degrees, 0 * lsst.geom.degrees, True)
        iwcToSkyWcs = afwgeom.makeSkyWcs(lsst.geom.Point2D(0, 0), tangentPoint, cdMatrix)
        iwcToSkyMap = iwcToSkyWcs.getFrameDict().getMapping("PIXELS", "SKY")
        skyFrame = iwcToSkyWcs.getFrameDict().getFrame("SKY")

        frameDict = astshim.FrameDict(pixelFrame)
        frameDict.addFrame("PIXELS", normMap, normedPixelFrame)

        currentFrameName = "NORMEDPIXELS"

        # Dictionary values are ordered according to the maps' application.
        for m, mapElement in enumerate(mapDict.values()):
            mapType = mapElement["Type"]

            if mapType == "Poly":
                mapCoefficients = mapElement["Coefficients"]
                astMap = _convert_to_ast_polymap_coefficients(mapCoefficients)
            elif mapType == "Identity":
                astMap = astshim.UnitMap(2)
            else:
                raise ValueError(f"Converting map type {mapType} to WCS is not supported")

            if m == len(mapDict) - 1:
                newFrameName = "IWC"
            else:
                newFrameName = "INTERMEDIATE" + str(m)
            newFrame = astshim.Frame(2, f"Domain={newFrameName}")
            frameDict.addFrame(currentFrameName, astMap, newFrame)
            currentFrameName = newFrameName
        frameDict.addFrame("IWC", iwcToSkyMap, skyFrame)

        outWCS = afwgeom.SkyWcs(frameDict)
        return outWCS

    def _make_outputs(
        self,
        wcsf,
        visitSummaryTables,
        exposureInfo,
        mapTemplate,
        extensionInfo,
        inputCameraModel=None,
        inputCamera=None,
    ):
        """Make a WCS object out of the WCS models.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCSFit object, assumed to have fit model.
        visitSummaryTables : `list` [`lsst.afw.table.ExposureCatalog`]
            Catalogs with per-detector summary information from which to grab
            detector information.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector):
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        mapTemplate : `dict` [`str`, `str`]
            Dictionary containing the model description.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector).
        inputCameraModel : `dict` [`str`, `np.ndarray`], optional
            Parameters to use for the device part of the model. This must be
            provided if an input camera model was used.

        Returns
        -------
        catalogs : `dict` [`str`, `lsst.afw.table.ExposureCatalog`]
            Dictionary of `lsst.afw.table.ExposureCatalog` objects with the WCS
            set to the WCS fit in wcsf, keyed by the visit name.
        cameraParams : `dict` [`str`, `np.ndarray`], optional
            Parameters for the device part of the model in the format needed
            when used as input for future runs.
        colorFits : `dict` [`int`, `np.ndarray`], optional
            DCR parameters fit in RA and Dec directions for each visit.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            Camera object constructed from the per-detector model.
        """
        # Get the parameters of the fit models
        mapParams = wcsf.mapCollection.getParamDict()
        cameraParams = {}
        if self.config.saveCameraModel:
            for element in mapTemplate["BAND/DEVICE"]["Elements"]:
                for detector in exposureInfo.detectors:
                    detectorTemplate = element.replace("DEVICE", str(detector))
                    detectorTemplate = detectorTemplate.replace("BAND", ".+")
                    for k, params in mapParams.items():
                        if re.fullmatch(detectorTemplate, k):
                            cameraParams[k] = params
        if self.config.saveCameraObject:
            # Get the average rotation angle of the input visits.
            rotations = [
                visTable[0].visitInfo.boresightRotAngle.asRadians() for visTable in visitSummaryTables
            ]
            rotationAngle = np.mean(rotations)
            if inputCamera is None:
                raise RuntimeError(
                    "inputCamera must be provided to _make_outputs in order to build output camera."
                )
            camera = self.buildCamera.run(
                mapParams,
                mapTemplate,
                exposureInfo.detectors,
                exposureInfo.visits,
                inputCamera,
                rotationAngle,
            )
        else:
            camera = None
        if self.config.useInputCameraModel:
            if inputCameraModel is None:
                raise RuntimeError(
                    "inputCameraModel must be provided to _make_outputs in order to build output WCS."
                )
            mapParams.update(inputCameraModel)

        # Set up the schema for the output catalogs
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        schema.addField("visit", type="L", doc="Visit number")
        schema.addField(
            "recoveredWcs",
            type="Flag",
            doc="Input WCS missing, output recovered from other input visit/detectors.",
        )

        # Pixels will need to be rescaled before going into the mappings
        xscale = int(mapTemplate["BAND/DEVICE/poly"]["XMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["XMin"])
        yscale = int(mapTemplate["BAND/DEVICE/poly"]["YMax"]) - int(mapTemplate["BAND/DEVICE/poly"]["YMin"])

        # Make dictionary of bboxes for each detector.
        detectorBBoxes = {}
        for detector in exposureInfo.detectors:
            for visitSummary in visitSummaryTables:
                if detInfo := visitSummary.find(detector):
                    detectorBBoxes[detector] = detInfo.getBBox()
                    break

        catalogs = {}
        colorFits = {}
        partialOutputs = False
        for v, visitSummary in enumerate(visitSummaryTables):
            visit = visitSummary[0]["visit"]
            if visit not in extensionInfo.visit:
                self.log.warning("Visit %d was dropped because no detectors had valid WCSs.", visit)
                partialOutputs = True
                continue

            visitMaps = wcsf.mapCollection.orderAtoms(f"{visit}")
            if self.config.useColor:
                colorMap = visitMaps.pop(visitMaps.index(f"{visit}/dcr"))
                colorFits[visit] = mapParams[colorMap]
            visitMap = visitMaps[0]
            visitMapType = wcsf.mapCollection.getMapType(visitMap)
            if (visitMap not in mapParams) and (visitMapType != "Identity"):
                self.log.warning("Visit %d was dropped because of an insufficient amount of data.", visit)
                partialOutputs = True
                continue

            catalog = lsst.afw.table.ExposureCatalog(schema)
            catalog.resize(len(exposureInfo.detectors))
            catalog["visit"] = visit

            for d, detector in enumerate(exposureInfo.detectors):
                mapName = f"{visit}/{detector}"
                if mapName in wcsf.mapCollection.allMapNames():
                    mapElements = wcsf.mapCollection.orderAtoms(f"{mapName}/base")
                    catalog[d]["recoveredWcs"] = False
                else:
                    # This extension was not fit, but the WCS can be recovered
                    # using the maps fit from sources on other visits but the
                    # same detector and from sources on other detectors from
                    # this visit.
                    genericElements = mapTemplate["EXPOSURE/DEVICE/base"]["Elements"]
                    mapElements = []
                    band = visitSummary[0]["physical_filter"]

                    # Go through the generic map components to build the names
                    # of the specific maps for this extension.
                    for component in genericElements:
                        elements = mapTemplate[component]["Elements"]
                        for element in elements:
                            # TODO: DM-42519, gbdes sets the "BAND" to the
                            # instrument name currently. This will need to be
                            # disambiguated if we run on multiple bands at
                            # once.
                            element = element.replace("BAND", str(band))
                            element = element.replace("EXPOSURE", str(visit))
                            element = element.replace("DEVICE", str(detector))
                            mapElements.append(element)
                    catalog[d]["recoveredWcs"] = True
                mapDict = {}
                for m, mapElement in enumerate(mapElements):
                    mapType = wcsf.mapCollection.getMapType(mapElement)
                    if mapType == "Color":
                        # DCR fit should not go into the generic WCS.
                        continue
                    mapDict[mapElement] = {"Type": mapType}

                    if mapType == "Poly":
                        mapCoefficients = mapParams[mapElement]
                        mapDict[mapElement]["Coefficients"] = mapCoefficients

                # The RA and Dec of the visit are needed for the last step of
                # the mapping from the visit tangent plane to RA and Dec
                outWCS = self._make_afw_wcs(
                    mapDict,
                    exposureInfo.ras[v] * lsst.geom.radians,
                    exposureInfo.decs[v] * lsst.geom.radians,
                    doNormalizePixels=True,
                    xScale=xscale,
                    yScale=yscale,
                )

                catalog[d].setId(detector)
                catalog[d].setBBox(detectorBBoxes[detector])
                catalog[d].setWcs(outWCS)
            catalog.sort()
            catalogs[visit] = catalog
        if self.config.useColor:
            colorVisits = np.array(list(colorFits.keys()))
            colorRA = np.array([colorFits[vis][0] for vis in colorVisits])
            colorDec = np.array([colorFits[vis][1] for vis in colorVisits])
            colorFits = {"visit": colorVisits, "raCoefficient": colorRA, "decCoefficient": colorDec}

        return catalogs, cameraParams, colorFits, camera, partialOutputs

    def _compute_model_params(self, wcsf):
        """Get the WCS model parameters and covariance and convert to a
        dictionary that will be readable as a pandas dataframe or other table.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCSFit object, assumed to have fit model.

        Returns
        -------
        modelParams : `dict`
            Parameters and covariance of the best-fit WCS model.
        """
        modelParamDict = wcsf.mapCollection.getParamDict()
        modelCovariance = wcsf.getModelCovariance()

        modelParams = {k: [] for k in ["mapName", "coordinate", "parameter", "coefficientNumber"]}
        i = 0
        for mapName, params in modelParamDict.items():
            nCoeffs = len(params)
            # There are an equal number of x and y coordinate parameters
            nCoordCoeffs = nCoeffs // 2
            modelParams["mapName"].extend([mapName] * nCoeffs)
            modelParams["coordinate"].extend(["x"] * nCoordCoeffs)
            modelParams["coordinate"].extend(["y"] * nCoordCoeffs)
            modelParams["parameter"].extend(params)
            modelParams["coefficientNumber"].extend(np.arange(nCoordCoeffs))
            modelParams["coefficientNumber"].extend(np.arange(nCoordCoeffs))

            for p in range(nCoeffs):
                if p < nCoordCoeffs:
                    coord = "x"
                else:
                    coord = "y"
                modelParams[f"{mapName}_{coord}_{p}_cov"] = modelCovariance[i]
                i += 1

        # Convert the dictionary values from lists to numpy arrays.
        for key, value in modelParams.items():
            modelParams[key] = np.array(value)

        return modelParams


class GbdesAstrometricMultibandFitConnections(
    GbdesAstrometricFitConnections, dimensions=("skymap", "tract", "instrument")
):
    outputCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="gbdesAstrometricMultibandFit_fitStars",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract"),
    )
    starCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of best-fit object positions. Also includes the fit proper motion and parallax if "
            "fitProperMotion is True."
        ),
        name="gbdesAstrometricMultibandFit_starCatalog",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract"),
    )
    modelParams = pipeBase.connectionTypes.Output(
        doc="WCS parameters and covariance.",
        name="gbdesAstrometricMultibandFit_modelParams",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract"),
    )


class GbdesAstrometricMultibandFitConfig(
    GbdesAstrometricFitConfig, pipelineConnections=GbdesAstrometricMultibandFitConnections
):
    pass


class GbdesAstrometricMultibandFitTask(GbdesAstrometricFitTask):
    """Calibrate the WCS across multiple visits in multiple filters of the same
    field using the GBDES package.
    """

    ConfigClass = GbdesAstrometricMultibandFitConfig
    _DefaultName = "gbdesAstrometricMultibandFit"


class GbdesGlobalAstrometricFitConnections(
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "physical_filter")
):
    inputVisitSummaries = pipeBase.connectionTypes.Input(
        doc=(
            "Per-visit consolidated exposure metadata built from calexps. "
            "These catalogs use detector id for the id and must be sorted for "
            "fast lookups of a detector."
        ),
        name="visitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    referenceCatalog = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The astrometry reference catalog to match to loaded input catalog sources.",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )
    colorCatalog = pipeBase.connectionTypes.Input(
        doc="The catalog of magnitudes to match to input sources.",
        name="fgcm_Cycle4_StandardStars",
        storageClass="SimpleCatalog",
        dimensions=("instrument",),
    )
    isolatedStarSources = pipeBase.connectionTypes.Input(
        doc="Catalog of matched sources.",
        name="isolated_star_presources",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
        multiple=True,
        deferLoad=True,
    )
    isolatedStarCatalogs = pipeBase.connectionTypes.Input(
        doc="Catalog of objects corresponding to the isolatedStarSources.",
        name="isolated_star_presource_associations",
        storageClass="DataFrame",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
        multiple=True,
        deferLoad=True,
    )
    inputCameraModel = pipeBase.connectionTypes.PrerequisiteInput(
        doc="Camera parameters to use for 'device' part of model",
        name="gbdesAstrometricFit_cameraModel",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    inputCamera = pipeBase.connectionTypes.PrerequisiteInput(
        doc="Input camera to use when constructing camera from astrometric model.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    outputWcs = pipeBase.connectionTypes.Output(
        doc=(
            "Per-visit world coordinate systems derived from the fitted model. These catalogs only contain "
            "entries for detectors with an output, and use the detector id for the catalog id, sorted on id "
            "for fast lookups of a detector."
        ),
        name="gbdesGlobalAstrometricFitSkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    outputCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="gbdesGlobalAstrometricFit_fitStars",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    starCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of best-fit object positions. Also includes the fit proper motion and parallax if "
            "fitProperMotion is True."
        ),
        name="gbdesGlobalAstrometricFit_starCatalog",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    modelParams = pipeBase.connectionTypes.Output(
        doc="WCS parameters and covariance.",
        name="gbdesGlobalAstrometricFit_modelParams",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    outputCameraModel = pipeBase.connectionTypes.Output(
        doc="Camera parameters to use for 'device' part of model",
        name="gbdesAstrometricFit_cameraModel",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "physical_filter"),
    )
    dcrCoefficients = pipeBase.connectionTypes.Output(
        doc="Per-visit coefficients for DCR correction.",
        name="gbdesGlobalAstrometricFit_dcrCoefficients",
        storageClass="ArrowNumpyDict",
    )
    camera = pipeBase.connectionTypes.Output(
        doc="Camera object constructed using the per-detector part of the astrometric model",
        name="gbdesGlobalAstrometricFitCamera",
        storageClass="Camera",
        dimensions=("instrument", "physical_filter"),
    )

    def getSpatialBoundsConnections(self):
        return ("inputVisitSummaries",)

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not self.config.useColor:
            self.inputs.remove("colorCatalog")
            self.outputs.remove("dcrCoefficients")
        if not self.config.saveModelParams:
            self.outputs.remove("modelParams")
        if not self.config.useInputCameraModel:
            self.prerequisiteInputs.remove("inputCameraModel")
        if not self.config.saveCameraModel:
            self.outputs.remove("outputCameraModel")
        if not self.config.saveCameraObject:
            self.prerequisiteInputs.remove("inputCamera")
            self.outputs.remove("camera")


class GbdesGlobalAstrometricFitConfig(
    GbdesAstrometricFitConfig, pipelineConnections=GbdesGlobalAstrometricFitConnections
):
    visitOverlap = pexConfig.Field(
        dtype=float,
        default=1.0,
        doc=(
            "The linkage distance threshold above which clustered groups of visits will not be merged "
            "together in an agglomerative clustering algorithm. The linkage distance is calculated using the "
            "minimum distance between the field-of-view centers of a given visit and all other visits in a "
            "group, and is in units of the field-of-view radius. The resulting groups of visits define the "
            "fields for the astrometric fit."
        ),
    )


class GbdesGlobalAstrometricFitTask(GbdesAstrometricFitTask):
    """Calibrate the WCS across multiple visits and multiple fields using the
    GBDES package.

    This class assumes that the input visits can be separated into contiguous
    groups, for which an individual group covers an area of less than a
    hemisphere.
    """

    ConfigClass = GbdesGlobalAstrometricFitConfig
    _DefaultName = "gbdesAstrometricFit"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # We override runQuantum to set up the refObjLoaders
        inputs = butlerQC.get(inputRefs)

        instrumentName = butlerQC.quantum.dataId["instrument"]

        # Ensure the inputs are in a consistent and deterministic order
        inputSumVisits = np.array([inputSum[0]["visit"] for inputSum in inputs["inputVisitSummaries"]])
        inputs["inputVisitSummaries"] = [inputs["inputVisitSummaries"][v] for v in inputSumVisits.argsort()]
        inputRefHtm7s = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputRefs.referenceCatalog])
        inputRefCatRefs = [inputRefs.referenceCatalog[htm7] for htm7 in inputRefHtm7s.argsort()]
        inputRefCats = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputs["referenceCatalog"]])
        inputs["referenceCatalog"] = [inputs["referenceCatalog"][v] for v in inputRefCats.argsort()]
        inputIsolatedStarSourceTracts = np.array(
            [isolatedStarSource.dataId["tract"] for isolatedStarSource in inputs["isolatedStarSources"]]
        )
        inputIsolatedStarCatalogTracts = np.array(
            [isolatedStarCatalog.dataId["tract"] for isolatedStarCatalog in inputs["isolatedStarCatalogs"]]
        )
        for tract in inputIsolatedStarCatalogTracts:
            if tract not in inputIsolatedStarSourceTracts:
                raise RuntimeError(f"tract {tract} in isolated_star_cats but not isolated_star_sources")
        inputs["isolatedStarSources"] = np.array(
            [inputs["isolatedStarSources"][t] for t in inputIsolatedStarSourceTracts.argsort()]
        )
        inputs["isolatedStarCatalogs"] = np.array(
            [inputs["isolatedStarCatalogs"][t] for t in inputIsolatedStarSourceTracts.argsort()]
        )

        refConfig = LoadReferenceObjectsConfig()
        if self.config.applyRefCatProperMotion:
            refConfig.requireProperMotion = True
        refObjectLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefCatRefs],
            refCats=inputs.pop("referenceCatalog"),
            config=refConfig,
            log=self.log,
        )
        if self.config.useColor:
            colorCatalog = inputs.pop("colorCatalog")
        else:
            colorCatalog = None

        try:
            output = self.run(
                **inputs,
                instrumentName=instrumentName,
                refObjectLoader=refObjectLoader,
                colorCatalog=colorCatalog,
            )
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(e, self, log=self.log)
            # No partial outputs for butler to put
            raise error from e

        for outputRef in outputRefs.outputWcs:
            visit = outputRef.dataId["visit"]
            butlerQC.put(output.outputWcss[visit], outputRef)
        butlerQC.put(output.outputCatalog, outputRefs.outputCatalog)
        butlerQC.put(output.starCatalog, outputRefs.starCatalog)
        if self.config.saveModelParams:
            butlerQC.put(output.modelParams, outputRefs.modelParams)
        if self.config.saveCameraModel:
            butlerQC.put(output.cameraModelParams, outputRefs.outputCameraModel)
        if self.config.saveCameraObject:
            butlerQC.put(output.camera, outputRefs.camera)
        if self.config.useColor:
            butlerQC.put(output.colorParams, outputRefs.dcrCoefficients)
        if output.partialOutputs:
            e = RuntimeError("Some visits were dropped because data was insufficient to fit model.")
            error = pipeBase.AnnotatedPartialOutputsError.annotate(e, self, log=self.log)
            raise error from e

    def run(
        self,
        inputVisitSummaries,
        isolatedStarSources,
        isolatedStarCatalogs,
        instrumentName="",
        refEpoch=None,
        refObjectLoader=None,
        inputCameraModel=None,
        colorCatalog=None,
        inputCamera=None,
    ):
        """Run the WCS fit for a given set of visits

        Parameters
        ----------
        inputVisitSummaries : `list` [`lsst.afw.table.ExposureCatalog`]
            List of catalogs with per-detector summary information.
        isolatedStarSources : `list` [`DeferredDatasetHandle`]
            List of handles pointing to isolated star sources.
        isolatedStarCatalog: `list` [`DeferredDatasetHandle`]
            List of handles pointing to isolated star catalogs.
        instrumentName : `str`, optional
            Name of the instrument used. This is only used for labelling.
        refEpoch : `float`, optional
            Epoch of the reference objects in MJD.
        refObjectLoader : instance of
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`,
            optional
            Reference object loader instance.
        inputCameraModel : `dict` [`str`, `np.ndarray`], optional
            Parameters to use for the device part of the model.
        colorCatalog : `lsst.afw.table.SimpleCatalog`
            Catalog containing object coordinates and magnitudes.
        inputCamera : `lsst.afw.cameraGeom.Camera`, optional
            Camera to be used as template when constructing new camera.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``outputWcss`` : `list` [`lsst.afw.table.ExposureCatalog`]
                List of exposure catalogs (one per visit) with the WCS for each
                detector set by the new fitted WCS.
            ``fitModel`` : `wcsfit.WCSFit`
                Model-fitting object with final model parameters.
            ``outputCatalog`` : `pyarrow.Table`
                Catalog with fit residuals of all sources used.
            ``starCatalog`` : `pyarrow.Table`
                Catalog with best-fit positions of the objects fit.
            ``modelParams`` : `dict`
                Parameters and covariance of the best-fit WCS model.
            ``cameraModelParams`` : `dict` [`str`, `np.ndarray`]
                Parameters of the device part of the model, in the format
                needed as input for future runs.
            ``colorParams`` : `dict` [`int`, `np.ndarray`]
                DCR parameters fit in RA and Dec directions for each visit.
            ``camera`` : `lsst.afw.cameraGeom.Camera`
                Camera object constructed from the per-detector model.
        """
        self.log.info("Gather instrument, exposure, and field info")

        # Get information about the extent of the input visits
        fields, fieldRegions = self._prep_sky(inputVisitSummaries)

        # Get RA, Dec, MJD, etc., for the input visits
        exposureInfo, exposuresHelper, extensionInfo, instruments = self._get_exposure_info(
            inputVisitSummaries, fieldRegions=fieldRegions
        )

        self.log.info("Load associated sources")
        medianEpoch = astropy.time.Time(exposureInfo.medianEpoch, format="jyear").mjd
        allRefObjects, allRefCovariances = {}, {}
        for f, fieldRegion in fieldRegions.items():
            refObjects, refCovariance = self._load_refcat(
                refObjectLoader, extensionInfo, epoch=medianEpoch, region=fieldRegion
            )
            allRefObjects[f] = refObjects
            allRefCovariances[f] = refCovariance

        associations, sourceDict = self._associate_from_isolated_sources(
            isolatedStarSources, isolatedStarCatalogs, extensionInfo, allRefObjects
        )

        self.log.info("Fit the WCSs")
        # Set up a YAML-type string using the config variables and a sample
        # visit
        inputYaml, mapTemplate = self.make_yaml(
            inputVisitSummaries[0],
            inputCameraModel=(inputCameraModel if self.config.useInputCameraModel else None),
        )

        # Set the verbosity level for WCSFit from the task log level.
        # TODO: DM-36850, Add lsst.log to gbdes so that log messages are
        # properly propagated.
        loglevel = self.log.getEffectiveLevel()
        if loglevel >= self.log.WARNING:
            verbose = 0
        elif loglevel == self.log.INFO:
            verbose = 1
        else:
            verbose = 2

        # Set up the WCS-fitting class using the source matches from the
        # isolated star sources plus the reference catalog.
        wcsf = wcsfit.WCSFit(
            fields,
            instruments,
            exposuresHelper,
            extensionInfo.visitIndex,
            extensionInfo.detectorIndex,
            inputYaml,
            extensionInfo.wcs,
            associations.sequence,
            associations.extn,
            associations.obj,
            sysErr=self.config.systematicError,
            refSysErr=self.config.referenceSystematicError,
            usePM=self.config.fitProperMotion,
            verbose=verbose,
        )

        # Add the science and reference sources
        self._add_objects(wcsf, sourceDict, extensionInfo)
        for f in fieldRegions.keys():
            self._add_ref_objects(
                wcsf, allRefObjects[f], allRefCovariances[f], extensionInfo, fieldIndex=-1 * f
            )
        if self.config.useColor:
            self._add_color_objects(wcsf, colorCatalog)

        # Do the WCS fit
        try:
            wcsf.fit(
                reserveFraction=self.config.fitReserveFraction,
                randomNumberSeed=self.config.fitReserveRandomSeed,
                clipThresh=self.config.clipThresh,
                clipFraction=self.config.clipFraction,
            )
        except RuntimeError as e:
            if "Cholesky decomposition failed" in str(e):
                raise CholeskyError() from e
            else:
                raise

        self.log.info("WCS fitting done")

        outputWcss, cameraParams, colorParams, camera, partialOutputs = self._make_outputs(
            wcsf,
            inputVisitSummaries,
            exposureInfo,
            mapTemplate,
            extensionInfo,
            inputCameraModel=(inputCameraModel if self.config.useInputCameraModel else None),
            inputCamera=(inputCamera if self.config.buildCamera else None),
        )
        outputCatalog = wcsf.getOutputCatalog()
        outputCatalog["exposureName"] = np.array(outputCatalog["exposureName"])
        outputCatalog["deviceName"] = np.array(outputCatalog["deviceName"])

        starCatalog = wcsf.getStarCatalog()
        modelParams = self._compute_model_params(wcsf) if self.config.saveModelParams else None

        return pipeBase.Struct(
            outputWcss=outputWcss,
            fitModel=wcsf,
            outputCatalog=outputCatalog,
            starCatalog=starCatalog,
            modelParams=modelParams,
            cameraModelParams=cameraParams,
            colorParams=colorParams,
            camera=camera,
            partialOutputs=partialOutputs,
        )

    def _prep_sky(self, inputVisitSummaries):
        """Cluster the input visits into disjoint groups that will define
        separate fields in the astrometric fit, and, for each group, get the
        convex hull around all of its component visits.

        The groups are created such that each visit overlaps with at least one
        other visit in the same group by the `visitOverlap` amount, which is
        calculated as a fraction of the field-of-view radius, and no visits
        from separate groups overlap by more than this amount.

        Paramaters
        ----------
        inputVisitSummaries : `list` [`lsst.afw.table.ExposureCatalog`]
            List of catalogs with per-detector summary information.

        Returns
        -------
        fields : `wcsfit.Fields`
            Object with field information.
        fieldRegions : `dict` [`int`, `lsst.sphgeom.ConvexPolygon`]
            Dictionary of regions encompassing each group of input visits,
            keyed by an arbitrary index.
        """
        allDetectorCorners = []
        mjds = []
        radecs = []
        radii = []
        for visSum in inputVisitSummaries:
            detectorCorners = [
                lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees).getVector()
                for (ra, dec) in zip(visSum["raCorners"].ravel(), visSum["decCorners"].ravel())
                if (np.isfinite(ra) and (np.isfinite(dec)))
            ]
            if len(detectorCorners) == 0:
                # Skip this visit if none of the detectors have finite ra/dec
                # corners, which happens when the WCSs are missing. The visit
                # will get formally dropped elsewhere.
                continue
            allDetectorCorners.append(detectorCorners)

            # Get center and approximate radius of field of view
            boundingCircle = lsst.sphgeom.ConvexPolygon.convexHull(detectorCorners).getBoundingCircle()
            center = lsst.geom.SpherePoint(boundingCircle.getCenter())
            ra = center.getRa().asDegrees()
            dec = center.getDec().asDegrees()
            radecs.append([ra, dec])
            radius = boundingCircle.getOpeningAngle()
            radii.append(radius)

            obsDate = visSum[0].getVisitInfo().getDate()
            obsMJD = obsDate.get(obsDate.MJD)
            mjds.append(obsMJD)

        # Find groups of visits where any one of the visits overlaps another by
        # a given fraction of the field-of-view radius.
        distance = self.config.visitOverlap * np.median(radii)
        clustering = AgglomerativeClustering(
            distance_threshold=distance.asDegrees(), n_clusters=None, linkage="single"
        )
        clusters = clustering.fit(np.array(radecs))

        medianMJD = np.median(mjds)
        medianEpoch = astropy.time.Time(medianMJD, format="mjd").jyear

        fieldNames = []
        fieldRAs = []
        fieldDecs = []
        epochs = []
        fieldRegions = {}

        for i in range(clusters.n_clusters_):
            fieldInd = clusters.labels_ == i
            # Concatenate the lists of all detector corners that are in this
            # field
            fieldDetectors = sum([allDetectorCorners[f] for f, fInd in enumerate(fieldInd) if fInd], [])
            hull = lsst.sphgeom.ConvexPolygon.convexHull(fieldDetectors)
            center = lsst.geom.SpherePoint(hull.getCentroid())
            ra = center.getRa().asDegrees()
            dec = center.getDec().asDegrees()

            fieldRegions[i] = hull
            fieldNames.append(str(i))
            fieldRAs.append(ra)
            fieldDecs.append(dec)
            # Use the same median epoch for all fields so that the final object
            # positions are calculated for the same epoch.
            epochs.append(medianEpoch)

        fields = wcsfit.Fields(fieldNames, fieldRAs, fieldDecs, epochs)

        return fields, fieldRegions

    def _associate_from_isolated_sources(
        self, isolatedStarSourceRefs, isolatedStarCatalogRefs, extensionInfo, refObjects
    ):
        """Match the input catalog of isolated stars with the reference catalog
        and transform the combined isolated star sources and reference source
        into the format needed for gbdes.

        Parameters
        ----------
        isolatedStarSourceRefs : `list` [`DeferredDatasetHandle`]
            List of handles pointing to isolated star sources.
        isolatedStarCatalogRefs: `list` [`DeferredDatasetHandle`]
            List of handles pointing to isolated star catalogs.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector).
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        refObjects : `dict`
            Dictionary of dictionaries containing the position and error
            information of reference objects.

        Returns
        -------
        associations : `lsst.pipe.base.Struct`
            Struct containing the associations of sources with objects.
        sourceDict : `dict` [`int`, [`int`, [`str`, `list` [`float`]]]]
            Dictionary containing the source centroids for each visit.
        """
        sequences = []
        extensions = []
        object_indices = []

        sourceColumns = ["x", "y", "xErr", "yErr", "ixx", "ixy", "iyy", "obj_index", "visit", "detector"]
        catalogColumns = ["ra", "dec"]

        sourceDict = dict([(visit, {}) for visit in np.unique(extensionInfo.visit)])
        for visit, detector in zip(extensionInfo.visit, extensionInfo.detector):
            sourceDict[visit][detector] = {"x": [], "y": [], "xCov": [], "yCov": [], "xyCov": []}

        for isolatedStarCatalogRef, isolatedStarSourceRef in zip(
            isolatedStarCatalogRefs, isolatedStarSourceRefs
        ):
            isolatedStarCatalog = isolatedStarCatalogRef.get(parameters={"columns": catalogColumns})
            isolatedStarSources = isolatedStarSourceRef.get(parameters={"columns": sourceColumns})
            if len(isolatedStarCatalog) == 0:
                # This is expected when only one visit overlaps with a given
                # tract, meaning that no sources can be associated.
                self.log.debug(
                    "Skipping tract %d, which has no associated isolated stars",
                    isolatedStarCatalogRef.dataId["tract"],
                )
                continue

            # Match the reference stars to the existing isolated stars, then
            # insert the reference stars into the isolated star sources.
            allVisits = np.copy(isolatedStarSources["visit"])
            allDetectors = np.copy(isolatedStarSources["detector"])
            allObjectIndices = np.copy(isolatedStarSources["obj_index"])
            issIndices = np.copy(isolatedStarSources.index)
            for f, regionRefObjects in refObjects.items():
                # Use the same matching technique that is done in
                # isolatedStarAssociation and fgcmBuildFromIsolatedStars.
                with Matcher(
                    isolatedStarCatalog["ra"].to_numpy(), isolatedStarCatalog["dec"].to_numpy()
                ) as matcher:
                    idx, idx_isoStarCat, idx_refObjects, d = matcher.query_radius(
                        np.array(regionRefObjects["ra"]),
                        np.array(regionRefObjects["dec"]),
                        self.config.matchRadius / 3600.0,
                        return_indices=True,
                    )

                refSort = np.searchsorted(isolatedStarSources["obj_index"], idx_isoStarCat)
                refDetector = np.ones(len(idx_isoStarCat)) * -1
                # The "visit" for the reference catalogs is the field times -1.
                refVisit = np.ones(len(idx_isoStarCat)) * f * -1

                allVisits = np.insert(allVisits, refSort, refVisit)
                allDetectors = np.insert(allDetectors, refSort, refDetector)
                allObjectIndices = np.insert(allObjectIndices, refSort, idx_isoStarCat)
                issIndices = np.insert(issIndices, refSort, idx_refObjects)

            # Loop through the associated sources to convert them to the gbdes
            # format, which requires the extension index, the source's index in
            # the input table, and a sequence number corresponding to the
            # object with which it is associated.
            sequence = 0
            obj_index = allObjectIndices[0]
            for visit, detector, row, obj_ind in zip(allVisits, allDetectors, issIndices, allObjectIndices):
                extensionIndex = np.flatnonzero(
                    (extensionInfo.visit == visit) & (extensionInfo.detector == detector)
                )
                if len(extensionIndex) == 0:
                    # This happens for runs where you are not using all the
                    # visits overlapping a given tract that were included in
                    # the isolated star association task."
                    continue
                else:
                    extensionIndex = extensionIndex[0]

                extensions.append(extensionIndex)
                if visit <= 0:
                    object_indices.append(row)
                else:
                    object_indices.append(len(sourceDict[visit][detector]["x"]))
                    source = isolatedStarSources.loc[row]
                    sourceDict[visit][detector]["x"].append(source["x"])
                    sourceDict[visit][detector]["y"].append(source["y"])
                    xCov = source["xErr"] ** 2
                    yCov = source["yErr"] ** 2
                    xyCov = source["ixy"] * (xCov + yCov) / (source["ixx"] + source["iyy"])
                    # TODO: add correct xyErr if DM-7101 is ever done.
                    sourceDict[visit][detector]["xCov"].append(xCov)
                    sourceDict[visit][detector]["yCov"].append(yCov)
                    sourceDict[visit][detector]["xyCov"].append(xyCov)
                if obj_ind != obj_index:
                    sequence = 0
                    sequences.append(sequence)
                    obj_index = obj_ind
                    sequence += 1
                else:
                    sequences.append(sequence)
                    sequence += 1

        associations = pipeBase.Struct(extn=extensions, obj=object_indices, sequence=sequences)
        return associations, sourceDict

    def _add_objects(self, wcsf, sourceDict, extensionInfo):
        """Add science sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object.
        sourceDict : `dict`
            Dictionary containing the source centroids for each visit.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension (visit/detector).
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` [`int`]
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` [`int`]
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` [`lsst.afw.geom.SkyWcs`]
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` [`str`]
                "SCIENCE" or "REFERENCE".
        """
        for visit, visitSources in sourceDict.items():
            # Visit numbers equal or below zero connote the reference catalog.
            if visit <= 0:
                # This "visit" number corresponds to a reference catalog.
                continue

            for detector, sourceCat in visitSources.items():
                extensionIndex = np.flatnonzero(
                    (extensionInfo.visit == visit) & (extensionInfo.detector == detector)
                )[0]

                d = {
                    "x": np.array(sourceCat["x"]),
                    "y": np.array(sourceCat["y"]),
                    "xCov": np.array(sourceCat["xCov"]),
                    "yCov": np.array(sourceCat["yCov"]),
                    "xyCov": np.array(sourceCat["xyCov"]),
                }
                wcsf.setObjects(
                    extensionIndex,
                    d,
                    "x",
                    "y",
                    ["xCov", "yCov", "xyCov"],
                    defaultColor=self.config.referenceColor,
                )


class GbdesGlobalAstrometricMultibandFitConnections(
    GbdesGlobalAstrometricFitConnections,
    dimensions=("instrument",),
):
    outputCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="gbdesGlobalAstrometricMultibandFit_fitStars",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument",),
    )
    starCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Catalog of best-fit object positions. Also includes the fit proper motion and parallax if "
            "fitProperMotion is True."
        ),
        name="gbdesGlobalAstrometricMultibandFit_starCatalog",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument",),
    )
    modelParams = pipeBase.connectionTypes.Output(
        doc="WCS parameters and covariance.",
        name="gbdesGlobalAstrometricMultibandFit_modelParams",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument",),
    )


class GbdesGlobalAstrometricMultibandFitConfig(
    GbdesAstrometricFitConfig,
    pipelineConnections=GbdesGlobalAstrometricMultibandFitConnections,
):
    """Configuration for the GbdesGlobalAstrometricMultibandFitTask"""

    pass


class GbdesGlobalAstrometricMultibandFitTask(GbdesGlobalAstrometricFitTask):
    """Calibrate the WCS across multiple visits in multiple filters and
    multiple fields using the GBDES package.
    """

    ConfigClass = GbdesGlobalAstrometricMultibandFitConfig
    _DefaultName = "gbdesAstrometricMultibandFit"
