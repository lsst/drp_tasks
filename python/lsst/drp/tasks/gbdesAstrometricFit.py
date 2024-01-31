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
import astropy.coordinates
import astropy.time
import astropy.units as u
import astshim
import lsst.afw.geom as afwgeom
import lsst.afw.table
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.sphgeom
import numpy as np
import wcsfit
import yaml
from lsst.meas.algorithms import (
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    ReferenceSourceSelectorTask,
)
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry

__all__ = ["GbdesAstrometricFitConnections", "GbdesAstrometricFitConfig", "GbdesAstrometricFitTask"]


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
    cov : `list` of `float`
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
        raErr = (refCat["coord_raErr"] * inputUnit).to(outputCoordUnit).to_value()
        decErr = (refCat["coord_decErr"] * inputUnit).to(outputCoordUnit).to_value()
        raPMErr = (refCat["pm_raErr"] * inputUnit).to(outputPMUnit).to_value()
        decPMErr = (refCat["pm_decErr"] * inputUnit).to(outputPMUnit).to_value()
        parallaxErr = (refCat["parallaxErr"] * inputUnit).to(outputPMUnit).to_value()
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
                    cov[:, k] = (refCat[f"{pi}Err"] ** 2 * inputUnit**2).to_value(units[j] * units[j])
                elif i > j:
                    cov[:, k] = (refCat[f"{pj}_{pi}_Cov"] * inputUnit**2).to_value(units[i] * units[j])
                else:
                    cov[:, k] = (refCat[f"{pi}_{pj}_Cov"] * inputUnit**2).to_value(units[i] * units[j])

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


class GbdesAstrometricFitConnections(
    pipeBase.PipelineTaskConnections, dimensions=("skymap", "tract", "instrument", "physical_filter")
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
    outputWcs = pipeBase.connectionTypes.Output(
        doc=(
            "Per-tract, per-visit world coordinate systems derived from the fitted model."
            " These catalogs only contain entries for detectors with an output, and use"
            " the detector id for the catalog id, sorted on id for fast lookups of a detector."
        ),
        name="gbdesAstrometricFitSkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "skymap", "tract"),
        multiple=True,
    )
    outputCatalog = pipeBase.connectionTypes.Output(
        doc=(
            "Source table with stars used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="gbdesAstrometricFit_fitStars",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    starCatalog = pipeBase.connectionTypes.Output(
        doc="Star catalog.",
        name="gbdesAstrometricFit_starCatalog",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    modelParams = pipeBase.connectionTypes.Output(
        doc="WCS parameter covariance.",
        name="gbdesAstrometricFit_modelParams",
        storageClass="ArrowNumpyDict",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )

    def getSpatialBoundsConnections(self):
        return ("inputVisitSummaries",)

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not self.config.saveModelParams:
            self.outputs.remove("modelParams")


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

    def setDefaults(self):
        # Use only stars because aperture fluxes of galaxies are biased and
        # depend on seeing.
        self.sourceSelector["science"].doUnresolved = True
        self.sourceSelector["science"].unresolved.name = "extendedness"

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

    def validate(self):
        super().validate()

        # Check if all components of the device and exposure models are
        # supported.
        for component in self.deviceModel:
            if not (("poly" in component.lower()) or ("identity" in component.lower())):
                raise pexConfig.FieldValidationError(
                    GbdesAstrometricFitConfig.deviceModel,
                    self,
                    f"deviceModel component {component} is not supported.",
                )

        for component in self.exposureModel:
            if not (("poly" in component.lower()) or ("identity" in component.lower())):
                raise pexConfig.FieldValidationError(
                    GbdesAstrometricFitConfig.exposureModel,
                    self,
                    f"exposureModel component {component} is not supported.",
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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # We override runQuantum to set up the refObjLoaders
        inputs = butlerQC.get(inputRefs)

        instrumentName = butlerQC.quantum.dataId["instrument"]

        # Ensure the inputs are in a consistent order
        inputCatVisits = np.array([inputCat.dataId["visit"] for inputCat in inputs["inputCatalogRefs"]])
        inputs["inputCatalogRefs"] = [inputs["inputCatalogRefs"][v] for v in inputCatVisits.argsort()]
        inputSumVisits = np.array([inputSum[0]["visit"] for inputSum in inputs["inputVisitSummaries"]])
        inputs["inputVisitSummaries"] = [inputs["inputVisitSummaries"][v] for v in inputSumVisits.argsort()]
        inputRefHtm7s = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputRefs.referenceCatalog])
        inputRefCatRefs = [inputRefs.referenceCatalog[htm7] for htm7 in inputRefHtm7s.argsort()]
        inputRefCats = np.array([inputRefCat.dataId["htm7"] for inputRefCat in inputs["referenceCatalog"]])
        inputs["referenceCatalog"] = [inputs["referenceCatalog"][v] for v in inputRefCats.argsort()]

        sampleRefCat = inputs["referenceCatalog"][0].get()
        refEpoch = sampleRefCat[0]["epoch"]

        refConfig = LoadReferenceObjectsConfig()
        refConfig.anyFilterMapsToThis = "phot_g_mean"
        refConfig.requireProperMotion = True
        refObjectLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefCatRefs],
            refCats=inputs.pop("referenceCatalog"),
            config=refConfig,
            log=self.log,
        )

        output = self.run(
            **inputs, instrumentName=instrumentName, refEpoch=refEpoch, refObjectLoader=refObjectLoader
        )

        wcsOutputRefDict = {outWcsRef.dataId["visit"]: outWcsRef for outWcsRef in outputRefs.outputWcs}
        for visit, outputWcs in output.outputWCSs.items():
            butlerQC.put(outputWcs, wcsOutputRefDict[visit])
        butlerQC.put(output.outputCatalog, outputRefs.outputCatalog)
        butlerQC.put(output.starCatalog, outputRefs.starCatalog)
        if self.config.saveModelParams:
            butlerQC.put(output.modelParams, outputRefs.modelParams)

    def run(
        self, inputCatalogRefs, inputVisitSummaries, instrumentName="", refEpoch=None, refObjectLoader=None
    ):
        """Run the WCS fit for a given set of visits

        Parameters
        ----------
        inputCatalogRefs : `list`
            List of `DeferredDatasetHandle`s pointing to visit-level source
            tables.
        inputVisitSummaries : `list` of `lsst.afw.table.ExposureCatalog`
            List of catalogs with per-detector summary information.
        instrumentName : `str`, optional
            Name of the instrument used. This is only used for labelling.
        refEpoch : `float`
            Epoch of the reference objects in MJD.
        refObjectLoader : instance of
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`
            Referencef object loader instance.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``outputWCSs`` : `list` of `lsst.afw.table.ExposureCatalog`
                List of exposure catalogs (one per visit) with the WCS for each
                detector set by the new fitted WCS.
            ``fitModel`` : `wcsfit.WCSFit`
                Model-fitting object with final model parameters.
            ``outputCatalog`` : `pyarrow.Table`
                Catalog with fit residuals of all sources used.
        """
        if (len(inputVisitSummaries) == 1) and self.config.deviceModel and self.config.exposureModel:
            raise RuntimeError(
                "More than one exposure is necessary to break the degeneracy between the "
                "device model and the exposure model."
            )
        self.log.info("Gathering instrument, exposure, and field info")
        # Set up an instrument object
        instrument = wcsfit.Instrument(instrumentName)

        # Get RA, Dec, MJD, etc., for the input visits
        exposureInfo, exposuresHelper, extensionInfo = self._get_exposure_info(
            inputVisitSummaries, instrument
        )

        # Get information about the extent of the input visits
        fields, fieldCenter, fieldRadius = self._prep_sky(inputVisitSummaries, exposureInfo.medianEpoch)

        self.log.info("Load catalogs and associate sources")
        # Set up class to associate sources into matches using a
        # friends-of-friends algorithm
        associations = wcsfit.FoFClass(
            fields,
            [instrument],
            exposuresHelper,
            [fieldRadius.asDegrees()],
            (self.config.matchRadius * u.arcsec).to(u.degree).value,
        )

        # Add the reference catalog to the associator
        medianEpoch = astropy.time.Time(exposureInfo.medianEpoch, format="decimalyear").mjd
        refObjects, refCovariance = self._load_refcat(
            associations, refObjectLoader, fieldCenter, fieldRadius, extensionInfo, epoch=medianEpoch
        )

        # Add the science catalogs and associate new sources as they are added
        sourceIndices, usedColumns = self._load_catalogs_and_associate(
            associations, inputCatalogRefs, extensionInfo
        )
        self._check_degeneracies(associations, extensionInfo)

        self.log.info("Fit the WCSs")
        # Set up a YAML-type string using the config variables and a sample
        # visit
        inputYAML, mapTemplate = self.make_yaml(inputVisitSummaries[0])

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
        wcsf = wcsfit.WCSFit(
            fields,
            [instrument],
            exposuresHelper,
            extensionInfo.visitIndex,
            extensionInfo.detectorIndex,
            inputYAML,
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
        self._add_objects(wcsf, inputCatalogRefs, sourceIndices, extensionInfo, usedColumns)
        self._add_ref_objects(wcsf, refObjects, refCovariance, extensionInfo)

        # There must be at least as many sources per visit as the number of
        # free parameters in the per-visit mapping. Set minFitExposures to be
        # the number of free parameters, so that visits with fewer visits are
        # dropped.
        nCoeffVisitModel = _nCoeffsFromDegree(self.config.exposurePolyOrder)
        # Do the WCS fit
        wcsf.fit(
            reserveFraction=self.config.fitReserveFraction,
            randomNumberSeed=self.config.fitReserveRandomSeed,
            minFitExposures=nCoeffVisitModel,
        )
        self.log.info("WCS fitting done")

        outputWCSs = self._make_outputs(wcsf, inputVisitSummaries, exposureInfo, mapTemplate=mapTemplate)
        outputCatalog = wcsf.getOutputCatalog()
        starCatalog = wcsf.getStarCatalog()
        modelParams = self._compute_model_params(wcsf) if self.config.saveModelParams else None

        return pipeBase.Struct(
            outputWCSs=outputWCSs,
            fitModel=wcsf,
            outputCatalog=outputCatalog,
            starCatalog=starCatalog,
            modelParams=modelParams,
        )

    def _prep_sky(self, inputVisitSummaries, epoch, fieldName="Field"):
        """Get center and radius of the input tract. This assumes that all
        visits will be put into the same `wcsfit.Field` and fit together.

        Paramaters
        ----------
        inputVisitSummaries : `list` of `lsst.afw.table.ExposureCatalog`
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
        self, inputVisitSummaries, instrument, fieldNumber=0, instrumentNumber=0, refEpoch=None
    ):
        """Get various information about the input visits to feed to the
        fitting routines.

        Parameters
        ----------
        inputVisitSummaries : `list` of `lsst.afw.table.ExposureCatalog`
            Tables for each visit with information for detectors.
        instrument : `wcsfit.Instrument`
            Instrument object to which detector information is added.
        fieldNumber : `int`
            Index of the field for these visits. Should be zero if all data is
            being fit together.
        instrumentNumber : `int`
            Index of the instrument for these visits. Should be zero if all
            data comes from the same instrument.
        refEpoch : `float`
            Epoch of the reference objects in MJD.

        Returns
        -------
        exposureInfo : `lsst.pipe.base.Struct`
            Struct containing general properties for the visits:
            ``visits`` : `list`
                List of visit names.
            ``detectors`` : `list`
                List of all detectors in any visit.
            ``ras`` : `list` of float
                List of boresight RAs for each visit.
            ``decs`` : `list` of float
                List of borseight Decs for each visit.
            ``medianEpoch`` : float
                Median epoch of all visits in decimal-year format.
        exposuresHelper : `wcsfit.ExposuresHelper`
            Object containing information about the input visits.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension:
            ``visit`` : `np.ndarray`
                Name of the visit for this extension.
            ``detector`` : `np.ndarray`
                Name of the detector for this extension.
            ``visitIndex` : `np.ndarray` of `int`
                Index of visit for this extension.
            ``detectorIndex`` : `np.ndarray` of `int`
                Index of the detector for this extension.
            ``wcss`` : `np.ndarray` of `lsst.afw.geom.SkyWcs`
                Initial WCS for this extension.
            ``extensionType`` : `np.ndarray` of `str`
                "SCIENCE" or "REFERENCE".
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

        extensionType = []
        extensionVisitIndices = []
        extensionDetectorIndices = []
        extensionVisits = []
        extensionDetectors = []
        # Get information for all the science visits
        for v, visitSummary in enumerate(inputVisitSummaries):
            visitInfo = visitSummary[0].getVisitInfo()
            visit = visitSummary[0]["visit"]
            visits.append(visit)
            exposureNames.append(str(visit))
            raDec = visitInfo.getBoresightRaDec()
            ras.append(raDec.getRa().asRadians())
            decs.append(raDec.getDec().asRadians())
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
                    detectorBounds = wcsfit.Bounds(
                        row["bbox_min_x"], row["bbox_max_x"], row["bbox_min_y"], row["bbox_max_y"]
                    )
                    instrument.addDevice(str(detector), detectorBounds)

                detectorIndex = np.flatnonzero(detector == np.array(detectors))[0]
                extensionVisitIndices.append(v)
                extensionDetectorIndices.append(detectorIndex)
                extensionVisits.append(visit)
                extensionDetectors.append(detector)
                extensionType.append("SCIENCE")

        fieldNumbers = list(np.ones(len(exposureNames), dtype=int) * fieldNumber)
        instrumentNumbers = list(np.ones(len(exposureNames), dtype=int) * instrumentNumber)

        # Set the reference epoch to be the median of the science visits.
        # The reference catalog will be shifted to this date.
        medianMJD = np.median(mjds)
        medianEpoch = astropy.time.Time(medianMJD, format="mjd").decimalyear

        # Add information for the reference catalog. Most of the values are
        # not used.
        exposureNames.append("REFERENCE")
        visits.append(-1)
        fieldNumbers.append(0)
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
        extensionVisits.append(-1)
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

        return exposureInfo, exposuresHelper, extensionInfo

    def _load_refcat(
        self, associations, refObjectLoader, center, radius, extensionInfo, epoch=None, fieldIndex=0
    ):
        """Load the reference catalog and add reference objects to the
        `wcsfit.FoFClass` object.

        Parameters
        ----------
        associations : `wcsfit.FoFClass`
            Object to which to add the catalog of reference objects.
        refObjectLoader :
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`
            Object set up to load reference catalog objects.
        center : `lsst.geom.SpherePoint`
            Center of the circle in which to load reference objects.
        radius : `lsst.sphgeom._sphgeom.Angle`
            Radius of the circle in which to load reference objects.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension.
        epoch : `float`
            MJD to which to correct the object positions.
        fieldIndex : `int`
            Index of the field. Should be zero if all the data is fit together.

        Returns
        -------
        refObjects : `dict`
            Position and error information of reference objects.
        refCovariance : `list` of `float`
            Flattened output covariance matrix.
        """
        formattedEpoch = astropy.time.Time(epoch, format="mjd")

        refFilter = refObjectLoader.config.anyFilterMapsToThis
        skyCircle = refObjectLoader.loadSkyCircle(center, radius, refFilter, epoch=formattedEpoch)

        selected = self.referenceSelector.run(skyCircle.refCat)
        # Need memory contiguity to get reference filters as a vector.
        if not selected.sourceCat.isContiguous():
            refCat = selected.sourceCat.copy(deep=True)
        else:
            refCat = selected.sourceCat

        # In Gaia DR3, missing values are denoted by NaNs.
        finiteInd = np.isfinite(refCat["coord_ra"]) & np.isfinite(refCat["coord_dec"])
        refCat = refCat[finiteInd]

        if self.config.excludeNonPMObjects:
            # Gaia DR2 has zeros for missing data, while Gaia DR3 has NaNs:
            hasPM = (
                (refCat["pm_raErr"] != 0) & np.isfinite(refCat["pm_raErr"]) & np.isfinite(refCat["pm_decErr"])
            )
            refCat = refCat[hasPM]

        ra = (refCat["coord_ra"] * u.radian).to(u.degree).to_value().tolist()
        dec = (refCat["coord_dec"] * u.radian).to(u.degree).to_value().tolist()
        raCov = ((refCat["coord_raErr"] * u.radian).to(u.degree).to_value() ** 2).tolist()
        decCov = ((refCat["coord_decErr"] * u.radian).to(u.degree).to_value() ** 2).tolist()

        # Get refcat version from refcat metadata
        refCatMetadata = refObjectLoader.refCats[0].get().getMetadata()
        refCatVersion = refCatMetadata["REFCAT_FORMAT_VERSION"]
        if refCatVersion == 2:
            raDecCov = (refCat["coord_ra_coord_dec_Cov"] * u.radian**2).to(u.degree**2).to_value().tolist()
        else:
            raDecCov = np.zeros(len(ra))

        refObjects = {"ra": ra, "dec": dec, "raCov": raCov, "decCov": decCov, "raDecCov": raDecCov}
        refCovariance = []

        if self.config.fitProperMotion:
            raPM = (refCat["pm_ra"] * u.radian).to(u.marcsec).to_value().tolist()
            decPM = (refCat["pm_dec"] * u.radian).to(u.marcsec).to_value().tolist()
            parallax = (refCat["parallax"] * u.radian).to(u.marcsec).to_value().tolist()
            cov = _make_ref_covariance_matrix(refCat, version=refCatVersion)
            pmDict = {"raPM": raPM, "decPM": decPM, "parallax": parallax}
            refObjects.update(pmDict)
            refCovariance = cov

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
            Struct containing properties for each extension.
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
        columns : `list` of `str`
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
            Struct containing properties for each extension.
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

        if "BAND/DEVICE/poly" in self.config.deviceModel:
            nCoeffDetectorModel = _nCoeffsFromDegree(self.config.devicePolyOrder)
            unconstrainedDetectors = []
            for detector in np.unique(extensionInfo.detector):
                numSources = (whichDetector == detector).sum()
                if numSources < nCoeffDetectorModel:
                    unconstrainedDetectors.append(str(detector))

            if unconstrainedDetectors:
                raise RuntimeError(
                    "The model is not constrained. The following detectors do not have enough "
                    f"sources ({nCoeffDetectorModel} required): ",
                    ", ".join(unconstrainedDetectors),
                )

    def make_yaml(self, inputVisitSummary, inputFile=None):
        """Make a YAML-type object that describes the parameters of the fit
        model.

        Parameters
        ----------
        inputVisitSummary : `lsst.afw.table.ExposureCatalog`
            Catalog with per-detector summary information.
        inputFile : `str`
            Path to a file that contains a basic model.

        Returns
        -------
        inputYAML : `wcsfit.YAMLCollector`
            YAML object containing the model description.
        inputDict : `dict` [`str`, `str`]
            Dictionary containing the model description.
        """
        if inputFile is not None:
            inputYAML = wcsfit.YAMLCollector(inputFile, "PixelMapCollection")
        else:
            inputYAML = wcsfit.YAMLCollector("", "PixelMapCollection")
        inputDict = {}
        modelComponents = ["INSTRUMENT/DEVICE", "EXPOSURE"]
        baseMap = {"Type": "Composite", "Elements": modelComponents}
        inputDict["EXPOSURE/DEVICE/base"] = baseMap

        xMin = str(inputVisitSummary["bbox_min_x"].min())
        xMax = str(inputVisitSummary["bbox_max_x"].max())
        yMin = str(inputVisitSummary["bbox_min_y"].min())
        yMax = str(inputVisitSummary["bbox_max_y"].max())

        deviceModel = {"Type": "Composite", "Elements": self.config.deviceModel.list()}
        inputDict["INSTRUMENT/DEVICE"] = deviceModel
        for component in self.config.deviceModel:
            if "poly" in component.lower():
                componentDict = {
                    "Type": "Poly",
                    "XPoly": {"OrderX": self.config.devicePolyOrder, "SumOrder": True},
                    "YPoly": {"OrderX": self.config.devicePolyOrder, "SumOrder": True},
                    "XMin": xMin,
                    "XMax": xMax,
                    "YMin": yMin,
                    "YMax": yMax,
                }
            elif "identity" in component.lower():
                componentDict = {"Type": "Identity"}

            inputDict[component] = componentDict

        exposureModel = {"Type": "Composite", "Elements": self.config.exposureModel.list()}
        inputDict["EXPOSURE"] = exposureModel
        for component in self.config.exposureModel:
            if "poly" in component.lower():
                componentDict = {
                    "Type": "Poly",
                    "XPoly": {"OrderX": self.config.exposurePolyOrder, "SumOrder": "true"},
                    "YPoly": {"OrderX": self.config.exposurePolyOrder, "SumOrder": "true"},
                }
            elif "identity" in component.lower():
                componentDict = {"Type": "Identity"}

            inputDict[component] = componentDict

        inputYAML.addInput(yaml.dump(inputDict))
        inputYAML.addInput("Identity:\n  Type:  Identity\n")

        return inputYAML, inputDict

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
            Struct containing properties for each extension.
        columns : `list` of `str`
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

                wcsf.setObjects(extensionIndex, d, "x", "y", ["xCov", "yCov", "xyCov"])

    def _add_ref_objects(self, wcsf, refObjects, refCovariance, extensionInfo):
        """Add reference sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object.
        refObjects : `dict`
            Position and error information of reference objects.
        refCovariance : `list` of `float`
            Flattened output covariance matrix.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension.
        """
        extensionIndex = np.flatnonzero(extensionInfo.extensionType == "REFERENCE")[0]

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

    def _make_outputs(self, wcsf, visitSummaryTables, exposureInfo, mapTemplate=None):
        """Make a WCS object out of the WCS models.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCSFit object, assumed to have fit model.
        visitSummaryTables : `list` of `lsst.afw.table.ExposureCatalog`
            Catalogs with per-detector summary information from which to grab
            detector information.
        extensionInfo : `lsst.pipe.base.Struct`
            Struct containing properties for each extension.

        Returns
        -------
        catalogs : `dict` of [`str`, `lsst.afw.table.ExposureCatalog`]
            Dictionary of `lsst.afw.table.ExposureCatalog` objects with the WCS
            set to the WCS fit in wcsf, keyed by the visit name.
        """
        # Get the parameters of the fit models
        mapParams = wcsf.mapCollection.getParamDict()

        # Set up the schema for the output catalogs
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        schema.addField("visit", type="L", doc="Visit number")

        # Pixels will need to be rescaled before going into the mappings
        sampleDetector = visitSummaryTables[0][0]
        xscale = sampleDetector["bbox_max_x"] - sampleDetector["bbox_min_x"]
        yscale = sampleDetector["bbox_max_y"] - sampleDetector["bbox_min_y"]

        catalogs = {}
        for v, visitSummary in enumerate(visitSummaryTables):
            visit = visitSummary[0]["visit"]

            visitMap = wcsf.mapCollection.orderAtoms(f"{visit}")[0]
            visitMapType = wcsf.mapCollection.getMapType(visitMap)
            if (visitMap not in mapParams) and (visitMapType != "Identity"):
                self.log.warning("Visit %d was dropped because of an insufficient number of sources.", visit)
                continue

            catalog = lsst.afw.table.ExposureCatalog(schema)
            catalog.resize(len(exposureInfo.detectors))
            catalog["visit"] = visit

            for d, detector in enumerate(visitSummary["id"]):
                mapName = f"{visit}/{detector}"
                if mapName in wcsf.mapCollection.allMapNames():
                    mapElements = wcsf.mapCollection.orderAtoms(f"{mapName}/base")
                else:
                    # This extension was not fit, but the WCS can be recovered
                    # using the maps fit from sources on other visits but the
                    # same detector and from sources on other detectors from
                    # this visit.
                    genericElements = mapTemplate["EXPOSURE/DEVICE/base"]["Elements"]
                    mapElements = []
                    instrument = visitSummary[0].getVisitInfo().instrumentLabel
                    # Go through the generic map components to build the names
                    # of the specific maps for this extension.
                    for component in genericElements:
                        elements = mapTemplate[component]["Elements"]
                        for element in elements:
                            # TODO: DM-42519, gbdes sets the "BAND" to the
                            # instrument name currently. This will need to be
                            # disambiguated if we run on multiple bands at
                            # once.
                            element = element.replace("BAND", str(instrument))
                            element = element.replace("EXPOSURE", str(visit))
                            element = element.replace("DEVICE", str(detector))
                            mapElements.append(element)
                mapDict = {}
                for m, mapElement in enumerate(mapElements):
                    mapType = wcsf.mapCollection.getMapType(mapElement)
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
                catalog[d].setWcs(outWCS)
            catalog.sort()
            catalogs[visit] = catalog

        return catalogs

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
