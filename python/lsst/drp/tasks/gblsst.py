# This file is part of pipe_tasks.
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
import numpy as np
import astropy.time
import astropy.units as u
import sip_tpv
import yaml
import wcsfit
import pandas as pd
import astshim

import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.skymap import BaseSkyMap
import lsst.afw.table
import lsst.afw.geom as afwgeom
from lsst.meas.algorithms import (ReferenceObjectLoader, ReferenceSourceSelectorTask,
                                  LoadIndexedReferenceObjectsTask)
from lsst.meas.algorithms.sourceSelector import sourceSelectorRegistry


def lookupVisitRefCats(datasetType, registry, quantumDataId, collections):
    """Lookup function that finds all refcats for all visits that overlap a
    tract, rather than just the refcats that directly overlap the tract.
    Borrowed from jointcal.
    Parameters
    ----------
    datasetType : `lsst.daf.butler.DatasetType`
        Type of dataset being searched for.
    registry : `lsst.daf.butler.Registry`
        Data repository registry to search.
    quantumDataId : `lsst.daf.butler.DataCoordinate`
        Data ID of the quantum; expected to be something we can use as a
        constraint to query for overlapping visits.
    collections : `Iterable` [ `str` ]
        Collections to search.
    Returns
    -------
    refs : `Iterator` [ `lsst.daf.butler.DatasetRef` ]
        Iterator over refcat references.
    """
    refs = set()
    # Use .expanded() on the query methods below because we need data IDs with
    # regions, both in the outer loop over visits (queryDatasets will expand
    # any data ID we give it, but doing it up-front in bulk is much more
    # efficient) and in the data IDs of the DatasetRefs this function yields
    # (because the RefCatLoader relies on them to do some of its own
    # filtering).
    for visit_data_id in set(registry.queryDataIds("visit", dataId=quantumDataId).expanded()):
        refs.update(
            registry.queryDatasets(
                datasetType,
                collections=collections,
                dataId=visit_data_id,
                findFirst=True,
            ).expanded()
        )
    yield from refs


def convertToAstPolyMapCoefficients(coefficients):
    """Convert vector of polynomial coefficients from the format used in
    `gbdes` into AST format (see Poly2d::vectorIndex(i, j) in
    gbdes/gbutil/src/Poly2d.cpp). This assumes two input and two output
    coordinates.

    Parameters
    ----------
    coefficients : `list`
        Coefficients of the polynomials
    degree : `int`
        Degree of the polynomial.

    Returns
    -------
    astPoly : `astshim.PolyMap`
        Coefficients in AST polynomial format
    """
    polyArray = np.zeros((len(coefficients), 4))
    N = len(coefficients) / 2
    degree = int(-1.5 + 0.5 * (1 + 8 * N)**0.5)

    for outVar in [1, 2]:
        for i in range(degree + 1):
            for j in range(degree + 1):
                if (i + j) > degree:
                    continue
                vectorIndex = int(((i+j)*(i+j+1))/2+j + N * (outVar - 1))
                polyArray[vectorIndex, 0] = coefficients[vectorIndex]
                polyArray[vectorIndex, 1] = outVar
                polyArray[vectorIndex, 2] = i
                polyArray[vectorIndex, 3] = j

    # TODO: add inverse coefficients, see jointcal for how to compute
    astPoly = astshim.PolyMap(polyArray, 2, options="IterInverse=1,NIterInverse=10")
    return astPoly


def getWCSfromSIP(butlerWcs):
    """Get wcsfit.Wcs in TPV format from the SIP-formatted input WCS

    Parameters
    ----------
    butlerWcs : `lsst.afw.geom.SkyWcs`
        Input WCS from the calexp in SIP format

    Returns
    -------
    wcs : `wcsfit.Wcs`
        WCS object in TPV format
    """
    fits_metadata = butlerWcs.getFitsMetadata()
    if not ((fits_metadata.get('CTYPE1') == 'RA---TAN-SIP')
            and (fits_metadata.get('CTYPE2') == 'DEC--TAN-SIP')):
        raise ValueError(f'CTYPES {fits_metadata.get("CTYPE1")} and {fits_metadata.get("CTYPE2")}'
                         'do not match SIP convention')

    floatDict = {k: fits_metadata[k] for k in fits_metadata if isinstance(fits_metadata[k], (int, float))}
    wcs = wcsfit.readTPVFromSIP(floatDict, "test")

    return wcs


class GblsstTaskConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("skymap", "tract", "instrument", "physical_filter")):
    """Middleware input/output connections for gblsst data."""
    inputSourceTableVisitRefs = pipeBase.connectionTypes.Input(
        doc="Source table in parquet format, per visit",
        name="sourceTable_visit",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        multiple=True,
    )
    inputVisitSummary = pipeBase.connectionTypes.Input(
        doc=("Per-visit consolidated exposure metadata built from calexps. "
             "These catalogs use detector id for the id and must be sorted for "
             "fast lookups of a detector."),
        name="visitSummary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/bbox and projection/wcs for input exposures",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap", ),
        storageClass="SkyMap",
    )
    astrometryRefCat = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The astrometry reference catalog to match to loaded input catalog sources.",
        name="gaia_dr2_20200414",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
        lookupFunction=lookupVisitRefCats,
    )
    outputWcs = pipeBase.connectionTypes.Output(
        doc=("Per-tract, per-visit world coordinate systems derived from the fitted model."
             " These catalogs only contain entries for detectors with an output, and use"
             " the detector id for the catalog id, sorted on id for fast lookups of a detector."),
        name="GblsstSkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "skymap", "tract"),
        multiple=True
    )


class GblsstConfig(pipeBase.PipelineTaskConfig,
                   pipelineConnections=GblsstTaskConnections):
    """Configuration for GblsstTask"""
    sourceSelector = sourceSelectorRegistry.makeField(
        doc="How to select sources for cross-matching",
        default="science"
    )
    astrometryReferenceSelector = pexConfig.ConfigurableField(
        target=ReferenceSourceSelectorTask,
        doc="How to down-select the loaded astrometry reference catalog.",
    )
    astrometryRefObjLoader = pexConfig.ConfigurableField(
        target=LoadIndexedReferenceObjectsTask,
        doc="Reference object loader for astrometric fit",
    )
    matchRadius = pexConfig.Field(
        doc="Matching tolerance between associated objects (arcseconds)",
        dtype=float,
        default=1.0
    )
    minMatches = pexConfig.Field(
        doc="Number of matches required to keep a source object",
        dtype=int,
        default=2
    )
    allowSelfMatches = pexConfig.Field(
        doc="Allow multiple object matches from the same viist",
        dtype=bool,
        default=False
    )
    sourceFluxType = pexConfig.Field(
        dtype=str,
        doc="Source flux field to use in source selection and to get fluxes from the catalog.",
        default='apFlux_12_0'
    )
    systematicError = pexConfig.Field(
        dtype=float,
        doc="Systematic error padding for the science images (arcsec)",
        default=0.0
    )
    referenceSystematicError = pexConfig.Field(
        dtype=float,
        doc="Systematic error padding for the reference catalog (arcsec)",
        default=0.0
    )
    modelComponents = pexConfig.ListField(
        dtype=str,
        doc="List of mappings to apply to transform from pixels to sky, in order of their application",
        default=["INSTRUMENT/DEVICE", "EXPOSURE"]
    )
    deviceModel = pexConfig.ListField(
        dtype=str,
        doc="List of mappings to apply to transform from device pixels to intermediate frame",
        default=["BAND/DEVICE/poly"]
    )
    exposureModel = pexConfig.ListField(
        dtype=str,
        doc="List of mappyings to apply to transform from intermediate frame to sky coordinates",
        default=["EXPOSURE/poly"]
    )
    devicePolyOrder = pexConfig.Field(
        dtype=int,
        doc="Order of device polynomial model",
        default=4
    )
    exposurePolyOrder = pexConfig.Field(
        dtype=int,
        doc="Order of device polynomial model",
        default=6
    )
    useProperMotion = pexConfig.Field(
        dtype=bool,
        doc="Fit the proper motions of the objects",
        default=False
    )
    verbose = pexConfig.Field(
        dtype=int,
        doc="Verbosity level=0, 1, or 2",
        default=2
    )

    def setDefaults(self):
        # Use only stars because aperture fluxes of galaxies are biased and depend on seeing.
        self.sourceSelector["science"].doUnresolved = True
        self.sourceSelector["science"].unresolved.name = "extendedness"

        # Use only isolated sources.
        self.sourceSelector["science"].doIsolated = True
        self.sourceSelector["science"].isolated.parentName = "parentSourceId"
        self.sourceSelector["science"].isolated.nChildName = "deblend_nChild"
        # Do not trust either flux or centroid measurements with flags,
        # chosen from the usual QA flags for stars.
        self.sourceSelector["science"].doFlags = True
        badFlags = ["pixelFlags_edge",
                    "pixelFlags_saturated",
                    "pixelFlags_interpolatedCenter",
                    "pixelFlags_interpolated",
                    "pixelFlags_crCenter",
                    "pixelFlags_bad",
                    "hsmPsfMoments_flag",
                    f"{self.sourceFluxType}_flag",
                    ]
        self.sourceSelector["science"].flags.bad = badFlags

        # Requirements based on using gaia_dr2_20200414 reference catalog
        self.astrometryRefObjLoader.requireProperMotion = True
        self.astrometryRefObjLoader.anyFilterMapsToThis = "phot_g_mean"


class GblsstTask(pipeBase.PipelineTask):
    """Calibrate the WCS across multiple visits of the same field.

    Parameters
    ----------
    initInputs : `dict`, optional
        Dictionary used to initialize PipelineTasks (empty for gblsst).
    """

    ConfigClass = GblsstConfig
    _DefaultName = "gblsst"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("sourceSelector")
        self.makeSubtask("astrometryReferenceSelector")

        self.visitDeviceInfo = []
        self.devices = []
        self.visits = []

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # We override runQuantum to set up the refObjLoaders
        inputs = butlerQC.get(inputRefs)

        # We want the tract number for getting the extent of the field
        tract = butlerQC.quantum.dataId['tract']
        instrumentName = butlerQC.quantum.dataId['instrument']

        self.astrometryRefObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId
                     for ref in inputRefs.astrometryRefCat],
            refCats=inputs.pop('astrometryRefCat'),
            config=self.config.astrometryRefObjLoader,
            log=self.log)

        outputCatalogs = self.run(**inputs, tract=tract, instrumentName=instrumentName)

        for outputRef in outputRefs.outputWcs:
            visit = outputRef.dataId['visit']
            butlerQC.put(outputCatalogs[visit], outputRef)

    def run(self, inputSourceTableVisitRefs, inputVisitSummary, skyMap, tract=None, instrumentName=None):
        """Run the WCS fit for a given set of visits

        Parameters
        ----------
        inputSourceTableVisitRefs : `list`
            List of `DeferredDatasetHandle`s pointing to visit-level source
            tables
        inputVisitSummary : `list` of `lsst.afw.table.ExposureCatalog`
            List of catalogs with per-detector summary information
        skyMap : 'lsst.skymap.ringsSkyMap.RingsSkyMap'
            Definition of geometry/bbox for input exposures
        tract : `int`, optional
            Tract of the input exposures
        instrumentName : `str`, optional
            Name of the instrument used

        Returns
        -------
        outputCatalog : `list` of `lsst.afw.table.ExposureCatalog`
            List of exposure catalogs (one per visit) with the WCS for each
            detector set by the new fitted WCS
        """
        # Set up an instrument object
        instrument = wcsfit.Instrument(instrumentName)

        # Get RA, Dec, MJD, etc., for the input visits
        exposuresHelper, medianEpoch = self._get_exposure_info(inputVisitSummary, instrument)

        # Get information about the extent of the input visits
        fields, fieldCenter, fieldRadius = self._prep_sky(tract, skyMap, epoch=medianEpoch)

        # Set up class to associate sources into matches using a friends-of-friends algorithm
        associations = wcsfit.FoFClass(fields, [instrument], exposuresHelper,
                                       [fieldRadius.asDegrees()],
                                       (self.config.matchRadius * u.arcsec).to(u.degree).value)

        # Add the reference catalog to the associator
        self._load_refCat(associations, fieldCenter, fieldRadius, epoch=medianEpoch)

        # Add the science catalogs and associate new sources as they are added
        self._load_catalogs_and_associate(associations, inputSourceTableVisitRefs)

        # Set up a YAML-type string using the config variables and a sample visit
        inputYAML = self.makeYAML(inputVisitSummary[0])

        # Set up the WCS-fitting class using the results of the FOF associator
        wcsf = wcsfit.WCSFit(fields, [instrument], exposuresHelper,
                             self.extensionInfo['visitIndex'], self.extensionInfo['deviceIndex'],
                             inputYAML, self.extensionInfo['wcs'], associations.sequence, associations.extn,
                             associations.obj, sysErr=self.config.systematicError,
                             refSysErr=self.config.referenceSystematicError,
                             usePM=self.config.useProperMotion,
                             verbose=self.config.verbose)

        # Add the science and reference sources
        self._addObjects(wcsf, inputSourceTableVisitRefs)
        self._addRefObjects(wcsf)

        # Do the WCS fit
        wcsf.fit()

        import time
        time_str = time.time()

        # Get the output
        # TODO: remove this last thing before pull request
        wcsf.saveResults(f'tmp_output/wcsfit_{time_str}.wcs',
                         f'tmp_output/wcscat_{time_str}.fits',
                         f'tmp_output/starcat_{time_str}.fits')
        outputCatalogs = self._makeOutputs(wcsf, inputVisitSummary[0])
        import ipdb; ipdb.set_trace()
        return outputCatalogs

    def _prep_sky(self, tract, skyMap, epoch=None):
        """Get center and radius of the input tract. This assumes that all
        visits will be put into the same `wcsfit.Field` and fit together.

        Paramaters
        ----------
        tract : int
            Number of the tract in which the exposures were observed
        skyMap : `lsst.skymap.ringsSkyMap.RingsSkyMap`
            Skymap describing the tracts for the data
        epoch : float
            Reference epoch

        Returns
        -------
        fields : `wcsfit.Fields`
            Object with field information
        center : `lsst.geom.SpherePoint`
            Center of the field
        radius : `lsst.sphgeom._sphgeom.Angle`
            Radius of the bounding circle of the tract
        """
        tractInfo = skyMap.generateTract(tract)
        boundingCircle = tractInfo.getOuterSkyPolygon().getBoundingCircle()
        center = lsst.geom.SpherePoint(boundingCircle.getCenter())
        ra = center.getRa().asDegrees()
        dec = center.getDec().asDegrees()
        radius = boundingCircle.getOpeningAngle()

        # wcsfit.Fields describes a list of fields, but we assume all
        # observations will be fit together in one field.
        fields = wcsfit.Fields([str(tract)], [ra], [dec], [epoch])

        return fields, center, radius

    def _get_exposure_info(self, visitSummaryTables, instrument, fieldNumber=0, instrumentNumber=0):
        """Get various information about the input visits to feed to the
        fitting routines.

        Parameters
        ----------
        visitSummaryTables : `list` of `lsst.afw.table.ExposureCatalog`
            Tables for each visit with information for detectors
        instrument : `wcsfit.Instrument`
            Instrument object to which detector information is added
        fieldNumber : `int`
            Index of the field for these visits. Should be zero if all data is
            being fit together.
        instrumentNumber : `int`
            Index of the instrument for these visits. Should be zero if all
            data comes from the same instrument.

        Returns
        -------
        exposuresHelper : `wcsfit.ExposuresHelper`
            Object containing information about the input visits
        medianEpoch : `float`
            Median epoch of the input visits
        """
        exposureNames = []
        self.ras = []
        self.decs = []
        airmasses = []
        exposureTimes = []
        mjds = []
        wcss = []

        extensionType = []
        extensionExposures = []
        extensionDevices = []
        visits = []
        devices = []
        # Get information for all the science visits
        for v, visitSummary in enumerate(visitSummaryTables):

            visitInfo = visitSummary[0].getVisitInfo()

            visit = visitSummary[0]['visit']
            if visit not in self.visits:
                self.visits.append(visit)
            exposureNames.append(str(visit))
            raDec = visitInfo.getBoresightRaDec()
            self.ras.append(raDec.getRa().asRadians())
            self.decs.append(raDec.getDec().asRadians())
            airmasses.append(visitInfo.getBoresightAirmass())
            exposureTimes.append(visitInfo.getExposureTime())
            obsDate = visitInfo.getDate()
            mjds.append(obsDate.get(obsDate.MJD))

            for row in visitSummary:
                detector = row['id']
                if detector not in self.devices:
                    self.devices.append(detector)
                    detectorBounds = wcsfit.Bounds(row['bbox_min_x'], row['bbox_max_x'],
                                                   row['bbox_min_y'], row['bbox_max_y'])
                    instrument.addDevice(str(detector), detectorBounds)

                deviceIndex = np.flatnonzero(detector == np.array(self.devices))[0]
                extensionExposures.append(v)
                extensionDevices.append(deviceIndex)
                visits.append(visit)
                devices.append(detector)
                extensionType.append('SCIENCE')

                wcs = row.getWcs()
                wcss.append(getWCSfromSIP(wcs))

        fieldNumbers = list(np.ones(len(exposureNames), dtype=int) * fieldNumber)
        instrumentNumbers = list(np.ones(len(exposureNames), dtype=int) * instrumentNumber)

        # Set the reference epoch to be the median of the science visits.
        # The reference catalog will be shifted to this date.
        medianEpoch = np.median(mjds)

        # Add information for the reference catalog. Most of the values are
        # not used.
        exposureNames.append("REFERENCE")
        self.visits.append(-1)
        fieldNumbers.append(0)
        instrumentNumbers.append(-1)
        self.ras.append(0.0)
        self.decs.append(0.0)
        airmasses.append(0.0)
        exposureTimes.append(0)
        mjds.append(medianEpoch)
        identity = wcsfit.IdentityMap()
        icrs = wcsfit.SphericalICRS()
        refWcs = wcsfit.Wcs(identity, icrs, "Identity", np.pi / 180.)
        wcss.append(refWcs)

        extensionExposures.append(len(exposureNames) - 1)
        extensionDevices.append(-1)  # REFERENCE device must be -1
        visits.append(-1)
        devices.append(-1)
        extensionType.append('REFERENCE')

        # Make a table of information to use elsewhere in the class
        self.extensionInfo = pd.DataFrame({'visit': visits, 'device': devices,
                                           'visitIndex': extensionExposures,
                                           'deviceIndex': extensionDevices,
                                           'wcs': wcss, 'extensionType': extensionType})

        # Make the exposureHelper object to use in the fitting routines
        exposuresHelper = wcsfit.ExposuresHelper(exposureNames,
                                                 fieldNumbers,
                                                 instrumentNumbers,
                                                 self.ras,
                                                 self.decs,
                                                 airmasses,
                                                 exposureTimes,
                                                 mjds)

        return exposuresHelper, medianEpoch

    def _load_refCat(self, associations, center, radius, epoch=None, fieldIndex=0, instrumentIndex=-1):
        """Load the reference catalog and add reference objects to the
        `wcsfit.FoFClass` object.

        Parameters
        ----------
        associations : `wcsfit.FoFClass`
            Object to which to add the catalog of reference objects
        center : `lsst.geom.SpherePoint`
            Center of the circle in which to load reference objects
        radius : `lsst.sphgeom._sphgeom.Angle`
            Radius of the circle in which to load reference objects
        epoch : `float`
            MJD to which to correct the object positions
        """
        formattedEpoch = astropy.time.Time(epoch, format='decimalyear')

        refFilter = self.config.astrometryRefObjLoader.anyFilterMapsToThis
        skyCircle = self.astrometryRefObjLoader.loadSkyCircle(center, radius, refFilter, epoch=formattedEpoch)

        selected = self.astrometryReferenceSelector.run(skyCircle.refCat)
        # Need memory contiguity to get reference filters as a vector.
        if not selected.sourceCat.isContiguous():
            refCat = selected.sourceCat.copy(deep=True)
        else:
            refCat = selected.sourceCat

        ra = (refCat['coord_ra'] * u.radian).to(u.degree).to_value().tolist()
        dec = (refCat['coord_dec'] * u.radian).to(u.degree).to_value().tolist()
        raCov = ((refCat['coord_raErr'] * u.radian).to(u.degree).to_value()**2).tolist()
        decCov = ((refCat['coord_decErr'] * u.radian).to(u.degree).to_value()**2).tolist()

        # TODO: DM-35130 we need the full gaia covariance here
        self.refObjects = {'ra': ra, 'dec': dec, 'raCov': raCov, 'decCov': decCov,
                           'raDecCov': np.zeros(len(ra))}

        extensionIndex = self.extensionInfo[self.extensionInfo['extensionType'] == 'REFERENCE'].index[0]
        visitIndex = self.extensionInfo.iloc[extensionIndex]['visitIndex']
        detectorIndex = self.extensionInfo.iloc[extensionIndex]['deviceIndex']
        refWcs = self.extensionInfo.iloc[extensionIndex]['wcs']

        associations.addCatalog(refWcs, "STELLAR", visitIndex, fieldIndex, instrumentIndex, detectorIndex,
                                extensionIndex, np.ones(len(refCat), dtype=bool),
                                ra, dec, np.arange(len(ra)))

    def _load_catalogs_and_associate(self, associations, inputSourceTableVisitRefs,
                                     fieldIndex=0, instrumentIndex=0):
        """Load the science catalogs and add the sources to the associator
        class `wcsfit.FoFClass`, associating them into matches as you go.

        Parameters
        ----------
        associations : `wcsfit.FoFClass`
            Object to which to add the catalog of reference objects
        inputSourceTableVisitRefs : `list`
            List of DeferredDatasetHandles pointing to visit-level source
            tables
        fieldIndex : `int`
            Index of the field for these catalogs. Should be zero assuming all
            data is being fit together.
        instrumentIndex : `int`
            Index of the instrument for these catalogs. Should be zero
            assuming all data comes from the same instrument.
        """
        self.columns = ['detector', 'sourceId', 'x', 'xErr', 'y', 'yErr',
                        f"{self.config.sourceFluxType}_instFlux", f"{self.config.sourceFluxType}_instFluxErr"]
        if self.sourceSelector.config.doFlags:
            self.columns.extend(self.sourceSelector.config.flags.bad)
        if self.sourceSelector.config.doUnresolved:
            self.columns.append(self.sourceSelector.config.unresolved.name)
        if self.sourceSelector.config.doIsolated:
            self.columns.append(self.sourceSelector.config.isolated.parentName)
            self.columns.append(self.sourceSelector.config.isolated.nChildName)

        self.sourceIndices = [None] * len(self.extensionInfo)
        for inputSourceTableVisitRef in inputSourceTableVisitRefs:
            visit = inputSourceTableVisitRef.dataId['visit']
            vis_index = np.flatnonzero(np.array(self.visits) == visit)[0]
            sourceTable = inputSourceTableVisitRef.get(parameters={'columns': self.columns})
            detectors = set(sourceTable['detector'])

            for detector in detectors:
                detectorSources = sourceTable[sourceTable['detector'] == detector]
                selected = self.sourceSelector.run(detectorSources)

                isStar = np.ones(len(selected))

                extensionIndex = self.extensionInfo[(self.extensionInfo['visit'] == visit)
                                                    & (self.extensionInfo['device'] == detector)].index[0]
                detectorIndex = self.extensionInfo.iloc[extensionIndex]['deviceIndex']
                visitIndex = self.extensionInfo.iloc[extensionIndex]['visitIndex']
                assert vis_index == visitIndex

                self.sourceIndices[extensionIndex] = selected.selected

                wcs = self.extensionInfo['wcs'][extensionIndex]
                associations.reprojectWCS(wcs, fieldIndex)

                associations.addCatalog(wcs, "STELLAR", vis_index, fieldIndex,
                                        instrumentIndex, detectorIndex, extensionIndex, isStar,
                                        selected.sourceCat['x'].to_list(), selected.sourceCat['y'].to_list(),
                                        np.arange(len(selected.sourceCat)))

        associations.sortMatches(fieldIndex, minMatches=self.config.minMatches,
                                 allowSelfMatches=self.config.allowSelfMatches)

    def makeYAML(self, inputVisitSummary, inputFile=None):
        """Make a YAML-type object that describes the parameters of the fit
        model.

        Parameters
        ----------
        inputVisitSummary : `list` of `lsst.afw.table.ExposureCatalog`
            List of catalogs with per-detector summary information
        inputFile : `str`
            Path to a file that contains a basic model

        Returns
        -------
        inputYAML : `wcsfit.YAMLCollector`
            YAML object containing the model description
        """
        if inputFile is not None:
            inputYAML = wcsfit.YAMLCollector(inputFile, "PixelMapCollection")
        else:
            inputYAML = wcsfit.YAMLCollector("", "PixelMapCollection")
        inputDict = {}
        modelComponents = ["INSTRUMENT/DEVICE", "EXPOSURE"]
        baseMap = {"Type": "Composite", "Elements": modelComponents}
        inputDict["EXPOSURE/DEVICE/base"] = baseMap

        xMin = str(inputVisitSummary['bbox_min_x'].min())
        xMax = str(inputVisitSummary['bbox_max_x'].max())
        yMin = str(inputVisitSummary['bbox_min_y'].min())
        yMax = str(inputVisitSummary['bbox_max_y'].max())

        deviceModel = {"Type": "Composite", "Elements": self.config.deviceModel.list()}
        inputDict["INSTRUMENT/DEVICE"] = deviceModel
        for component in self.config.deviceModel:
            if "poly" in component.lower():
                componentDict = {"Type": "Poly",
                                 "XPoly": {"OrderX": self.config.devicePolyOrder,
                                           "SumOrder": True},
                                 "YPoly": {"OrderX": self.config.devicePolyOrder,
                                           "SumOrder": True},
                                 "XMin": xMin, "XMax": xMax, "YMin": yMin, "YMax": yMax}
            elif "identity" in component.lower():
                componentDict = {"Type": "Identity"}
            else:
                raise ValueError(f"deviceModel component {component} is not supported")
            inputDict[component] = componentDict

        exposureModel = {"Type": "Composite", "Elements": self.config.exposureModel.list()}
        inputDict["EXPOSURE"] = exposureModel
        for component in self.config.exposureModel:
            if "poly" in component.lower():
                componentDict = {"Type": "Poly",
                                 "XPoly": {"OrderX": self.config.exposurePolyOrder,
                                           "SumOrder": "true"},
                                 "YPoly": {"OrderX": self.config.exposurePolyOrder,
                                           "SumOrder": "true"}}
            elif "identity" in component.lower():
                componentDict = {"Type": "Identity"}
            else:
                raise ValueError(f"exposureModel component {component} is not supported")
            inputDict[component] = componentDict

        inputYAML.addInput(yaml.dump(inputDict))
        inputYAML.addInput("Identity:\n  Type:  Identity\n")

        return inputYAML

    def _addObjects(self, wcsf, inputSourceTableVisitRefs):
        """Add science sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object
        inputSourceTableVisitRefs : `list`
            List of DeferredDatasetHandles pointing to visit-level source
            tables
        """
        for inputSourceTableVisitRef in inputSourceTableVisitRefs:
            visit = inputSourceTableVisitRef.dataId['visit']

            sourceTable = inputSourceTableVisitRef.get(parameters={'columns': self.columns})
            detectors = set(sourceTable['detector'])

            for detector in detectors:
                detectorSources = sourceTable[sourceTable['detector'] == detector]
                selected = self.sourceSelector.run(detectorSources)

                extensionIndex = self.extensionInfo[(self.extensionInfo['visit'] == visit)
                                                    & (self.extensionInfo['device'] == detector)].index[0]

                d = {"x": selected.sourceCat['x'].to_numpy(),
                     "y": selected.sourceCat['y'].to_numpy(),
                     "xCov": selected.sourceCat['xErr'].to_numpy()**2,
                     "yCov": selected.sourceCat['yErr'].to_numpy()**2,
                     "xyCov": np.zeros(len(selected.sourceCat))}
                # TODO: add correct xyErr if DM-7101 is ever done

                wcsf.setObjects(extensionIndex, d, 'x', 'y', ['xCov', 'yCov', 'xyCov'])

    def _addRefObjects(self, wcsf):
        """Add reference sources to the wcsfit.WCSFit object.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCS-fitting object
        """
        extensionIndex = self.extensionInfo[self.extensionInfo['extensionType'] == 'REFERENCE'].index[0]

        wcsf.setObjects(extensionIndex, self.refObjects, 'ra', 'dec', ['raCov', 'decCov', 'raDecCov'])

    def _makeOutputs(self, wcsf, inputVisitSummary):
        """Make a WCS object out of the WCS models.

        Parameters
        ----------
        wcsf : `wcsfit.WCSFit`
            WCSFit object, assumed to have fit model
        inputVisitSummary : `lsst.afw.table.ExposureCatalog`
            Catalogs with per-detector summary information from which to grab
            detector information

        Returns
        -------
        catalogs : `dict`
            Dictionary of `lsst.afw.table.ExposureCatalog` objects with the WCS
            set to the WCS fit in wcsf
        """
        # Get the parameters of the fit models
        mapParams = wcsf.mapCollection.getParamDict()

        # Set up the schema for the output catalogs
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        schema.addField('visit', type='L', doc='Visit number')

        # Set up pixel frames
        pixelFrame = astshim.Frame(2, "Domain=PIXELS")
        normedPixelFrame = astshim.Frame(2, "Domain=NORMEDPIXELS")

        # Pixels will need to be rescaled before going into the mappings
        xscale = inputVisitSummary[0]['bbox_max_x'] - inputVisitSummary[0]['bbox_min_x']
        yscale = inputVisitSummary[0]['bbox_max_y'] - inputVisitSummary[0]['bbox_min_y']
        normCoefficients = [-1.0, 2.0/xscale, 0,
                            -1.0, 0, 2.0/yscale]

        normMap = convertToAstPolyMapCoefficients(normCoefficients)

        catalogs = {}
        for v, visit in enumerate(self.visits):
            if visit == -1:
                continue
            catalog = lsst.afw.table.ExposureCatalog(schema)
            catalog.resize(len(self.devices))
            catalog['visit'] = visit

            # All of the detectors for one visit map to the same tangent plane
            tangentPoint = lsst.geom.SpherePoint(self.ras[v], self.decs[v], lsst.geom.radians)
            cdMatrix = afwgeom.makeCdMatrix(1.0 * lsst.geom.degrees, 0 * lsst.geom.degrees, True)
            iwcToSkyWcs = afwgeom.makeSkyWcs(lsst.geom.Point2D(0, 0), tangentPoint, cdMatrix)
            iwcToSkyMap = iwcToSkyWcs.getFrameDict().getMapping("PIXELS", "SKY")
            skyFrame = iwcToSkyWcs.getFrameDict().getFrame("SKY")

            for d, detector in enumerate(self.devices):
                mapName = f"{visit}/{detector}"

                frameDict = astshim.FrameDict(pixelFrame)

                frameDict.addFrame("PIXELS", normMap, normedPixelFrame)

                mapElements = wcsf.mapCollection.orderAtoms(f"{mapName}/base")

                currentFrameName = "NORMEDPIXELS"
                for m, mapElement in enumerate(mapElements):
                    mapType = wcsf.mapCollection.getMapType(mapElement)

                    if mapType == 'Poly':
                        mapCoefficients = mapParams[mapElement]
                        astMap = convertToAstPolyMapCoefficients(mapCoefficients)
                    elif mapType == 'Identity':
                        astMap = astshim.UnitMap(2)
                    else:
                        raise ValueError(f"Converting map type {mapType} to WCS is not supported")

                    if m == len(mapElements) - 1:
                        newFrameName = "IWC"
                    else:
                        newFrameName = "INTERMEDIATE" + str(m)
                    newFrame = astshim.Frame(2, f"Domain={newFrameName}")
                    frameDict.addFrame(currentFrameName, astMap, newFrame)
                    currentFrameName = newFrameName
                frameDict.addFrame("IWC", iwcToSkyMap, skyFrame)

                outWCS = afwgeom.SkyWcs(frameDict)

                catalog[d].setId(detector)
                catalog[d].setWcs(outWCS)
            catalog.sort()
            catalogs[visit] = catalog

        return catalogs
