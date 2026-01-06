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

__all__ = ["FitStellarMotionConfig", "FitStellarMotionConnections", "FitStellarMotionTask"]

import astropy.coordinates
import astropy.units as u
import numpy as np
import wcsfit
from astropy.table import Table, hstack, join, vstack

import lsst.afw.geom as afwGeom
import lsst.geom
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.skymap import BaseSkyMap


class FitStellarMotionConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=(
        "instrument",
        "tract",
        "skymap",
    ),
):
    visitSummaries = pipeBase.connectionTypes.Input(
        doc=(
            "Per-visit consolidated exposure metadata built from calexps. "
            "These catalogs use detector id for the id and must be sorted for "
            "fast lookups of a detector."
        ),
        name="preliminary_visit_summary",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )
    starSourceRef = pipeBase.connectionTypes.Input(
        doc="Catalog of matched sources.",
        name="isolated_star",
        storageClass="ArrowAstropy",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
        deferLoad=True,
    )
    starCatalogRef = pipeBase.connectionTypes.Input(
        doc="Catalog of objects corresponding to the matched sources.",
        name="isolated_star_association",
        storageClass="ArrowAstropy",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
        deferLoad=True,
    )
    inputSources = pipeBase.connectionTypes.Input(
        doc="Source table in parquet format, per visit.",
        name="recalibrated_star",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        deferLoad=True,
        multiple=True,
    )
    referenceCatalog = pipeBase.connectionTypes.PrerequisiteInput(
        doc="The astrometry reference catalog to match to loaded input catalog sources.",
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )
    skymap = pipeBase.connectionTypes.Input(
        doc="Input definition of bbox containing the associated sources.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    visitTable = pipeBase.connectionTypes.Input(
        doc="Survey-wide table of visits, which will be used to get median epoch.",
        name="preliminary_visit_table",
        storageClass="ArrowAstropy",
        dimensions=("instrument",),
        deferLoad=True,
    )
    outputCatalog = pipeBase.connectionTypes.Output(
        doc="Best fit position, proper motion and parallax for input objects.",
        name="isolated_star_stellar_motions",
        storageClass="ArrowAstropy",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
    )
    predictedPositions = pipeBase.connectionTypes.Output(
        doc="Predicted position for each source at the epoch of observation.",
        name="isolated_star_predicted_positions",
        storageClass="ArrowAstropy",
        dimensions=(
            "instrument",
            "skymap",
            "tract",
        ),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not self.config.includeReferenceCatalog:
            self.inputs.remove("referenceCatalog")
            self.inputs.remove("skymap")
        if self.config.outputEpoch:
            self.inputs.remove("visitTable")


class FitStellarMotionConfig(pipeBase.PipelineTaskConfig, pipelineConnections=FitStellarMotionConnections):
    includeReferenceCatalog = pexConfig.Field(
        doc="Include the reference catalog in the fit.",
        dtype=bool,
        default=True,
    )
    referenceFilter = pexConfig.Field(
        dtype=str,
        doc="Name of filter to load from reference catalog. This is a required argument, although the values"
        "returned are not used.",
        default="phot_g_mean",
    )
    referenceMatchRadius = pexConfig.Field(
        dtype=float,
        doc="Maximum matching distance in arcseconds between the star catalog and the reference catalog.",
        default=0.1,
    )
    outputEpoch = pexConfig.Field(
        dtype=float,
        doc="Epoch to which output positions will correspond. If not set, the median epoch of all visits in "
        "visitTable will be used.",
        default=None,
        optional=True,
    )


class FitStellarMotionTask(pipeBase.PipelineTask):
    """Fit proper motion and parallax for associated sources.

    Input sources are assumed to be isolated point sources.
    """

    ConfigClass = FitStellarMotionConfig
    _DefaultName = "fitStellarMotions"

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        # Override runQuantum to set up the refObjLoaders and turn input lists
        # into dicts.
        inputs = butlerQC.get(inputRefs)

        inputSourceDict = {inputSource.dataId["visit"]: inputSource for inputSource in inputs["inputSources"]}
        inputs["inputSources"] = inputSourceDict
        visitSummaryDict = {
            visitSummary.dataId["visit"]: visitSummary for visitSummary in inputs["visitSummaries"]
        }
        inputs["visitSummaries"] = visitSummaryDict

        if self.config.includeReferenceCatalog:
            tractId = inputs["starCatalogRef"].dataId["tract"]
            skymap = inputs.pop("skymap")
            tractRegion = skymap.generateTract(tractId).outer_sky_polygon

            refConfig = LoadReferenceObjectsConfig()
            refConfig.requireProperMotion = True
            refObjectLoader = ReferenceObjectLoader(
                dataIds=[ref.datasetRef.dataId for ref in inputRefs.referenceCatalog],
                refCats=inputs.pop("referenceCatalog"),
                config=refConfig,
                log=self.log,
            )
        else:
            refObjectLoader = None
            tractRegion = None

        if self.config.outputEpoch:
            epoch = astropy.time.Time(self.config.outputEpoch, format="mjd")
        else:
            # Use the median epoch of all visits in the survey.
            visitTable = inputs.pop("visitTable")
            allVisits = visitTable.get(parameters={"columns": ["expMidptMJD"]})
            epoch = astropy.time.Time(np.median(allVisits["expMidptMJD"]), format="mjd")

        output = self.run(**inputs, epoch=epoch, refObjectLoader=refObjectLoader, tractRegion=tractRegion)

        butlerQC.put(output.outputCatalog, outputRefs.outputCatalog)
        butlerQC.put(output.predictedPositions, outputRefs.predictedPositions)

    def run(
        self,
        starSourceRef,
        inputSources,
        starCatalogRef,
        visitSummaries,
        epoch,
        refObjectLoader=None,
        tractRegion=None,
    ):
        """Fit proper motion and parallax for isolated stars.

        Parameters
        ----------
        starSourceRef : `DeferredDatasetHandle`
            Handle pointing to catalog of associated sources.
        inputSources : `dict` [`int`, `DeferredDatasetHandle`]
            Dictionary of source catalog handles, keyed by their visit id.
        starCatalogRef : `DeferredDatasetHandle`
            Handle pointing to catalog of objects corresponding to associated
            sources.
        visitSummaries : `dict` [`int`, `lsst.afw.table.ExposureCatalog`]
            Dictionary of catalogs with per-detector summary information, keyed
            by their visit id.
        epoch : `float`
            Epoch in MJD at which to fit positions of objects.
        refObjectLoader :
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`,
            optional
            Reference object loader
        tractRegion : `lsst.sphgeom.Region`
            Region containing the associated sources.

        Returns
        -------
        result : `lsst.pipe.Base.Struct`
            ``outputCatalog`` : `astropy.table.Table`
                Catalog with postion, proper motion and parallax for all input
                objects, with NAN for objects without enough data to fit
                parameters.
            ``predictedPositions`` : `astropy.table.Table`
                Catalog with predicted positions for all input sources at their
                epoch observation, with NAN for objects with insufficient data.
        """
        # Load needed columns for associated sources.
        starSources = starSourceRef.get(parameters={"columns": ["visit", "sourceId", "obj_index"]})
        if not starSources:
            raise pipeBase.NoWorkFound("No isolated stars found in this region.")

        starSources.add_index("sourceId")

        # Load reference objects.
        if self.config.includeReferenceCatalog:
            refCatalog = self._load_refCat(refObjectLoader, tractRegion, epoch)
        else:
            refCatalog = None

        # Load needed columns from source catalogs and get visit info.
        visitStars, visitInfo = self._load_sources(starSources, visitSummaries, inputSources)

        # Fit postion, proper motion and parallax for all objects.
        outCat, predictedRADec = self._fit_objects(
            visitStars, starCatalogRef, starSources, visitInfo, epoch, refCatalog=refCatalog
        )

        return pipeBase.Struct(outputCatalog=outCat, predictedPositions=predictedRADec)

    def _load_refCat(self, refObjectLoader, region, epoch):
        """Load reference catalog.

        Parameters
        ----------
        refObjectLoader :
            `lsst.meas.algorithms.loadReferenceObjects.ReferenceObjectLoader`
            Reference object loader
        tractRegion : `lsst.sphgeom.Region`
            Region containing the associated sources.
        epoch : `astropy.time.Time`
            Epoch to which the reference catalog will be shifted.

        Returns
        -------
        refCatalog : `astropy.table.Table`
            Catalog of reference objects.
        """

        refCat = refObjectLoader.loadRegion(region, self.config.referenceFilter, epoch=epoch).refCat
        refCat = refCat.asAstropy()

        # In Gaia DR3, missing values are denoted by NaNs.
        finiteInd = np.isfinite(refCat["coord_ra"]) & np.isfinite(refCat["coord_dec"])
        refCat = refCat[finiteInd]

        ra = (refCat["coord_ra"]).to(u.degree)
        dec = (refCat["coord_dec"]).to(u.degree)
        raPM = (refCat["pm_ra"]).to(u.marcsec)
        decPM = (refCat["pm_dec"]).to(u.marcsec)
        parallax = (refCat["parallax"]).to(u.marcsec)

        cov = np.zeros((len(refCat), 5, 5))
        positionParameters = ["coord_ra", "coord_dec", "pm_ra", "pm_dec", "parallax"]
        for i, pi in enumerate(positionParameters):
            for j, pj in enumerate(positionParameters):
                if i == j:
                    cov[:, i, i] = ((refCat[f"{pi}Err"].value) ** 2 * u.radian**2).to(u.marcsec**2).value
                elif i > j:
                    cov[:, i, j] = (refCat[f"{pj}_{pi}_Cov"].value * u.radian**2).to_value(u.marcsec**2)
                else:
                    cov[:, i, j] = (refCat[f"{pi}_{pj}_Cov"].value * u.radian**2).to_value(u.marcsec**2)
        refCatalog = Table(
            {"ra": ra, "dec": dec, "raPM": raPM, "decPM": decPM, "parallax": parallax, "covariance": cov}
        )
        return refCatalog

    def _load_sources(self, starSources, visitSummaries, inputSources):
        """Load isolated sources and get visit information.

        Parameters
        ----------
        starSources : `astropy.table.Table`
            Catalog of associated sources.
        visitSummaries : `dict` [`int`, `lsst.afw.table.ExposureCatalog`]
            Dictionary of catalogs with per-detector summary information keyed
            by their visit id.
        inputSources : `dict` [`int`, `DeferredDatasetHandle`]
            Dictionary of source catalog handles, keyed by their visit id.

        Returns
        -------
        allVisitStars : `astropy.table.Table`
            Catalog with all needed information for associated sources.
        visitInfo : `astropy.table.Table`
            Catalog with observation epoch and location in ICRS coordinates.
        """
        visits = np.unique(starSources["visit"])
        visits.sort()
        observatories = []
        mjds = []
        allVisitStars = []
        finalVisits = []
        for visit in visits:
            if (visit not in visitSummaries) or (visit not in inputSources):
                continue

            visitSummary = visitSummaries[visit].get()
            finalVisits.append(visit)
            visitInfo = visitSummary[0].visitInfo

            # Get MJD
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
            observatory = observatory_icrs.cartesian.xyz.to(u.AU).value
            observatories.append(observatory)

            # Load sources and keep isolated ones.
            visitSources = inputSources[visit].get(
                parameters={
                    "columns": [
                        "sourceId",
                        "ra",
                        "dec",
                        "raErr",
                        "decErr",
                        "ra_dec_Cov",
                    ]
                }
            )
            visitStars = join(
                visitSources,
                starSources[starSources["visit"] == visit],
                keys="sourceId",
                join_type="inner",
            )
            allVisitStars.append(visitStars)
        allVisitStars = vstack(allVisitStars)
        visitInfo = Table({"visit": finalVisits, "observatory": observatories, "mjd": mjds})
        visitInfo.add_index("visit")

        return allVisitStars, visitInfo

    def _fit_objects(self, visitStars, starCatalogRef, starSources, visitInfo, fitEpoch, refCatalog=None):
        """Fit full 5-d position, proper motion, and parallax for associated
        sources.

        Parameters
        ----------
        visitStars : `astropy.table.Table`
            Catalog with position information for associated sources.
        starCatalogRef : `DeferredDatasetHandle`
            Handle pointing to catalog of objects corresponding to associated
            sources.
        starSources : `astropy.table.Table`
            Catalog of associated sources.
        visitInfo : `astropy.table.Table`
            Catalog with observation epoch and location in ICRS coordinates.
        fitEpoch : `astropy.time.Time`
            Epoch at which to fit positions of objects.
        refCatalog : `astropy.table.Table`, optional
            Catalog of reference objects. Used if
            self.config.includeReferenceCatalog is true.

        Returns
        -------
        outCat : `astropy.table.Table`
            Catalog with postion, proper motion and parallax for all input
            objects, with NAN for objects without enough data to fit
            parameters.
        predictedPositions : `astropy.table.Table`
            Catalog with predicted positions for all input sources at their
            epoch observation, with NAN for objects with insufficient data.
        """

        starCatalog = starCatalogRef.get(parameters={"columns": ["isolated_star_id", "ra", "dec"]})

        if self.config.includeReferenceCatalog:
            starCoord = astropy.coordinates.SkyCoord(
                starCatalog["ra"] * u.degree, starCatalog["dec"] * u.degree
            )
            refCoord = astropy.coordinates.SkyCoord(refCatalog["ra"], refCatalog["dec"])
            refId, refD2d, _ = starCoord.match_to_catalog_sky(refCoord)

            identity = wcsfit.IdentityMap()
            icrs = wcsfit.SphericalICRS()
            refWcs = wcsfit.Wcs(identity, icrs, "Identity", np.pi / 180.0)

        objects = np.unique(visitStars["obj_index"])
        objects.sort()

        # Make empty arrays to fill in, with NaN for any unfittable objects.
        objectPositions = np.ones((len(starCatalog), 5)) * np.nan
        objectCovariances = np.ones((len(starCatalog), 5, 5)) * np.nan
        predictedRADec = np.ones((len(starSources), 2)) * np.nan
        includesReference = np.zeros(len(starCatalog), dtype=bool)
        nSources = np.zeros(len(starCatalog), dtype=int)
        refPositions = Table(
            np.ones((len(starCatalog), 5)) * np.nan,
            names=("ref_ra", "ref_dec", "ref_raPM", "ref_decPM", "ref_covariance"),
            dtype=("f8", "f8", "f8", "f8", "f8"),
        )
        refCovariances = np.ones((len(starCatalog), 5, 5)) * np.nan
        for object in objects:
            # Get all detections for this object.
            detectionInds = visitStars["obj_index"] == object
            detections = visitStars[detectionInds]
            nDetections = len(detections)
            scienceDetections = np.ones(len(detections), dtype=bool)

            objectObservatories = visitInfo.loc[detections["visit"]]["observatory"]
            objectMjds = visitInfo.loc[detections["visit"]]["mjd"]

            # Move detections to be tangent plane around median position.
            medRA = np.median(detections["ra"])
            medDec = np.median(detections["dec"])
            tangentPoint = lsst.geom.SpherePoint(medRA, medDec, lsst.geom.degrees)
            cdMatrix = afwGeom.makeCdMatrix(1.0 * lsst.geom.degrees, 0 * lsst.geom.degrees, True)
            iwcToSkyWcs = afwGeom.makeSkyWcs(lsst.geom.Point2D(0, 0), tangentPoint, cdMatrix)
            tanX, tanY = iwcToSkyWcs.skyToPixelArray(detections["ra"], detections["dec"], degrees=True)

            match = wcsfit.PMMatch(
                tanX,
                tanY,
                detections["raErr"] ** 2,
                detections["decErr"] ** 2,
                detections["ra_dec_Cov"],
                objectMjds,
                objectObservatories,
                medRA,
                medDec,
                fitEpoch.mjd,
            )

            if self.config.includeReferenceCatalog and (
                refD2d[object].arcsecond < self.config.referenceMatchRadius
            ):
                nDetections += 1
                refMatch = refCatalog[refId[object]]
                match.addPMDetection(
                    refMatch["ra"],
                    refMatch["dec"],
                    refMatch["raPM"],
                    refMatch["decPM"],
                    refMatch["parallax"],
                    refMatch["covariance"],
                    refWcs,
                )
                scienceDetections = np.append(scienceDetections, False)
                includesReference[object] = True
                refPositions[object] = refMatch[["ra", "dec", "raPM", "decPM", "parallax"]]
                refCovariances[object] = refMatch["covariance"]

            elif nDetections < 3:
                # If there is no associated reference object, there must be at
                # least three detections in order to fit the 5-d solution.
                continue

            # Solve, get best-fit position and covariance, and prediction for
            # the object position at the detection epochs.
            match.solve()
            fullPosition = match.getFit()
            objectPositions[object] = fullPosition
            objectCovariances[object] = match.getFitCovariance()
            nSources[object] = nDetections
            predictedPositions = match.predictAtDetections()
            predictedRADec[starSources.loc_indices[detections["sourceId"]]] = predictedPositions[
                scienceDetections
            ]

        outCat = Table(objectPositions, names=("ra", "dec", "raPM", "decPM", "parallax"))
        outCat["hasReference"] = includesReference
        outCat["covariance"] = objectCovariances
        outCat = hstack([outCat, refPositions])
        outCat["ref_covariance"] = refCovariances
        outCat["isolated_star_id"] = starCatalog["isolated_star_id"]
        outCat.meta["epoch"] = fitEpoch

        predictedRADec = Table(predictedRADec, names=("ra", "dec"))
        predictedRADec["sourceId"] = starSources["sourceId"]

        return outCat, predictedRADec
