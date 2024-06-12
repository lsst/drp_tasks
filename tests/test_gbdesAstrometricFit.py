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
import unittest
from copy import copy

import astropy.units as u
import lsst.afw.geom as afwgeom
import lsst.afw.table as afwTable
import lsst.geom
import lsst.utils
import numpy as np
import pandas as pd
import wcsfit
import yaml
from lsst import sphgeom
from lsst.daf.base import PropertyList
from lsst.drp.tasks.gbdesAstrometricFit import (
    GbdesAstrometricFitConfig,
    GbdesAstrometricFitTask,
    GbdesGlobalAstrometricFitConfig,
    GbdesGlobalAstrometricFitTask,
)
from lsst.meas.algorithms import ReferenceObjectLoader
from lsst.meas.algorithms.testUtils import MockRefcatDataId
from lsst.pipe.base import InMemoryDatasetHandle
from smatch.matcher import Matcher

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class TestGbdesAstrometricFit(lsst.utils.tests.TestCase):
    """This class tests `GbdesAstrometricFit` using real `visitSummaryTable`s
    from HSC RC2 processing, with simulated sources for those visits.
    """

    @classmethod
    def setUpClass(cls):
        # Set random seed
        np.random.seed(1234)

        # Make fake data
        cls.datadir = os.path.join(TESTDIR, "data")

        cls.fieldNumber = 0
        cls.nFields = 1
        cls.instrumentName = "HSC"
        cls.refEpoch = 57205.5

        # Make test inputVisitSummary. VisitSummaryTables are taken from
        # collection HSC/runs/RC2/w_2022_20/DM-34794
        cls.testVisits = [1176, 17900, 17930, 17934]
        cls.inputVisitSummary = []
        for testVisit in cls.testVisits:
            visSum = afwTable.ExposureCatalog.readFits(
                os.path.join(cls.datadir, f"visitSummary_{testVisit}.fits")
            )
            cls.inputVisitSummary.append(visSum)

        cls.config = GbdesAstrometricFitConfig()
        cls.config.systematicError = 0
        cls.config.devicePolyOrder = 4
        cls.config.exposurePolyOrder = 6
        cls.config.fitReserveFraction = 0
        cls.config.fitReserveRandomSeed = 1234
        cls.config.saveModelParams = True
        cls.config.allowSelfMatches = True
        cls.config.saveCameraModel = True
        cls.task = GbdesAstrometricFitTask(config=cls.config)

        (
            cls.exposureInfo,
            cls.exposuresHelper,
            cls.extensionInfo,
            cls.instruments,
        ) = cls.task._get_exposure_info(cls.inputVisitSummary, refEpoch=cls.refEpoch)

        cls.fields, cls.fieldCenter, cls.fieldRadius = cls.task._prep_sky(
            cls.inputVisitSummary, cls.exposureInfo.medianEpoch
        )

        # Bounding box of observations:
        raMins, raMaxs = [], []
        decMins, decMaxs = [], []
        for visSum in cls.inputVisitSummary:
            raMins.append(visSum["raCorners"].min())
            raMaxs.append(visSum["raCorners"].max())
            decMins.append(visSum["decCorners"].min())
            decMaxs.append(visSum["decCorners"].max())
        raMin = min(raMins)
        raMax = max(raMaxs)
        decMin = min(decMins)
        decMax = max(decMaxs)

        corners = [
            lsst.geom.SpherePoint(raMin, decMin, lsst.geom.degrees).getVector(),
            lsst.geom.SpherePoint(raMax, decMin, lsst.geom.degrees).getVector(),
            lsst.geom.SpherePoint(raMax, decMax, lsst.geom.degrees).getVector(),
            lsst.geom.SpherePoint(raMin, decMax, lsst.geom.degrees).getVector(),
        ]
        cls.boundingPolygon = sphgeom.ConvexPolygon(corners)

        # Make random set of data in a bounding box determined by input visits
        # Make wcs objects for the "true" model
        cls.nStars, starIds, starRAs, starDecs = cls._make_simulated_stars(
            raMin, raMax, decMin, decMax, cls.config.matchRadius
        )

        # Fraction of simulated stars in the reference catalog and science
        # exposures
        inReferenceFraction = 1
        inScienceFraction = 1

        # Make a reference catalog and load it into ReferenceObjectLoader
        refDataId, deferredRefCat = cls._make_refCat(
            starIds, starRAs, starDecs, inReferenceFraction, cls.boundingPolygon
        )
        cls.refObjectLoader = ReferenceObjectLoader([refDataId], [deferredRefCat])
        cls.refObjectLoader.config.requireProperMotion = False
        cls.refObjectLoader.config.anyFilterMapsToThis = "test_filter"

        cls.task.refObjectLoader = cls.refObjectLoader

        # Get True WCS for stars:
        with open(os.path.join(cls.datadir, "sample_wcs.yaml"), "r") as f:
            cls.trueModel = yaml.load(f, Loader=yaml.Loader)

        trueWCSs = cls._make_wcs(cls.trueModel, cls.inputVisitSummary)

        # Make source catalogs:
        cls.inputCatalogRefs = cls._make_sourceCat(starIds, starRAs, starDecs, trueWCSs, inScienceFraction)

        cls.outputs = cls.task.run(
            cls.inputCatalogRefs,
            cls.inputVisitSummary,
            instrumentName=cls.instrumentName,
            refEpoch=cls.refEpoch,
            refObjectLoader=cls.refObjectLoader,
        )

    @staticmethod
    def _make_simulated_stars(raMin, raMax, decMin, decMax, matchRadius, nStars=10000):
        """Generate random positions for "stars" in an RA/Dec box.

        Parameters
        ----------
        raMin : `float`
            Minimum RA for simulated stars.
        raMax : `float`
            Maximum RA for simulated stars.
        decMin : `float`
            Minimum Dec for simulated stars.
        decMax : `float`
            Maximum Dec for simulated stars.
        matchRadius : `float`
            Minimum allowed distance in arcsec between stars.
        nStars : `int`, optional
            Number of stars to simulate. Final number will be lower if any
            too-close stars are dropped.

        Returns
        -------
        nStars : `int`
            Number of stars simulated.
        starIds: `np.ndarray`
            Unique identification number for stars.
        starRAs: `np.ndarray`
            Simulated Right Ascensions.
        starDecs: `np.ndarray`
            Simulated Declination.
        """
        starIds = np.arange(nStars)
        starRAs = np.random.random(nStars) * (raMax - raMin) + raMin
        starDecs = np.random.random(nStars) * (decMax - decMin) + decMin
        # Remove neighbors:
        with Matcher(starRAs, starDecs) as matcher:
            idx = matcher.query_groups(matchRadius / 3600.0, min_match=2)
        if len(idx) > 0:
            neighbors = np.unique(np.concatenate(idx))
            starRAs = np.delete(starRAs, neighbors)
            starDecs = np.delete(starDecs, neighbors)
            nStars = len(starRAs)
            starIds = np.arange(nStars)

        return nStars, starIds, starRAs, starDecs

    @classmethod
    def _make_refCat(cls, starIds, starRas, starDecs, inReferenceFraction, bounds):
        """Make reference catalog from a subset of the simulated data

        Parameters
        ----------
        starIds : `np.ndarray` [`int`]
            Source ids for the simulated stars
        starRas : `np.ndarray` [`float`]
            RAs of the simulated stars
        starDecs : `np.ndarray` [`float`]
            Decs of the simulated stars
        inReferenceFraction : float
            Percentage of simulated stars to include in reference catalog
        bounds : `lsst.sphgeom.ConvexPolygon`
            Boundary of the reference catalog region

        Returns
        -------
        refDataId : `lsst.meas.algorithms.testUtils.MockRefcatDataId`
            Object that replicates the functionality of a dataId.
        deferredRefCat : `lsst.pipe.base.InMemoryDatasetHandle`
            Dataset handle for reference catalog.
        """
        nRefs = int(cls.nStars * inReferenceFraction)
        refStarIndices = np.random.choice(cls.nStars, nRefs, replace=False)
        # Make simpleCatalog to hold data, create datasetRef with `region`
        # determined by bounding box used in above simulate.
        refSchema = afwTable.SimpleTable.makeMinimalSchema()
        idKey = refSchema.addField("sourceId", type="I")
        fluxKey = refSchema.addField("test_filter_flux", units="nJy", type=np.float64)
        raErrKey = refSchema.addField("coord_raErr", units="rad", type=np.float64)
        decErrKey = refSchema.addField("coord_decErr", units="rad", type=np.float64)
        pmraErrKey = refSchema.addField("pm_raErr", units="rad2 / yr", type=np.float64)
        pmdecErrKey = refSchema.addField("pm_decErr", units="rad2 / yr", type=np.float64)
        refCat = afwTable.SimpleCatalog(refSchema)
        ref_md = PropertyList()
        ref_md.set("REFCAT_FORMAT_VERSION", 1)
        refCat.table.setMetadata(ref_md)
        for i in refStarIndices:
            record = refCat.addNew()
            record.set(idKey, starIds[i])
            record.setRa(lsst.geom.Angle(starRas[i], lsst.geom.degrees))
            record.setDec(lsst.geom.Angle(starDecs[i], lsst.geom.degrees))
            record.set(fluxKey, 1)
            record.set(raErrKey, 0.00001)
            record.set(decErrKey, 0.00001)
            record.set(pmraErrKey, 1e-9)
            record.set(pmdecErrKey, 1e-9)
        refDataId = MockRefcatDataId(bounds)
        deferredRefCat = InMemoryDatasetHandle(refCat, storageClass="SourceCatalog", htm7="mockRefCat")

        return refDataId, deferredRefCat

    @classmethod
    def _make_sourceCat(cls, starIds, starRas, starDecs, trueWCSs, inScienceFraction):
        """Make a `pd.DataFrame` catalog with the columns needed for the
        object selector.

        Parameters
        ----------
        starIds : `np.ndarray` [`int`]
            Source ids for the simulated stars
        starRas : `np.ndarray` [`float`]
            RAs of the simulated stars
        starDecs : `np.ndarray` [`float`]
            Decs of the simulated stars
        trueWCSs : `list` [`lsst.afw.geom.SkyWcs`]
            WCS with which to simulate the source pixel coordinates
        inReferenceFraction : float
            Percentage of simulated stars to include in reference catalog

        Returns
        -------
        sourceCat : `list` [`lsst.pipe.base.InMemoryDatasetHandle`]
            List of reference to source catalogs.
        """
        inputCatalogRefs = []
        # Take a subset of the simulated data
        # Use true wcs objects to put simulated data into ccds
        bbox = lsst.geom.BoxD(
            lsst.geom.Point2D(
                cls.inputVisitSummary[0][0]["bbox_min_x"], cls.inputVisitSummary[0][0]["bbox_min_y"]
            ),
            lsst.geom.Point2D(
                cls.inputVisitSummary[0][0]["bbox_max_x"], cls.inputVisitSummary[0][0]["bbox_max_y"]
            ),
        )
        bboxCorners = bbox.getCorners()
        cls.inputCatalogRefs = []
        for v, visit in enumerate(cls.testVisits):
            nVisStars = int(cls.nStars * inScienceFraction)
            visitStarIndices = np.random.choice(cls.nStars, nVisStars, replace=False)
            visitStarIds = starIds[visitStarIndices]
            visitStarRas = starRas[visitStarIndices]
            visitStarDecs = starDecs[visitStarIndices]
            sourceCats = []
            for detector in trueWCSs[visit]:
                detWcs = detector.getWcs()
                detectorId = detector["id"]
                radecCorners = detWcs.pixelToSky(bboxCorners)
                detectorFootprint = sphgeom.ConvexPolygon([rd.getVector() for rd in radecCorners])
                detectorIndices = detectorFootprint.contains(
                    (visitStarRas * u.degree).to(u.radian), (visitStarDecs * u.degree).to(u.radian)
                )
                nDetectorStars = detectorIndices.sum()
                detectorArray = np.ones(nDetectorStars, dtype=bool) * detector["id"]

                ones_like = np.ones(nDetectorStars)
                zeros_like = np.zeros(nDetectorStars, dtype=bool)

                x, y = detWcs.skyToPixelArray(
                    visitStarRas[detectorIndices], visitStarDecs[detectorIndices], degrees=True
                )

                origWcs = (cls.inputVisitSummary[v][cls.inputVisitSummary[v]["id"] == detectorId])[0].getWcs()
                inputRa, inputDec = origWcs.pixelToSkyArray(x, y, degrees=True)

                sourceDict = {}
                sourceDict["detector"] = detectorArray
                sourceDict["sourceId"] = visitStarIds[detectorIndices]
                sourceDict["x"] = x
                sourceDict["y"] = y
                sourceDict["xErr"] = 1e-3 * ones_like
                sourceDict["yErr"] = 1e-3 * ones_like
                sourceDict["inputRA"] = inputRa
                sourceDict["inputDec"] = inputDec
                sourceDict["trueRA"] = visitStarRas[detectorIndices]
                sourceDict["trueDec"] = visitStarDecs[detectorIndices]
                for key in ["apFlux_12_0_flux", "apFlux_12_0_instFlux", "ixx", "iyy"]:
                    sourceDict[key] = ones_like
                for key in [
                    "pixelFlags_edge",
                    "pixelFlags_saturated",
                    "pixelFlags_interpolatedCenter",
                    "pixelFlags_interpolated",
                    "pixelFlags_crCenter",
                    "pixelFlags_bad",
                    "hsmPsfMoments_flag",
                    "apFlux_12_0_flag",
                    "extendedness",
                    "sizeExtendedness",
                    "parentSourceId",
                    "deblend_nChild",
                    "ixy",
                ]:
                    sourceDict[key] = zeros_like
                sourceDict["apFlux_12_0_instFluxErr"] = 1e-3 * ones_like
                sourceDict["detect_isPrimary"] = ones_like.astype(bool)

                sourceCat = pd.DataFrame(sourceDict)
                sourceCats.append(sourceCat)

            visitSourceTable = pd.concat(sourceCats)

            inputCatalogRef = InMemoryDatasetHandle(
                visitSourceTable, storageClass="DataFrame", dataId={"visit": visit}
            )

            inputCatalogRefs.append(inputCatalogRef)

        return inputCatalogRefs

    @classmethod
    def _make_wcs(cls, model, inputVisitSummaries):
        """Make a `lsst.afw.geom.SkyWcs` from given model parameters

        Parameters
        ----------
        model : `dict`
            Dictionary with WCS model parameters
        inputVisitSummaries : `list` [`lsst.afw.table.ExposureCatalog`]
            Visit summary catalogs
        Returns
        -------
        catalogs : `dict` [`int`, `lsst.afw.table.ExposureCatalog`]
            Visit summary catalogs with WCS set to input model
        """

        # Pixels will need to be rescaled before going into the mappings
        xscale = inputVisitSummaries[0][0]["bbox_max_x"] - inputVisitSummaries[0][0]["bbox_min_x"]
        yscale = inputVisitSummaries[0][0]["bbox_max_y"] - inputVisitSummaries[0][0]["bbox_min_y"]

        catalogs = {}
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        schema.addField("visit", type="L", doc="Visit number")
        for visitSum in inputVisitSummaries:
            visit = visitSum[0]["visit"]
            visitMapName = f"{visit}/poly"
            visitModel = model[visitMapName]

            catalog = lsst.afw.table.ExposureCatalog(schema)
            catalog.resize(len(visitSum))
            catalog["visit"] = visit

            raDec = visitSum[0].getVisitInfo().getBoresightRaDec()

            visitMapType = visitModel["Type"]
            visitDict = {"Type": visitMapType}
            if visitMapType == "Poly":
                mapCoefficients = visitModel["XPoly"]["Coefficients"] + visitModel["YPoly"]["Coefficients"]
                visitDict["Coefficients"] = mapCoefficients

            for d, detector in enumerate(visitSum):
                detectorId = detector["id"]
                filterName = detector["physical_filter"]
                detectorMapName = f"{filterName}/{detectorId}/poly"
                detectorModel = model[detectorMapName]

                detectorMapType = detectorModel["Type"]
                mapDict = {detectorMapName: {"Type": detectorMapType}, visitMapName: visitDict}
                if detectorMapType == "Poly":
                    mapCoefficients = (
                        detectorModel["XPoly"]["Coefficients"] + detectorModel["YPoly"]["Coefficients"]
                    )
                    mapDict[detectorMapName]["Coefficients"] = mapCoefficients

                outWCS = cls.task._make_afw_wcs(
                    mapDict,
                    raDec.getRa(),
                    raDec.getDec(),
                    doNormalizePixels=True,
                    xScale=xscale,
                    yScale=yscale,
                )
                catalog[d].setId(detectorId)
                catalog[d].setWcs(outWCS)

            catalog.sort()
            catalogs[visit] = catalog

        return catalogs

    def test_get_exposure_info(self):
        """Test that information for input exposures is as expected and that
        the WCS in the class object gives approximately the same results as the
        input `lsst.afw.geom.SkyWcs`.
        """

        # The total number of extensions is the number of detectors for each
        # visit plus one for the reference catalog for each field.
        totalExtensions = sum([len(visSum) for visSum in self.inputVisitSummary]) + self.nFields

        self.assertEqual(totalExtensions, len(self.extensionInfo.visit))

        taskVisits = set(self.extensionInfo.visit)
        refVisits = np.arange(0, -1 * self.nFields, -1).tolist()
        self.assertEqual(taskVisits, set(self.testVisits + refVisits))

        xx = np.linspace(0, 2000, 3)
        yy = np.linspace(0, 4000, 6)
        xgrid, ygrid = np.meshgrid(xx, yy)
        for visSum in self.inputVisitSummary:
            visit = visSum[0]["visit"]
            for detectorInfo in visSum:
                detector = detectorInfo["id"]
                extensionIndex = np.flatnonzero(
                    (self.extensionInfo.visit == visit) & (self.extensionInfo.detector == detector)
                )[0]
                fitWcs = self.extensionInfo.wcs[extensionIndex]
                calexpWcs = detectorInfo.getWcs()

                tanPlaneXY = np.array([fitWcs.toWorld(x, y) for (x, y) in zip(xgrid.ravel(), ygrid.ravel())])

                calexpra, calexpdec = calexpWcs.pixelToSkyArray(xgrid.ravel(), ygrid.ravel(), degrees=True)

                # The pixel origin maps to a position slightly off from the
                # tangent plane origin, so we want to use this value for the
                # tangent-plane-to-sky part of the mapping.
                tangentPlaneCenter = fitWcs.toWorld(
                    calexpWcs.getPixelOrigin().getX(), calexpWcs.getPixelOrigin().getY()
                )
                tangentPlaneOrigin = lsst.geom.Point2D(tangentPlaneCenter)
                skyOrigin = calexpWcs.pixelToSky(
                    calexpWcs.getPixelOrigin().getX(), calexpWcs.getPixelOrigin().getY()
                )
                cdMatrix = afwgeom.makeCdMatrix(1.0 * lsst.geom.degrees, 0.0 * lsst.geom.degrees, True)
                iwcToSkyWcs = afwgeom.makeSkyWcs(tangentPlaneOrigin, skyOrigin, cdMatrix)
                newRAdeg, newDecdeg = iwcToSkyWcs.pixelToSkyArray(
                    tanPlaneXY[:, 0], tanPlaneXY[:, 1], degrees=True
                )

                np.testing.assert_allclose(calexpra, newRAdeg)
                np.testing.assert_allclose(calexpdec, newDecdeg)

    def test_refCatLoader(self):
        """Test that we can load objects from refCat"""

        tmpAssociations = wcsfit.FoFClass(
            self.fields,
            self.instruments,
            self.exposuresHelper,
            [self.fieldRadius.asDegrees()],
            (self.task.config.matchRadius * u.arcsec).to(u.degree).value,
        )

        self.task._load_refcat(
            self.refObjectLoader,
            self.extensionInfo,
            associations=tmpAssociations,
            center=self.fieldCenter,
            radius=self.fieldRadius,
            epoch=self.exposureInfo.medianEpoch,
        )

        # We have only loaded one catalog, so getting the 'matches' should just
        # return the same objects we put in, except some random objects that
        # are too close together.
        tmpAssociations.sortMatches(self.fieldNumber, minMatches=1)

        nMatches = (np.array(tmpAssociations.sequence) == 0).sum()

        self.assertLessEqual(nMatches, self.nStars)
        self.assertGreater(nMatches, self.nStars * 0.9)

    def test_loading_and_association(self):
        """Test that objects can be loaded and correctly associated."""
        # Running `_load_catalogs_and_associate` changes the input WCSs, so
        # recalculate them here so that the variables shared among tests are
        # not affected.
        _, exposuresHelper, extensionInfo, instruments = self.task._get_exposure_info(
            self.inputVisitSummary, refEpoch=self.refEpoch
        )

        tmpAssociations = wcsfit.FoFClass(
            self.fields,
            self.instruments,
            exposuresHelper,
            [self.fieldRadius.asDegrees()],
            (self.task.config.matchRadius * u.arcsec).to(u.degree).value,
        )
        self.task._load_catalogs_and_associate(tmpAssociations, self.inputCatalogRefs, extensionInfo)

        tmpAssociations.sortMatches(self.fieldNumber, minMatches=2)

        matchIds = []
        correctMatches = []
        for s, e, o in zip(tmpAssociations.sequence, tmpAssociations.extn, tmpAssociations.obj):
            objVisitInd = extensionInfo.visitIndex[e]
            objDet = extensionInfo.detector[e]
            extnInds = self.inputCatalogRefs[objVisitInd].get()["detector"] == objDet
            objInfo = self.inputCatalogRefs[objVisitInd].get()[extnInds].iloc[o]
            if s == 0:
                if len(matchIds) > 0:
                    correctMatches.append(len(set(matchIds)) == 1)
                matchIds = []

            matchIds.append(objInfo["sourceId"])

        # A few matches may incorrectly associate sources because of the random
        # positions
        self.assertGreater(sum(correctMatches), len(correctMatches) * 0.95)

    def test_make_outputs(self):
        """Test that the run method recovers the input model parameters."""
        for v, visit in enumerate(self.testVisits):
            visitSummary = self.inputVisitSummary[v]
            outputWcsCatalog = self.outputs.outputWcss[visit]
            visitSources = self.inputCatalogRefs[v].get()
            for d, detectorRow in enumerate(visitSummary):
                detectorId = detectorRow["id"]
                fitwcs = outputWcsCatalog[d].getWcs()
                detSources = visitSources[visitSources["detector"] == detectorId]
                fitRA, fitDec = fitwcs.pixelToSkyArray(detSources["x"], detSources["y"], degrees=True)
                dRA = fitRA - detSources["trueRA"]
                dDec = fitDec - detSources["trueDec"]
                # Check that input coordinates match the output coordinates
                self.assertAlmostEqual(np.mean(dRA), 0)
                self.assertAlmostEqual(np.std(dRA), 0)
                self.assertAlmostEqual(np.mean(dDec), 0)
                self.assertAlmostEqual(np.std(dDec), 0)

    def test_compute_model_params(self):
        """Test the optional model parameters and covariance output."""
        modelParams = pd.DataFrame(self.outputs.modelParams)
        # Check that DataFrame is the expected size.
        shape = modelParams.shape
        self.assertEqual(shape[0] + 4, shape[1])
        # Check that covariance matrix is symmetric.
        covariance = (modelParams.iloc[:, 4:]).to_numpy()
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-18)

    def test_run(self):
        """Test that run method recovers the input model parameters"""
        outputMaps = self.outputs.fitModel.mapCollection.getParamDict()

        for v, visit in enumerate(self.testVisits):
            visitSummary = self.inputVisitSummary[v]
            visitMapName = f"{visit}/poly"

            origModel = self.trueModel[visitMapName]
            if origModel["Type"] != "Identity":
                fitModel = outputMaps[visitMapName]
                origXPoly = origModel["XPoly"]["Coefficients"]
                origYPoly = origModel["YPoly"]["Coefficients"]
                fitXPoly = fitModel[: len(origXPoly)]
                fitYPoly = fitModel[len(origXPoly) :]

                absDiffX = abs(fitXPoly - origXPoly)
                absDiffY = abs(fitYPoly - origYPoly)
                # Check that input visit model matches fit
                np.testing.assert_array_less(absDiffX, 1e-6)
                np.testing.assert_array_less(absDiffY, 1e-6)
            for d, detectorRow in enumerate(visitSummary):
                detectorId = detectorRow["id"]
                filterName = detectorRow["physical_filter"]
                detectorMapName = f"{filterName}/{detectorId}/poly"
                origModel = self.trueModel[detectorMapName]
                if (origModel["Type"] != "Identity") and (v == 0):
                    fitModel = outputMaps[detectorMapName]
                    origXPoly = origModel["XPoly"]["Coefficients"]
                    origYPoly = origModel["YPoly"]["Coefficients"]
                    fitXPoly = fitModel[: len(origXPoly)]
                    fitYPoly = fitModel[len(origXPoly) :]
                    absDiffX = abs(fitXPoly - origXPoly)
                    absDiffY = abs(fitYPoly - origYPoly)
                    # Check that input detector model matches fit
                    np.testing.assert_array_less(absDiffX, 1e-7)
                    np.testing.assert_array_less(absDiffY, 1e-7)

    def test_missingWcs(self):
        """Test that task does not fail when the input WCS is None for one
        extension and that the fit WCS for that extension returns a finite
        result.
        """
        inputVisitSummary = self.inputVisitSummary.copy()
        # Set one WCS to be None
        testVisit = 0
        testDetector = 20
        inputVisitSummary[testVisit][testDetector].setWcs(None)

        outputs = self.task.run(
            self.inputCatalogRefs,
            inputVisitSummary,
            instrumentName=self.instrumentName,
            refEpoch=self.refEpoch,
            refObjectLoader=self.refObjectLoader,
        )

        # Check that the fit WCS for the extension with input WCS=None returns
        # finite sky values.
        testWcs = outputs.outputWcss[self.testVisits[testVisit]][testDetector].getWcs()
        testSky = testWcs.pixelToSky(0, 0)
        self.assertTrue(testSky.isFinite())

    def test_inputCameraModel(self):
        """Test running task with an input camera model, and check that true
        object coordinates are recovered.
        """
        config = copy(self.config)
        config.saveCameraModel = False
        config.useInputCameraModel = True
        task = GbdesAstrometricFitTask(config=config)
        with open(os.path.join(self.datadir, "sample_camera_model.yaml"), "r") as f:
            cameraModel = yaml.load(f, Loader=yaml.Loader)

        outputs = task.run(
            self.inputCatalogRefs,
            self.inputVisitSummary,
            instrumentName=self.instrumentName,
            refObjectLoader=self.refObjectLoader,
            inputCameraModel=cameraModel,
        )

        # Check that the output WCS is close (not necessarily exactly equal) to
        # the result when fitting the full model.
        for v, visit in enumerate(self.testVisits):
            visitSummary = self.inputVisitSummary[v]
            outputWcsCatalog = outputs.outputWcss[visit]
            visitSources = self.inputCatalogRefs[v].get()
            for d, detectorRow in enumerate(visitSummary):
                detectorId = detectorRow["id"]
                fitwcs = outputWcsCatalog[d].getWcs()
                detSources = visitSources[visitSources["detector"] == detectorId]
                fitRA, fitDec = fitwcs.pixelToSkyArray(detSources["x"], detSources["y"], degrees=True)
                dRA = fitRA - detSources["trueRA"]
                dDec = fitDec - detSources["trueDec"]
                # Check that input coordinates match the output coordinates
                self.assertAlmostEqual(np.mean(dRA), 0)
                self.assertAlmostEqual(np.std(dRA), 0)
                self.assertAlmostEqual(np.mean(dDec), 0)
                self.assertAlmostEqual(np.std(dDec), 0)


class TestGbdesGlobalAstrometricFit(TestGbdesAstrometricFit):
    @classmethod
    def setUpClass(cls):
        # Set random seed
        np.random.seed(1234)

        # Fraction of simulated stars in the reference catalog and science
        # exposures
        inReferenceFraction = 1
        inScienceFraction = 1

        # Make fake data
        cls.datadir = os.path.join(TESTDIR, "data")

        cls.nFields = 2
        cls.instrumentName = "HSC"
        cls.refEpoch = 57205.5

        # Make test inputVisitSummary. VisitSummaryTables are taken from
        # collection HSC/runs/RC2/w_2022_20/DM-34794
        cls.testVisits = [1176, 17900, 17930, 17934, 36434, 36460, 36494, 36446]
        cls.inputVisitSummary = []
        for testVisit in cls.testVisits:
            visSum = afwTable.ExposureCatalog.readFits(
                os.path.join(cls.datadir, f"visitSummary_{testVisit}.fits")
            )
            cls.inputVisitSummary.append(visSum)

        cls.config = GbdesGlobalAstrometricFitConfig()
        cls.config.systematicError = 0
        cls.config.devicePolyOrder = 4
        cls.config.exposurePolyOrder = 6
        cls.config.fitReserveFraction = 0
        cls.config.fitReserveRandomSeed = 1234
        cls.config.saveModelParams = True
        cls.config.saveCameraModel = True
        cls.task = GbdesGlobalAstrometricFitTask(config=cls.config)

        cls.fields, cls.fieldRegions = cls.task._prep_sky(cls.inputVisitSummary)

        (
            cls.exposureInfo,
            cls.exposuresHelper,
            cls.extensionInfo,
            cls.instruments,
        ) = cls.task._get_exposure_info(cls.inputVisitSummary, fieldRegions=cls.fieldRegions)

        refDataIds, deferredRefCats = [], []
        allStarIds = []
        allStarRAs = []
        allStarDecs = []
        for region in cls.fieldRegions.values():
            # Bounding box of observations:
            bbox = region.getBoundingBox()
            raMin = bbox.getLon().getA().asDegrees()
            raMax = bbox.getLon().getB().asDegrees()
            decMin = bbox.getLat().getA().asDegrees()
            decMax = bbox.getLat().getB().asDegrees()

            # Make random set of data in a bounding box determined by input
            # visits
            cls.nStars, starIds, starRAs, starDecs = cls._make_simulated_stars(
                raMin, raMax, decMin, decMax, cls.config.matchRadius
            )

            allStarIds.append(starIds)
            allStarRAs.append(starRAs)
            allStarDecs.append(starDecs)

            corners = [
                lsst.geom.SpherePoint(ra, dec, lsst.geom.degrees).getVector()
                for (ra, dec) in zip([raMin, raMax, raMax, raMin], [decMin, decMin, decMax, decMax])
            ]
            conv_box = lsst.sphgeom.ConvexPolygon.convexHull(corners)
            # Make a reference catalog that will be loaded into
            # ReferenceObjectLoader
            refDataId, deferredRefCat = cls._make_refCat(
                starIds, starRAs, starDecs, inReferenceFraction, conv_box
            )
            refDataIds.append(refDataId)
            deferredRefCats.append(deferredRefCat)

        cls.refObjectLoader = ReferenceObjectLoader(refDataIds, deferredRefCats)
        cls.refObjectLoader.config.requireProperMotion = False
        cls.refObjectLoader.config.anyFilterMapsToThis = "test_filter"

        cls.task.refObjectLoader = cls.refObjectLoader

        allRefObjects, allRefCovariances = {}, {}
        for f, fieldRegion in cls.fieldRegions.items():
            refObjects, refCovariance = cls.task._load_refcat(
                cls.refObjectLoader, cls.extensionInfo, epoch=cls.exposureInfo.medianEpoch, region=fieldRegion
            )
            allRefObjects[f] = refObjects
            allRefCovariances[f] = refCovariance
        cls.refObjects = allRefObjects

        # Get True WCS for stars:
        with open(os.path.join(cls.datadir, "sample_global_wcs.yaml"), "r") as f:
            cls.trueModel = yaml.load(f, Loader=yaml.Loader)

        cls.trueWCSs = cls._make_wcs(cls.trueModel, cls.inputVisitSummary)

        cls.isolatedStarCatalogs, cls.isolatedStarSources = cls._make_isolatedStars(
            allStarIds, allStarRAs, allStarDecs, cls.trueWCSs, inScienceFraction
        )

        cls.outputs = cls.task.run(
            cls.inputVisitSummary,
            cls.isolatedStarSources,
            cls.isolatedStarCatalogs,
            instrumentName=cls.instrumentName,
            refEpoch=cls.refEpoch,
            refObjectLoader=cls.refObjectLoader,
        )

    @classmethod
    def _make_isolatedStars(cls, allStarIds, allStarRAs, allStarDecs, trueWCSs, inScienceFraction):
        """Given a subset of the simulated data, make source catalogs and star
        catalogs.

        This takes the true WCSs to go from the RA and Decs of the simulated
        stars to pixel coordinates for a given visit and detector. If those
        pixel coordinates are within the bounding box of the detector, the
        source and visit information is put in the corresponding catalog of
        isolated star sources.

        Parameters
        ----------
        allStarIds : `np.ndarray` [`int`]
            Source ids for the simulated stars
        allStarRas : `np.ndarray` [`float`]
            RAs of the simulated stars
        allStarDecs : `np.ndarray` [`float`]
            Decs of the simulated stars
        trueWCSs : `list` [`lsst.afw.geom.SkyWcs`]
            WCS with which to simulate the source pixel coordinates
        inReferenceFraction : float
            Percentage of simulated stars to include in reference catalog

        Returns
        -------
        isolatedStarCatalogRefs : `list`
            [`lsst.pipe.base.InMemoryDatasetHandle`]
            List of references to isolated star catalogs.
        isolatedStarSourceRefs : `list`
            [`lsst.pipe.base.InMemoryDatasetHandle`]
            List of references to isolated star sources.
        """
        bbox = lsst.geom.BoxD(
            lsst.geom.Point2D(
                cls.inputVisitSummary[0][0]["bbox_min_x"], cls.inputVisitSummary[0][0]["bbox_min_y"]
            ),
            lsst.geom.Point2D(
                cls.inputVisitSummary[0][0]["bbox_max_x"], cls.inputVisitSummary[0][0]["bbox_max_y"]
            ),
        )
        bboxCorners = bbox.getCorners()

        isolatedStarCatalogRefs = []
        isolatedStarSourceRefs = []
        for i in range(len(allStarIds)):
            starIds = allStarIds[i]
            starRAs = allStarRAs[i]
            starDecs = allStarDecs[i]
            isolatedStarCatalog = pd.DataFrame({"ra": starRAs, "dec": starDecs}, index=starIds)
            isolatedStarCatalogRefs.append(
                InMemoryDatasetHandle(isolatedStarCatalog, storageClass="DataFrame", dataId={"tract": 0})
            )
            sourceCats = []
            for v, visit in enumerate(cls.testVisits):
                nVisStars = int(cls.nStars * inScienceFraction)
                visitStarIndices = np.random.choice(cls.nStars, nVisStars, replace=False)
                visitStarIds = starIds[visitStarIndices]
                visitStarRas = starRAs[visitStarIndices]
                visitStarDecs = starDecs[visitStarIndices]
                for detector in trueWCSs[visit]:
                    detWcs = detector.getWcs()
                    detectorId = detector["id"]
                    radecCorners = detWcs.pixelToSky(bboxCorners)
                    detectorFootprint = sphgeom.ConvexPolygon([rd.getVector() for rd in radecCorners])
                    detectorIndices = detectorFootprint.contains(
                        (visitStarRas * u.degree).to(u.radian), (visitStarDecs * u.degree).to(u.radian)
                    )
                    nDetectorStars = detectorIndices.sum()
                    detectorArray = np.ones(nDetectorStars, dtype=int) * detector["id"]
                    visitArray = np.ones(nDetectorStars, dtype=int) * visit

                    ones_like = np.ones(nDetectorStars)
                    zeros_like = np.zeros(nDetectorStars, dtype=bool)

                    x, y = detWcs.skyToPixelArray(
                        visitStarRas[detectorIndices], visitStarDecs[detectorIndices], degrees=True
                    )

                    origWcs = (cls.inputVisitSummary[v][cls.inputVisitSummary[v]["id"] == detectorId])[
                        0
                    ].getWcs()
                    inputRa, inputDec = origWcs.pixelToSkyArray(x, y, degrees=True)

                    sourceDict = {}
                    sourceDict["detector"] = detectorArray
                    sourceDict["visit"] = visitArray
                    sourceDict["obj_index"] = visitStarIds[detectorIndices]
                    sourceDict["x"] = x
                    sourceDict["y"] = y
                    sourceDict["xErr"] = 1e-3 * ones_like
                    sourceDict["yErr"] = 1e-3 * ones_like
                    sourceDict["inputRA"] = inputRa
                    sourceDict["inputDec"] = inputDec
                    sourceDict["trueRA"] = visitStarRas[detectorIndices]
                    sourceDict["trueDec"] = visitStarDecs[detectorIndices]
                    for key in ["apFlux_12_0_flux", "apFlux_12_0_instFlux", "ixx", "iyy"]:
                        sourceDict[key] = ones_like
                    for key in [
                        "pixelFlags_edge",
                        "pixelFlags_saturated",
                        "pixelFlags_interpolatedCenter",
                        "pixelFlags_interpolated",
                        "pixelFlags_crCenter",
                        "pixelFlags_bad",
                        "hsmPsfMoments_flag",
                        "apFlux_12_0_flag",
                        "extendedness",
                        "parentSourceId",
                        "deblend_nChild",
                        "ixy",
                    ]:
                        sourceDict[key] = zeros_like
                    sourceDict["apFlux_12_0_instFluxErr"] = 1e-3 * ones_like
                    sourceDict["detect_isPrimary"] = ones_like.astype(bool)

                    sourceCat = pd.DataFrame(sourceDict)
                    sourceCats.append(sourceCat)

            isolatedStarSourceTable = pd.concat(sourceCats, ignore_index=True)
            isolatedStarSourceTable = isolatedStarSourceTable.sort_values(by=["obj_index"])
            isolatedStarSourceRefs.append(
                InMemoryDatasetHandle(isolatedStarSourceTable, storageClass="DataFrame", dataId={"tract": 0})
            )

        return isolatedStarCatalogRefs, isolatedStarSourceRefs

    def test_loading_and_association(self):
        """Test that associated objects actually correspond to the same
        simulated object."""
        associations, sourceDict = self.task._associate_from_isolated_sources(
            self.isolatedStarSources, self.isolatedStarCatalogs, self.extensionInfo, self.refObjects
        )

        object_groups = np.flatnonzero(np.array(associations.sequence) == 0)
        for i in range(len(object_groups) - 1)[:10]:
            ras, decs = [], []
            for ind in np.arange(object_groups[i], object_groups[i + 1]):
                visit = self.extensionInfo.visit[associations.extn[ind]]
                detectorInd = self.extensionInfo.detectorIndex[associations.extn[ind]]
                detector = self.extensionInfo.detector[associations.extn[ind]]
                if detectorInd == -1:
                    ra = self.refObjects[visit * -1]["ra"][associations.obj[ind]]
                    dec = self.refObjects[visit * -1]["dec"][associations.obj[ind]]
                    ras.append(ra)
                    decs.append(dec)
                else:
                    x = sourceDict[visit][detector]["x"][associations.obj[ind]]
                    y = sourceDict[visit][detector]["y"][associations.obj[ind]]
                    ra, dec = self.trueWCSs[visit][detectorInd].getWcs().pixelToSky(x, y)
                    ras.append(ra.asDegrees())
                    decs.append(dec.asDegrees())
            np.testing.assert_allclose(ras, ras[0])
            np.testing.assert_allclose(decs, decs[0])

    def test_refCatLoader(self):
        """Test loading objects from the refCat in each of the fields."""

        for region in self.fieldRegions.values():
            refCat, refCov = self.task._load_refcat(
                self.refObjectLoader, self.extensionInfo, region=region, epoch=self.exposureInfo.medianEpoch
            )
            assert len(refCat) > 0

    def test_make_outputs(self):
        """Test that the run method recovers the input model parameters."""
        for isolatedStarSourceRef in self.isolatedStarSources:
            iss = isolatedStarSourceRef.get()
            visits = np.unique(iss["visit"])
            for v, visit in enumerate(visits):
                outputWcsCatalog = self.outputs.outputWcss[visit]
                visitSources = iss[iss["visit"] == visit]
                detectors = outputWcsCatalog["id"]
                for d, detectorId in enumerate(detectors):
                    fitwcs = outputWcsCatalog[d].getWcs()
                    detSources = visitSources[visitSources["detector"] == detectorId]
                    fitRA, fitDec = fitwcs.pixelToSkyArray(detSources["x"], detSources["y"], degrees=True)
                    dRA = fitRA - detSources["trueRA"]
                    dDec = fitDec - detSources["trueDec"]
                    # Check that input coordinates match the output coordinates
                    self.assertAlmostEqual(np.mean(dRA), 0)
                    self.assertAlmostEqual(np.std(dRA), 0)
                    self.assertAlmostEqual(np.mean(dDec), 0)
                    self.assertAlmostEqual(np.std(dDec), 0)

    def test_missingWcs(self):
        """Test that task does not fail when the input WCS is None for one
        extension and that the fit WCS for that extension returns a finite
        result.
        """
        inputVisitSummary = self.inputVisitSummary.copy()
        # Set one WCS to be None
        testVisit = 0
        testDetector = 20
        inputVisitSummary[testVisit][testDetector].setWcs(None)

        outputs = self.task.run(
            inputVisitSummary,
            self.isolatedStarSources,
            self.isolatedStarCatalogs,
            instrumentName=self.instrumentName,
            refEpoch=self.refEpoch,
            refObjectLoader=self.refObjectLoader,
        )

        # Check that the fit WCS for the extension with input WCS=None returns
        # finite sky values.
        testWcs = outputs.outputWcss[self.testVisits[testVisit]][testDetector].getWcs()
        testSky = testWcs.pixelToSky(0, 0)
        self.assertTrue(testSky.isFinite())

    def test_inputCameraModel(self):
        """Test running task with an input camera model, and check that true
        object coordinates are recovered.
        """
        config = copy(self.config)
        config.saveCameraModel = False
        config.useInputCameraModel = True
        task = GbdesGlobalAstrometricFitTask(config=config)
        with open(os.path.join(self.datadir, "sample_global_camera_model.yaml"), "r") as f:
            cameraModel = yaml.load(f, Loader=yaml.Loader)

        outputs = task.run(
            self.inputVisitSummary,
            self.isolatedStarSources,
            self.isolatedStarCatalogs,
            instrumentName=self.instrumentName,
            refObjectLoader=self.refObjectLoader,
            inputCameraModel=cameraModel,
        )

        # Check that the output WCS is close (not necessarily exactly equal) to
        # the result when fitting the full model.
        for isolatedStarSourceRef in self.isolatedStarSources:
            iss = isolatedStarSourceRef.get()
            visits = np.unique(iss["visit"])
            for v, visit in enumerate(visits):
                outputWcsCatalog = outputs.outputWcss[visit]
                visitSources = iss[iss["visit"] == visit]
                detectors = outputWcsCatalog["id"]
                for d, detectorId in enumerate(detectors):
                    fitwcs = outputWcsCatalog[d].getWcs()
                    detSources = visitSources[visitSources["detector"] == detectorId]
                    fitRA, fitDec = fitwcs.pixelToSkyArray(detSources["x"], detSources["y"], degrees=True)
                    dRA = fitRA - detSources["trueRA"]
                    dDec = fitDec - detSources["trueDec"]
                    # Check that input coordinates match the output coordinates
                    self.assertAlmostEqual(np.mean(dRA), 0, places=6)
                    self.assertAlmostEqual(np.std(dRA), 0)
                    self.assertAlmostEqual(np.mean(dDec), 0, places=6)
                    self.assertAlmostEqual(np.std(dDec), 0)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
