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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import unittest

import astropy.time
import astropy.units as u
import numpy as np
import numpy.testing as npt
from astro_metadata_translator import makeObservationInfo
from astropy.coordinates import Angle, EarthLocation, SkyCoord

import lsst.afw.table as afwTable
from lsst.drp.tasks.fit_stellar_motion import FitStellarMotionTask
from lsst.obs.base import MakeRawVisitInfoViaObsInfo
from lsst.pipe.base import InMemoryDatasetHandle


class FitStellarMotionTestCase(unittest.TestCase):
    def setUp(self):

        visits = [2025051500122, 2025051500133, 2025051500144, 2025052300192, 2025052400145]
        self.time0 = astropy.time.Time(2025.5, format="jyear")
        timeDeltas = np.arange(-2, 3) * astropy.time.TimeDelta(300 * u.day)
        times = self.time0 + timeDeltas
        self.visitSummaryHandles = {
            visit: self._makeVisitSummaryTableHandle(time) for visit, time in zip(visits, times)
        }

        nObjects = 10
        self.ras = np.linspace(150, 151, nObjects)
        self.decs = np.linspace(2, 3, nObjects)
        self.pmRAs = np.linspace(-5, 5, nObjects)
        self.pmDecs = np.linspace(-5, -3, nObjects)
        self.parallaxes = np.zeros(nObjects)

        self.starCatalogHandle, self.visitCatalogHandles, self.starSources = self._makeCatalogs(
            visits, nObjects, timeDeltas
        )

        self.task = FitStellarMotionTask()

        self.visitStars, self.visitInfo = self.task._load_sources(
            self.starSources, self.visitSummaryHandles, self.visitCatalogHandles
        )

        self.refCat = self._make_refCat(nObjects)

    def _makeVisitSummaryTableHandle(self, time):
        """Make some arbitrary visit summary tables."""
        schema = afwTable.ExposureTable.makeMinimalSchema()
        visitSummaryTable = afwTable.ExposureCatalog(schema)

        record = visitSummaryTable.addNew()
        lsstLat = -30.244639 * u.degree
        lsstLon = -70.749417 * u.degree
        lsstAlt = 2663.0 * u.m
        loc = EarthLocation(lat=lsstLat, lon=lsstLon, height=lsstAlt)

        obsInfo = makeObservationInfo(
            location=loc,
            datetime_begin=time - 15 * u.second,
            datetime_end=time + 15 * u.second,
            boresight_rotation_angle=Angle(0.0 * u.degree),
            boresight_rotation_coord="sky",
            observation_type="science",
        )

        visitInfo = MakeRawVisitInfoViaObsInfo.observationInfo2visitInfo(obsInfo)
        record.setVisitInfo(visitInfo)
        handle = InMemoryDatasetHandle(visitSummaryTable)
        return handle

    def _makeCatalogs(self, visits, nObjects, timeDeltas):
        """Make catalogs to match the isolated_star, isolated_star_association,
        and source catalogs."""
        nVisits = len(visits)

        objectCoords = SkyCoord(
            self.ras * u.degree,
            self.decs * u.degree,
            pm_ra_cosdec=self.pmRAs * np.cos((self.decs * u.degree).to(u.radian)) * u.mas / u.yr,
            pm_dec=self.pmDecs * u.mas / u.yr,
            obstime=self.time0,
        )

        objIndices = np.arange(nObjects)

        starCatalog = astropy.table.Table({"isolated_star_id": objIndices, "ra": self.ras, "dec": self.decs})
        starCatalogHandle = InMemoryDatasetHandle(starCatalog, storageClass="ArrowAstropy")

        sourceIds = []
        visitCatalogHandles = {}
        for v, visit in enumerate(visits):
            visitCoords = objectCoords.apply_space_motion(dt=timeDeltas[v])

            # Make some arbitrary sourceIds
            visitSourceIds = visit * 100 + np.arange(nObjects)
            sourceIds.extend(visitSourceIds)
            catalog = {
                "sourceId": visitSourceIds,
                "ra": visitCoords.ra.degree,
                "dec": visitCoords.dec.degree,
                "raErr": np.ones(nObjects) * 1e-6,
                "decErr": np.ones(nObjects) * 1e-6,
                "ra_dec_Cov": np.ones(nObjects) * 1e-14,
            }
            catalog = astropy.table.Table(catalog)
            visitCatalogHandles[visit] = InMemoryDatasetHandle(
                catalog,
                storageClass="ArrowAstropy",
                parameters={
                    "columns": [
                        "sourceId",
                        "ra",
                        "dec",
                        "raErr",
                        "decErr",
                        "ra_dec_Cov",
                    ]
                },
            )
        starSourceDict = {
            "visit": np.repeat(visits, nObjects),
            "obj_index": np.tile(objIndices, nVisits),
            "sourceId": sourceIds,
        }
        starSources = astropy.table.Table(starSourceDict)
        starSources.add_index("sourceId")

        return starCatalogHandle, visitCatalogHandles, starSources

    def _make_refCat(self, nObjects):
        """Make a reference catalog."""
        covariance = np.zeros((nObjects, 5, 5))
        covariance[:, 0, 0] = 1
        covariance[:, 1, 1] = 1
        covariance[:, 2, 2] = 0.1
        covariance[:, 3, 3] = 0.1
        covariance[:, 4, 4] = 0.1
        refCat = {
            "ra": self.ras * u.degree,
            "dec": self.decs * u.degree,
            "raPM": self.pmRAs * u.mas / u.yr,
            "decPM": self.pmDecs * u.mas / u.yr,
            "parallax": self.parallaxes * u.mas,
            "covariance": covariance,
        }
        return astropy.table.Table(refCat)

    def test_fit_objects_without_reference(self):
        """Turn off task.config.includeReferenceCatalog to test fit without a
        reference catalog.
        """

        self.task.config.includeReferenceCatalog = False
        outCat, predictedRADec = self.task._fit_objects(
            self.visitStars,
            self.starCatalogHandle,
            self.starSources,
            self.visitInfo,
            self.time0,
        )

        npt.assert_allclose(self.pmRAs, outCat["raPM"], rtol=2e-3)
        npt.assert_allclose(self.pmDecs, outCat["decPM"], rtol=1e-3)
        npt.assert_allclose(self.parallaxes, outCat["parallax"], atol=2e-3)

        npt.assert_allclose(predictedRADec["ra"], self.visitStars["ra"])
        npt.assert_allclose(predictedRADec["dec"], self.visitStars["dec"])

    def test_fit_objects_with_reference(self):
        """Test fit with a reference catalog."""

        outCat, predictedRADec = self.task._fit_objects(
            self.visitStars,
            self.starCatalogHandle,
            self.starSources,
            self.visitInfo,
            self.time0,
            refCatalog=self.refCat,
        )

        npt.assert_allclose(self.pmRAs, outCat["raPM"], rtol=1e-4)
        npt.assert_allclose(self.pmDecs, outCat["decPM"], rtol=1e-4)
        npt.assert_allclose(self.parallaxes, outCat["parallax"], atol=1e-3)

        npt.assert_allclose(predictedRADec["ra"], self.visitStars["ra"])
        npt.assert_allclose(predictedRADec["dec"], self.visitStars["dec"])


if __name__ == "__main__":
    unittest.main()
