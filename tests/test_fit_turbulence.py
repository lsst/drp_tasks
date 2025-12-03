# This file is part of drp_tasks.
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
import unittest

import astropy.units as u
import numpy as np
import treegp
from astropy.table import Table, vstack

import lsst.afw.table as afwTable
import lsst.utils.tests
from lsst.drp.tasks.fit_turbulence import (
    GaussianProcessesTurbulenceFitConfig,
    GaussianProcessesTurbulenceFitTask,
)
from lsst.obs.base import createInitialSkyWcsFromBoresight
from lsst.obs.base.instrument_tests import DummyCam


class FitTurbulenceTestCase(lsst.utils.tests.TestCase):

    def setUp(self):

        visit = 1234
        self.noise = 0.01
        self.npoints = 250
        self.rng = np.random.RandomState(12345)

        config = GaussianProcessesTurbulenceFitConfig()
        # Set the max separation lower, since we only have two detectors.
        config.correlationSeparationMax = 0.1
        config.initKernel = "15**2 * AnisotropicVonKarman(invLam=array([[1.0/0.1**2,0],[0,1.0/0.1**2]]))"
        config.splineNNodes = 60
        self.task = GaussianProcessesTurbulenceFitTask(config=config)

        kernel_skl = treegp.eval_kernel(self.task.config.initKernel)

        # Make a simulated WCS catalog and position catalog.
        self.wcsCatalog = self._makeWcsCatalog(visit)
        self.positions = self._makeResidualCatalog(self.wcsCatalog, visit, kernel_skl)

        # Run the Gaussian Processes and predict at the input positions.
        self.gpx, self.gpy, *_ = self.task.runGP(self.wcsCatalog, self.positions)

        sourceCatalog = Table(
            {
                "x": self.positions["xpix"],
                "y": self.positions["ypix"],
                "detector": self.positions["deviceName"],
            }
        )
        self.predictCat = self.task.predict(self.gpx, self.gpy, self.wcsCatalog, sourceCatalog)

    def _makeWcsCatalog(self, visit):
        """Make a simulated WCS catalog."""
        camera = DummyCam().getCamera()
        schema = afwTable.ExposureTable.makeMinimalSchema()
        schema.addField("visit", type="L", doc="Visit number")
        wcsCatalog = afwTable.ExposureCatalog(schema)

        boresight = lsst.geom.SpherePoint(150 * lsst.geom.degrees, 2.5 * lsst.geom.degrees)

        for detectorId in [0, 1]:
            detector = camera[detectorId]
            wcs = createInitialSkyWcsFromBoresight(boresight, 0 * lsst.geom.degrees, detector)
            record = wcsCatalog.addNew()
            record.setId(detectorId)
            record.setWcs(wcs)
            record.setBBox(detector.getBBox())
        wcsCatalog["visit"] = visit
        return wcsCatalog

    def _makeResidualCatalog(self, wcsCatalog, visit, kernel):
        """Make a simulated catalog of source positions and residuals."""
        visitPositions = []
        for detector in [0, 1]:
            bbox = wcsCatalog[detector].getBBox()
            pixToTP = wcsCatalog[detector].wcs.getFrameDict().getMapping("PIXELS", "IWC")

            x = self.rng.uniform(bbox.getBeginX(), bbox.getEndX(), self.npoints)
            y = self.rng.uniform(bbox.getBeginY(), bbox.getEndY(), self.npoints)
            xTP, yTP = pixToTP.applyForward(np.array([x, y]))
            xSky, ySky = wcsCatalog[detector].wcs.pixelToSkyArray(x, y, degrees=True)
            detectorPositions = Table(
                {
                    "xpix": x,
                    "ypix": y,
                    "xworld": xTP,
                    "yworld": yTP,
                    "xsky": xSky,
                    "ysky": ySky,
                }
            )
            detectorPositions["deviceName"] = detector
            visitPositions.append(detectorPositions)
        visitPositions = vstack(visitPositions)
        totPoints = len(visitPositions)

        K = kernel.__call__(np.array([visitPositions["xworld"], visitPositions["yworld"]]).T)
        dx = self.rng.multivariate_normal(np.zeros(totPoints), K)
        dy = self.rng.multivariate_normal(np.zeros(totPoints), K)

        dx += np.random.normal(scale=self.noise, size=totPoints)
        dy += np.random.normal(scale=self.noise, size=totPoints)

        residError = np.ones(totPoints) * self.noise
        visitPositions["xresw"] = dx
        visitPositions["yresw"] = dy
        visitPositions["covTotalW_00"] = residError**2
        visitPositions["covTotalW_11"] = residError**2
        visitPositions["exposureName"] = str(visit)

        return visitPositions

    def test_runGP(self):
        """Check that the predicted position is about the same as the simulated
        dx/dy values, modulo the scatter in the data."""
        dRA = self.positions["xsky"] - self.predictCat["coord_ra"]
        dDec = self.positions["ysky"] - self.predictCat["coord_dec"]

        np.testing.assert_allclose(dRA, (self.positions["xresw"] * u.mas).to(u.degree).value, rtol=1e-1)
        np.testing.assert_allclose(dDec, (self.positions["yresw"] * u.mas).to(u.degree).value, rtol=1e-1)

    def test_makeWcs(self):
        """Check that the output WCS maps pixel positions to the same RA and
        Dec as the Gaussian Processes prediction.
        """

        wcsWithGP = self.task.addGPToWcs(self.gpx, self.gpy, self.wcsCatalog)
        for detectorRow in self.wcsCatalog:
            detector = detectorRow.getId()
            newWcs = wcsWithGP.find(detector).wcs

            sources = self.predictCat[self.predictCat["detector"] == detector]
            ra, dec = newWcs.pixelToSkyArray(sources["x"], sources["y"], degrees=True)

            np.testing.assert_allclose(ra, sources["coord_ra"])
            np.testing.assert_allclose(dec, sources["coord_dec"], rtol=1e-6)


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
