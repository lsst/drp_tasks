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

import treegp
from tqdm import tqdm
import numpy as np
import astropy.units as u
import lsst.pipe.base as pipeBase
import lsst.pex.config as pexConfig
import matplotlib.pyplot as plt


class GaussianProcessesTurbulenceFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "skymap", "tract"),
    defaultTemplates={
        "inputName": "gbdesAstrometricFit",
    },
):
    inputWcs = pipeBase.connectionTypes.Input(
        doc=(
            "Per-tract, per-visit world coordinate systems derived from the fitted model."
            " These catalogs only contain entries for detectors with an output, and use"
            " the detector id for the catalog id, sorted on id for fast lookups of a detector."
        ),
        name="{inputName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "skymap", "tract"),
    )
    inputPositions = pipeBase.connectionTypes.Input(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="{inputName}_fitStars",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "skymap", "tract", "physical_filter"),
    )
    inputSources = pipeBase.connectionTypes.Input(
        doc="Source table in parquet format.",
        name="single_visit_star",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        deferLoad=True,
    )
    # outputWcs -> same dimensions as inputWcs, but with GP map added, can't make this for now.
    outputSources = pipeBase.connectionTypes.Output(
        doc="Source table in parquet format with sky positions corrected for turbulence.",
        name="gp_calibrated_star",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
    )


class GaussianProcessesTurbulenceFitConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=GaussianProcessesTurbulenceFitConnections
):
    initKernel = pexConfig.Field(
        dtype=str,
        doc="The type of function that will be used to modeled spatial correlation.",
        default="15**2 * AnisotropicVonKarman(invLam=array([[1./3000.**2,0],[0,1./3000.**2]]))",
    )
    initCorrelationLength = pexConfig.ListField(
        dtype=float,
        doc=(
            "These are the initial parameters for fiting the anisotropic correlation length. p0[0] is the "
            "equivalent of the isotropic correlation length, and p0[1]/p0[2] are ellipticity parameters and "
            "are mathematically equivalent to e1/e2 in weak-lensing. p0[1]/p0[2] must be in the range "
            "[-1,1], where 0 means the correlation is isotropic."
        ),
        default=[1, -0.2, -0.2],
    )
    correlationSeparationMin = pexConfig.Field(
        dtype=float,
        doc="Minimum distance separation in arcsec in the computation of the 2-point correlation function.",
        default=0.0,
    )
    correlationSeparationMax = pexConfig.Field(
        dtype=float,
        doc="Maximum distance separation in arcsec in the computation of the 2-point correlation function.",
        default=0.3,
    )
    maxTrainingPoints = pexConfig.Field(
        dtype=int,
        doc="Maximum number of points to use in the Gaussian Processes training.",
        default=10000,
    )
    pixelSize = pexConfig.Field(
        dtype=float,
        doc="Pixel size in arcseconds.",
        default=0.2,
    )
    doValidation = pexConfig.Field(
        dtype=bool, doc="Validate gaussian processes results using test data.", default=False
    )


class GaussianProcessesTurbulenceFitTask(pipeBase.PipelineTask):
    """TODO: Run GP on astrometric residuals (assuming they are due to turbulence)"""

    ConfigClass = GaussianProcessesTurbulenceFitConfig
    _DefaultName = "gaussianProcessesTurbulenceFit"

    def run(self, inputWcs, inputPositions, inputSources):

        visit = str(inputWcs[0]["visit"])

        visitPositions = inputPositions[inputPositions["exposureName"] == visit]

        x = visitPositions["xworld"]
        y = visitPositions["yworld"]
        dx = visitPositions["xresw"]
        dy = visitPositions["yresw"]
        residError = (visitPositions["sigpix"] * self.config.pixelSize * u.arcsec).to(u.mas).value

        gpx, gpy, trainInd, testInd = self.runGP(x, y, dx, dy, residError)

        self.evaluate(gpx, gpy, trainInd, testInd, x, y, dx, dy, residError)

        sourceCatalog = inputSources.get(parameters={"columns": ["x", "y", "detector"]})

        predictedCatalog = self.predict(gpx, gpy, inputWcs, sourceCatalog)

        return pipeBase.Struct(outputSources=predictedCatalog)

    def runGP(self, x, y, dx, dy, residError):
        """TODO"""
        # Set up the Gaussian Processes.

        gpx = treegp.GPInterpolation(
            kernel=self.config.initKernel,
            optimizer="anisotropic",
            normalize=True,
            nbins=21,
            min_sep=self.config.correlationSeparationMin,
            max_sep=self.config.correlationSeparationMax,
            p0=self.config.initCorrelationLength,
        )

        coord = np.array([x, y]).T
        rng = np.random.default_rng(1234)
        nPoints = len(coord)
        nTrain = min([nPoints, self.config.maxTrainingPoints])
        perm = rng.permutation(np.arange(nPoints))
        trainInds = perm[:nTrain]
        testInds = perm[nTrain:]

        # Predict on du:
        gpx.initialize(coord[trainInds], dx[trainInds], y_err=residError[trainInds])
        gpx.solve()

        # Predict on du:
        gpy = treegp.GPInterpolation(
            kernel=self.config.initKernel,
            optimizer="anisotropic",
            normalize=True,
            nbins=21,
            min_sep=self.config.correlationSeparationMin,
            max_sep=self.config.correlationSeparationMax,
            p0=self.config.initCorrelationLength,
        )

        gpy.initialize(coord[trainInds], dy[trainInds], y_err=residError[trainInds])
        gpy.solve()

        return gpx, gpy, trainInds, testInds

    def predict(self, gpx, gpy, inputWcs, sourceCatalog):
        """TODO"""
        correctedCoordinates = np.zeros((len(sourceCatalog), 2))

        for detector in tqdm(inputWcs):
            detId = detector["id"]
            detWCS = detector.wcs
            detInd = sourceCatalog["detector"] == detId
            detectorSources = sourceCatalog[detInd]

            initialSky = detWCS.pixelToSkyArray(detectorSources["x"], detectorSources["y"])
            tangentPlaneToSky = detWCS.getFrameDict().getMapping("IWC", "SKY")
            tangentPlaneCoords = tangentPlaneToSky.applyInverse(np.array(initialSky)).T

            xPrediction = gpx.predict(tangentPlaneCoords)
            yPrediction = gpy.predict(tangentPlaneCoords)

            correctedTangentPlaneX = tangentPlaneCoords[:, 0] - xPrediction
            correctedTangentPlaneY = tangentPlaneCoords[:, 1] - yPrediction
            correctedSkyCoords = tangentPlaneToSky.applyForward(
                np.array([correctedTangentPlaneX, correctedTangentPlaneY])
            )
            correctedCoordinates[detInd] = correctedSkyCoords.T

        return correctedCoordinates

    def plot_visit(self, x, y, dx, dy, predx, predy):

        xie, xib, logr = treegp.comp_eb_treecorr(x, y, dx, dy, rmin=20 / 3600, rmax=0.6, dlogr=0.3)
        xie_resid, xib_resid, logr_resid = treegp.comp_eb_treecorr(
            x, y, dx - predx, dy - predy, rmin=20 / 3600, rmax=0.6, dlogr=0.3
        )

        residualLimit = np.nanstd(dx)

        fig, subs = plt.subplot_mosaic(
            [["dx", "predx", "residx", "eb"], ["dy", "predy", "residy", "eb"]],
            figsize=(15, 8),
            layout="constrained",
        )
        plt.subplots_adjust(wspace=0.3, right=0.99, left=0.05)
        im = subs["dx"].scatter(x, y, c=dx, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)
        subs["dy"].scatter(x, y, c=dy, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)

        subs["predx"].scatter(
            x, y, c=predx, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1
        )
        subs["predy"].scatter(
            x, y, c=predy, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1
        )

        subs["residx"].scatter(
            x, y, c=dx - predx, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1
        )
        subs["residy"].scatter(
            x, y, c=dy - predy, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1
        )

        cb = fig.colorbar(
            im, ax=[subs["dx"], subs["dy"], subs["predx"], subs["predy"], subs["residx"], subs["residy"]]
        )

        subs["eb"].scatter(np.exp(logr) * 60, xie, c="b", label="E-mode")
        subs["eb"].scatter(np.exp(logr) * 60, xib, c="r", label="B-mode")

        subs["eb"].scatter(
            np.exp(logr_resid) * 60, xie_resid, c="b", marker="+", label="E-mode after GP correction"
        )
        subs["eb"].scatter(
            np.exp(logr_resid) * 60, xib_resid, c="r", marker="+", label="B-mode after GP correction"
        )
        subs["eb"].legend()
        subs["eb"].grid(True)

        subs["dx"].set_aspect("equal")
        subs["dy"].set_aspect("equal")
        subs["predx"].set_aspect("equal")
        subs["predy"].set_aspect("equal")
        subs["residx"].set_aspect("equal")
        subs["residy"].set_aspect("equal")
        subs["dy"].set_xlabel("x (degree)")
        subs["predy"].set_xlabel("x (degree)")
        subs["residy"].set_xlabel("x (degree)")
        subs["dy"].set_ylabel("y (degree)")
        subs["dx"].set_ylabel("y (degree)")

        subs["dx"].set_title(r"$\delta$x")
        subs["predx"].set_title("GP prediction")
        subs["residx"].set_title("Residual")

        subs["dy"].set_title(r"$\delta$y")
        subs["predy"].set_title("GP prediction")
        subs["residy"].set_title("Residual")

        cb.set_label("mas")

        subs["eb"].set_title("E and B modes")
        subs["eb"].set_ylabel(r"$\xi_{E/B}$ (mas$^2$)")
        subs["eb"].set_xlabel(r"$\Delta \theta$ (arcmin)")

        return fig

    def evaluate(self, gpx, gpy, trainInd, testInd, x, y, dx, dy, residError):
        """Will do eb correlation before and after here, and validate
        prediction on some of the test data.

        Just copied some stuff over here for now.
        """

        coord = np.array([x, y]).T

        # Calculate E/B modes before and after Gaussian Processes correction.
        xPredict = gpx.predict(coord[trainInd])
        yPredict = gpy.predict(coord[trainInd])
        xie, xib, logr = treegp.comp_eb_treecorr(
            x[trainInd], y[trainInd], dx[trainInd], dy[trainInd], rmin=20 / 3600, rmax=0.6, dlogr=0.3
        )
        start, stop = np.searchsorted(np.exp(logr), [0, 15])
        meanE = np.mean(xie[start:stop])
        meanB = np.mean(xib[start:stop])
        self.log.info(
            "Original average correlation level over 0-15 arcminutes: E-mode=%0.2f, B-mode=%0.2f",
            meanE,
            meanB,
        )

        xie_resid, xib_resid, logr = treegp.comp_eb_treecorr(
            x[trainInd],
            y[trainInd],
            dx[trainInd] - xPredict,
            dy[trainInd] - yPredict,
            rmin=20 / 3600,
            rmax=0.6,
            dlogr=0.3,
        )
        start, stop = np.searchsorted(np.exp(logr), [0, 15])
        meanE_resid = np.mean(xie_resid[start:stop])
        meanB_resid = np.mean(xib_resid[start:stop])
        self.log.info(
            "Correlation level after GP correction over 0-15 arcminutes: E-mode=%0.2f, B-mode=%0.2f",
            meanE_resid,
            meanB_resid,
        )

        # Predict on all test data and make a plot.
        if self.config.doValidation:
            chunkSize = 5000
            nChunks = np.ceil(len(testInd) / chunkSize).astype(int)
            xPredict = np.zeros(len(testInd))
            yPredict = np.zeros(len(testInd))
            for i in tqdm(range(nChunks)):
                ind = testInd[chunkSize * i : chunkSize * (i + 1)]
                xPredict[chunkSize * i : chunkSize * (i + 1)] = gpx.predict(coord[ind])
                yPredict[chunkSize * i : chunkSize * (i + 1)] = gpy.predict(coord[ind])
            fig = self.plot_visit(x[testInd], y[testInd], dx[testInd], dy[testInd], xPredict, yPredict)
            # I don't really want to make an official output figure. We can
            # just save the figure locally for development work.
            # fig.savefig(Fill in something here)
