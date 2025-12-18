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

import astropy.units as u
import astshim as ast
import matplotlib.pyplot as plt
import numpy as np
import treegp
from astropy.table import Table
from scipy.interpolate import RectBivariateSpline

import lsst.afw.geom as afwgeom
import lsst.afw.table
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

__all__ = [
    "GaussianProcessesTurbulenceFitConnections",
    "GaussianProcessesTurbulenceFitConfig",
    "GaussianProcessesTurbulenceFitTask",
]


def plot_visit(x, y, dx, dy, predx, predy):
    """Utility function for plotting Gaussian Processes results.

    Parameters
    ----------
    x : `np.ndarray`
        x-direction coordinates.
    y : `np.ndarray`
        y-direction coordinates.
    dx : `np.ndarray`
        x-direction residuals to be fit.
    dy : `np.ndarray`
        x-direction residuals to be fit.
    predx : `np.ndarray`
        x-direction prediction.
    predy : `np.ndarray`
        y-direction prediction.

    Returns
    -------
    fig : `matplotlib.pyplot.Figure`
        Figure showing input data, Gaussian Processes prediction, and E and
        B-modes.
    """

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

    subs["predx"].scatter(x, y, c=predx, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)
    subs["predy"].scatter(x, y, c=predy, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)

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


class GaussianProcessesTurbulenceFitConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "visit", "healpix3"),
    defaultTemplates={
        "inputName": "gbdesHealpix3AstrometricFit",
    },
):
    inputWcs = pipeBase.connectionTypes.Input(
        doc=(
            "Per-healpix, per-visit world coordinate systems derived from the fitted model."
            " These catalogs only contain entries for detectors with an output, and use"
            " the detector id for the catalog id, sorted on id for fast lookups of a detector."
        ),
        name="{inputName}SkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "healpix3"),
    )
    inputPositions = pipeBase.connectionTypes.Input(
        doc=(
            "Catalog of sources used in fit, along with residuals in pixel coordinates and tangent "
            "plane coordinates and chisq values."
        ),
        name="{inputName}_fitStars",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "healpix3", "physical_filter"),
        deferLoad=True,
    )
    outputWcs = pipeBase.connectionTypes.Output(
        doc=(
            "Per-visit world coordinate systems derived from the fitted model. These catalogs only contain "
            "entries for detectors with an output, and use the detector id for the catalog id, sorted on id "
            "for fast lookups of a detector."
        ),
        name="turbulenceCorrectedSkyWcsCatalog",
        storageClass="ExposureCatalog",
        dimensions=("instrument", "visit", "healpix3"),
    )
    hyperparameters = pipeBase.connectionTypes.Output(
        doc="Best fit hyperparameters for the Gaussian Processes fit.",
        name="turbulence_fit_hyperparameters",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "healpix3"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not self.config.healpix:
            self.dimensions.remove("healpix3")
            self.dimensions.add("tract")
            self.dimensions.add("skymap")
            self.inputWcs = dataclasses.replace(
                self.inputWcs, dimensions=("instrument", "visit", "tract", "skymap")
            )
            self.inputPositions = dataclasses.replace(
                self.inputPositions, dimensions=("instrument", "tract", "skymap", "band", "physical_filter")
            )


class GaussianProcessesTurbulenceFitConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=GaussianProcessesTurbulenceFitConnections
):
    initKernel = pexConfig.Field(
        dtype=str,
        doc="The type of function that will be used to modeled spatial correlation.",
        default="15**2 * AnisotropicVonKarman(invLam=array([[1./0.8**2,0],[0,1./0.8**2]]))",
    )
    initAnisotropicCorrelationLength = pexConfig.ListField(
        dtype=float,
        doc=(
            "The initial parameters for fiting the anisotropic correlation length. p0[0] is equivalent of "
            "the isotropic correlation length in degrees, and p0[1]/p0[2] are ellipticity parameters and are "
            "mathematically equivalent to e1/e2 in weak-lensing. p0[1]/p0[2] must be in the range [-1,1], "
            "where 0 means the correlation is isotropic."
        ),
        default=[1, -0.2, -0.2],
    )
    correlationSeparationMin = pexConfig.Field(
        dtype=float,
        doc="Minimum distance separation in degrees in the computation of the 2-point correlation function.",
        default=0.0,
        optional=True,
    )
    correlationSeparationMax = pexConfig.Field(
        dtype=float,
        doc="Maximum distance separation in degrees in the computation of the 2-point correlation function.",
        default=0.3,
        optional=True,
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
    healpix = pexConfig.Field(
        dtype=bool,
        doc="Use input WCS calculated over healpix-based region. If false, use tract-based WCS.",
        default=True,
    )
    splineDegree = pexConfig.Field(
        dtype=int,
        doc="Degree of the spline expressing Gaussian Processes prediction.",
        default=4,
    )
    splineNNodes = pexConfig.Field(
        dtype=int,
        doc="Number of nodes to use for the spline expressing Gaussian Processes prediction.",
        default=30,
    )
    splineBuffer = pexConfig.Field(
        dtype=float,
        doc="Minimum distance in degrees to extend spline map outside the detector boundary.",
        default=0.1,
    )


class GaussianProcessesTurbulenceFitTask(pipeBase.PipelineTask):
    """Run Gaussian Processes on astrometric residuals with the assumption that
    they are due to atmospheric turbulence."""

    ConfigClass = GaussianProcessesTurbulenceFitConfig
    _DefaultName = "gaussianProcessesTurbulenceFit"

    def run(self, inputWcs, inputPositions):
        """Run Gaussian Processes on position residuals and subtract the fitted
        Gaussian Processes prediction from the WCS to account for atmospheric
        turbulence.

        Parameters
        ----------
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        inputPositions : `astropy.table.Table`
            Catalog of input positions with residuals to the current best fit.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``outputWcs`` : `lsst.afw.table.ExposureCatalog`
                Catalog with WCS after inserting the correction for atmospheric
                turbulence.
            ``hyperparameters`` : `astropy.table.Table`
                Table of best-fit hyperparameters in x and y-directions.
        """

        visit = inputWcs[0]["visit"]

        inputPositions = inputPositions.get(
            parameters={
                "columns": [
                    "xworld",
                    "yworld",
                    "xresw",
                    "yresw",
                    "exposureName",
                    "xpix",
                    "ypix",
                    "deviceName",
                    "clip",
                    "covTotalW_00",
                    "covTotalW_11",
                ]
            }
        )

        visitPositions = inputPositions[
            (inputPositions["exposureName"] == str(visit)) & ~inputPositions["clip"]
        ]

        gpx, gpy, trainInd, testInd, hyperparameters = self.runGP(inputWcs, visitPositions)

        self.evaluate(gpx, gpy, visitPositions, trainInd, testInd, inputWcs)

        wcsWithSpline = self.addGPToWcs(gpx, gpy, inputWcs)

        return pipeBase.Struct(outputWcs=wcsWithSpline, hyperparameters=hyperparameters)

    def runGP(self, inputWcs, positions):
        """Run Gaussian Processes in tangent plane coordinates.

        Parameters
        ----------
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        inputPositions : `astropy.table.Table`
            Catalog of input positions with residuals to the current best fit.

        Returns
        -------
        gpx : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for x-direction residuals.
        gpy : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for y-direction residuals.
        trainInds : `numpy.ndarray`
            Array of indices for points used in training.
        testInds : `numpy.ndarray`
            Array of indices for points not used in training.
        """
        dx = positions["xresw"]
        dy = positions["yresw"]
        dxErr = positions["covTotalW_00"] ** 0.5
        dyErr = positions["covTotalW_11"] ** 0.5

        # Get tangent plane coordinates for input points
        allTPCoords = np.zeros((len(positions), 2))
        for detector in inputWcs:
            detId = detector["id"]
            detWCS = detector.wcs
            detInd = positions["deviceName"].astype(int) == detId
            detectorSources = positions[detInd]

            tangentPlaneToSky = detWCS.getFrameDict().getMapping("PIXELS", "IWC")
            tangentPlaneCoords = tangentPlaneToSky.applyForward(
                np.array([detectorSources["xpix"], detectorSources["ypix"]])
            )
            allTPCoords[detInd] = tangentPlaneCoords.T

        # Choose a random subset for training.
        rng = np.random.default_rng(1234)
        nPoints = len(allTPCoords)
        nTrain = min([nPoints, self.config.maxTrainingPoints])
        perm = rng.permutation(np.arange(nPoints))
        trainInds = perm[:nTrain]
        testInds = perm[nTrain:]

        # Solve Gaussian Processes in dx direction.
        gpx = treegp.GPInterpolation(
            kernel=self.config.initKernel,
            optimizer="anisotropic",
            normalize=True,
            nbins=21,
            min_sep=self.config.correlationSeparationMin,
            max_sep=self.config.correlationSeparationMax,
            p0=self.config.initAnisotropicCorrelationLength,
        )

        gpx.initialize(allTPCoords[trainInds], dx[trainInds], y_err=dxErr[trainInds])
        gpx.solve()

        # Solve Gaussian Processes in dy direction.
        gpy = treegp.GPInterpolation(
            kernel=self.config.initKernel,
            optimizer="anisotropic",
            normalize=True,
            nbins=21,
            min_sep=self.config.correlationSeparationMin,
            max_sep=self.config.correlationSeparationMax,
            p0=self.config.initAnisotropicCorrelationLength,
        )

        gpy.initialize(allTPCoords[trainInds], dy[trainInds], y_err=dyErr[trainInds])
        gpy.solve()

        hyperparameters = Table(
            {"x": np.array(gpx._optimizer._results_robust), "y": np.array(gpy._optimizer._results_robust)}
        )

        return gpx, gpy, trainInds, testInds, hyperparameters

    def predict(self, gpx, gpy, inputWcs, sourceCatalog):
        """Get the positions for sources after correction for atmospheric
        turbulence.

        Parameters
        ----------
        gpx : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for x-direction residuals.
        gpy : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for y-direction residuals.
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        inputPositions : `astropy.table.Table`
            Catalog of input positions with residuals to the current best fit.

        Returns
        -------
        outCat : `astropy.table.Table`
            Catalog matching `inputPositions`, with `coord_ra` and `coord_dec`
            columns corrected for atmospheric turbulence.
        """
        correctedCoordinates = np.zeros((len(sourceCatalog), 2))
        prediction = np.zeros((len(sourceCatalog), 2))
        allTPCoords = np.zeros((len(sourceCatalog), 2))
        allCoords = np.zeros((len(sourceCatalog), 2))

        for detector in inputWcs:
            detId = detector["id"]
            detWCS = detector.wcs
            detInd = sourceCatalog["detector"] == detId
            detectorSources = sourceCatalog[detInd]

            # The Gaussian Processes is fit on the tangent plane coordinates,
            # so we must transform points to the tangent plane, then subtract
            # the effect of atmospheric turbulence, then transform the tangent
            # plane coordinates to sky coordinates.
            initialSky = detWCS.pixelToSkyArray(detectorSources["x"], detectorSources["y"])
            allCoords[detInd] = np.array(initialSky).T
            tangentPlaneToSky = detWCS.getFrameDict().getMapping("IWC", "SKY")
            tangentPlaneCoords = tangentPlaneToSky.applyInverse(np.array(initialSky)).T
            allTPCoords[detInd] = tangentPlaneCoords

            xPred = gpx.predict(tangentPlaneCoords)
            xPrediction = (xPred * u.mas).to(u.degree)
            yPred = gpy.predict(tangentPlaneCoords)
            yPrediction = (yPred * u.mas).to(u.degree)
            prediction[detInd, 0] = xPred
            prediction[detInd, 1] = yPred

            correctedTangentPlaneX = tangentPlaneCoords[:, 0] * u.degree - xPrediction
            correctedTangentPlaneY = tangentPlaneCoords[:, 1] * u.degree - yPrediction
            correctedSkyCoords = tangentPlaneToSky.applyForward(
                np.array([correctedTangentPlaneX, correctedTangentPlaneY])
            )
            correctedCoordinates[detInd] = ((correctedSkyCoords.T) * u.radian).to(u.degree).value

        outCat = sourceCatalog.copy()
        outCat["coord_ra"] = correctedCoordinates[:, 0]
        outCat["coord_dec"] = correctedCoordinates[:, 1]

        return outCat

    def addGPToWcs(self, gpx, gpy, inputWcs):
        """Convert Gaussian Processes prediction to a spline, and insert it in
        the WCS for each detector.

        Parameters
        ----------
        gpx : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for x-direction residuals.
        gpy : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for y-direction residuals.
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.

        Returns
        -------
        catalog : `lsst.afw.table.ExposureCatalog`
            Exposure catalog with the WCS set to the existing WCS plus the
            gaussian processes fit.
        """
        pixelFrame = ast.Frame(2, "Domain=PIXELS")
        tpFrame = ast.Frame(2, "Domain=TP")
        iwcFrame = ast.Frame(2, "Domain=IWC")

        # Set up the schema for the output catalogs
        schema = lsst.afw.table.ExposureTable.makeMinimalSchema()
        schema.addField("visit", type="L", doc="Visit number")

        catalog = lsst.afw.table.ExposureCatalog(schema)
        catalog.resize(len(inputWcs))
        catalog["visit"] = inputWcs["visit"]

        for d, detectorRow in enumerate(inputWcs):
            detId = detectorRow.getId()
            catalog[d].setId(detId)

            # Make a grid of points in tangent plane coordinates.
            bbox = detectorRow.getBBox()
            catalog[d].setBBox(bbox)
            corners = np.array(
                [
                    [bbox.getBeginX(), bbox.getEndX(), bbox.getEndX(), bbox.getBeginX()],
                    [bbox.getBeginY(), bbox.getBeginY(), bbox.getEndY(), bbox.getEndY()],
                ]
            ).astype(float)

            initWcsRow = inputWcs.find(detId)
            pixToTPMap = initWcsRow.wcs.getFrameDict().getMapping("PIXELS", "IWC")
            tpToSky = initWcsRow.wcs.getFrameDict().getMapping("IWC", "SKY")
            skyFrame = initWcsRow.wcs.getFrameDict().getFrame("SKY")
            tangentPlaneX, tangentPlaneY = pixToTPMap.applyForward(corners)

            xs = np.linspace(
                tangentPlaneX.min() - self.config.splineBuffer,
                tangentPlaneX.max() + self.config.splineBuffer,
                self.config.splineNNodes,
            )
            ys = np.linspace(
                tangentPlaneY.min() - self.config.splineBuffer,
                tangentPlaneY.max() + self.config.splineBuffer,
                self.config.splineNNodes,
            )

            xx, yy = np.meshgrid(xs, ys)
            inArray = np.array([xx.ravel(), yy.ravel()]).T

            # Get Gaussian Processes prediction on grid and fit spline to it.
            xPred = (gpx.predict(inArray) * u.mas).to(u.degree).value

            splineX = RectBivariateSpline(
                xs,
                ys,
                (xx - xPred.reshape(self.config.splineNNodes, self.config.splineNNodes)).T,
                s=0,
                kx=self.config.splineDegree - 1,
                ky=self.config.splineDegree - 1,
            )
            (tx, ty) = splineX.get_knots()
            coeffsX = splineX.get_coeffs()

            yPred = (gpy.predict(inArray) * u.mas).to(u.degree).value
            splineY = RectBivariateSpline(
                xs,
                ys,
                (yy - yPred.reshape(self.config.splineNNodes, self.config.splineNNodes)).T,
                s=0,
                kx=self.config.splineDegree - 1,
                ky=self.config.splineDegree - 1,
            )
            coeffsY = splineY.get_coeffs()

            # Turn spline into AST object and insert in new WCS.
            splineMap = ast.SplineMap(
                self.config.splineDegree,
                self.config.splineDegree,
                self.config.splineNNodes,
                self.config.splineNNodes,
                tx,
                ty,
                coeffsX,
                coeffsY,
                options="OutUnit=1",
            )

            newFrameDict = ast.FrameDict(pixelFrame)
            newFrameDict.addFrame("PIXELS", pixToTPMap, tpFrame)
            newFrameDict.addFrame("TP", splineMap, iwcFrame)
            newFrameDict.addFrame("IWC", tpToSky, skyFrame)
            outWcs = afwgeom.SkyWcs(newFrameDict)
            catalog[d].setWcs(outWcs)

        return catalog

    def evaluate(self, gpx, gpy, positions, trainInd, testInd, inputWcs, makeValidationPlot=False):
        """Calculate E and B-modes in the 2-point correlation function before
        and after correcting for atmospheric turbulence, and validate
        prediction on some of the test data.

        Parameters
        ----------
        gpx : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for x-direction residuals.
        gpy : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for y-direction residuals.
        positions : `astropy.table.Table`
            Catalog of input positions with residuals to the best fit.
        trainInds : `numpy.ndarray`
            Array of indices for points used in training.
        testInds : `numpy.ndarray`
            Array of indices for points not used in training.
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        makeValidationPlot : `bool`, optional
            Whether to make a plot showing the prediction on the validation
            data.
        """
        dx = positions["xresw"]
        dy = positions["yresw"]

        # Get tangent plane coordinates for input points
        tpCoords = np.zeros((len(positions), 2))
        for detector in inputWcs:
            detId = detector["id"]
            detWCS = detector.wcs
            detInd = positions["deviceName"].astype(int) == detId
            detectorSources = positions[detInd]

            tangentPlaneToSky = detWCS.getFrameDict().getMapping("PIXELS", "IWC")
            tangentPlaneCoords = tangentPlaneToSky.applyForward(
                np.array([detectorSources["xpix"], detectorSources["ypix"]])
            )
            tpCoords[detInd] = tangentPlaneCoords.T

        # Calculate E/B modes before and after Gaussian Processes correction.
        xPredict = gpx.predict(tpCoords[trainInd])
        yPredict = gpy.predict(tpCoords[trainInd])
        xie, xib, logr = treegp.comp_eb_treecorr(
            tpCoords[trainInd, 0],
            tpCoords[trainInd, 1],
            dx[trainInd],
            dy[trainInd],
            rmin=20 / 3600,
            rmax=0.6,
            dlogr=0.3,
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
            tpCoords[trainInd, 0],
            tpCoords[trainInd, 1],
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
        if makeValidationPlot:
            print(len(testInd))
            testInd = testInd[:50000]
            chunkSize = 5000
            nChunks = np.ceil(len(testInd) / chunkSize).astype(int)
            xPredict = np.zeros(len(testInd))
            yPredict = np.zeros(len(testInd))
            for i in range(nChunks):
                ind = testInd[chunkSize * i : chunkSize * (i + 1)]
                xPredict[chunkSize * i : chunkSize * (i + 1)] = gpx.predict(tpCoords[ind])
                yPredict[chunkSize * i : chunkSize * (i + 1)] = gpy.predict(tpCoords[ind])
            fig = plot_visit(
                tpCoords[testInd, 0], tpCoords[testInd, 1], dx[testInd], dy[testInd], xPredict, yPredict
            )
            return fig
