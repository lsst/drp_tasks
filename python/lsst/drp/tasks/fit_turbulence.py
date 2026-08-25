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
import treecorr
import treegp
from astropy.table import Table
from scipy.interpolate import RectBivariateSpline

import lsst.afw.geom as afwgeom
import lsst.afw.table
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS

# We need to explicitly turn off multiprocessing in treecorr which is used
# by treegp.
treecorr.set_max_omp_threads(1)

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


class SingularMatrixError(pipeBase.AlgorithmError):
    """Raised if the Gaussian Processes fit raises a Singular Matrix linear
    algebra error."""

    def __init__(self, nSources) -> None:
        super().__init__("The Gaussian Processes fit failed with a singular matrix linear algebra error.")
        self._nSources = nSources

    @property
    def metadata(self):
        return {
            "nSources": self._nSources,
        }


class NotPositiveDefiniteMatrixError(pipeBase.AlgorithmError):
    """Raised if the Gaussian Processes fit raises a not positive definite
    linear algebra error."""

    def __init__(self, nSources) -> None:
        super().__init__(
            "The Gaussian Processes fit failed with a not-positive-definite linear algebra error."
        )
        self._nSources = nSources

    @property
    def metadata(self):
        return {
            "nSources": self._nSources,
        }


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
    sourceTable = pipeBase.connectionTypes.Output(
        doc=(
            "Per-source table with positions, residuals, and Gaussian Processes predictions in tangent"
            " plane and detector pixel coordinates, along with the training/validation split."
        ),
        name="turbulence_fit_sources",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit", "healpix3"),
    )
    camera = pipeBase.connectionTypes.PrerequisiteInput(
        doc="Input camera, used to transform source positions to the focal plane.",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not self.config.saveSourceTable:
            self.outputs.remove("sourceTable")
            self.prerequisiteInputs.remove("camera")

        if not self.config.healpix:
            self.dimensions.remove("healpix3")
            if self.config.healpix is None:
                extra_dimensions = []
            else:
                extra_dimensions = ["tract", "skymap"]
            self.dimensions.update(extra_dimensions)
            self.inputWcs = dataclasses.replace(
                self.inputWcs, dimensions=["instrument", "visit"] + extra_dimensions
            )
            self.inputPositions = dataclasses.replace(
                self.inputPositions, dimensions=["instrument", "band", "physical_filter"] + extra_dimensions
            )
            self.outputWcs = dataclasses.replace(
                self.outputWcs, dimensions=["instrument", "visit"] + extra_dimensions
            )
            self.hyperparameters = dataclasses.replace(
                self.hyperparameters, dimensions=["instrument", "visit"] + extra_dimensions
            )
            if self.config.saveSourceTable:
                self.sourceTable = dataclasses.replace(
                    self.sourceTable, dimensions=["instrument", "visit"] + extra_dimensions
                )


class GaussianProcessesTurbulenceFitConfig(
    pipeBase.PipelineTaskConfig, pipelineConnections=GaussianProcessesTurbulenceFitConnections
):
    optimizer = pexConfig.ChoiceField(
        dtype=str,
        doc="Gaussian Processes method used to model the astrometric residuals.",
        default="anisotropic",
        allowed={
            "anisotropic": (
                "Fit the hyperparameters of a parametric kernel on the measured"
                " 2-point correlation function (Leget et al. 2021, A&A 650, A81)."
            ),
            "empirical-2pcf": (
                "Use the measured 2-point correlation function directly as the"
                " kernel, with no hyperparameter fit (Gomes et al. 2025,"
                " AJ 170:361)."
            ),
        },
    )
    initKernel = pexConfig.Field(
        dtype=str,
        doc=(
            "The type of function that will be used to modeled spatial correlation."
            " Only used by the 'anisotropic' optimizer."
        ),
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
        doc=(
            "Maximum distance separation in degrees in the computation of the 2-point correlation function."
            " For the 'empirical-2pcf' optimizer, this is also the half width of the 2-point correlation"
            " function grid used as the kernel."
        ),
        default=0.3,
        optional=True,
    )
    correlationPixelSize = pexConfig.Field(
        dtype=float,
        doc=(
            "Pixel size in degrees of the 2-point correlation function grid used as the kernel."
            " Only used by the 'empirical-2pcf' optimizer."
        ),
        default=0.00556,
    )
    powerThreshold = pexConfig.Field(
        dtype=float,
        doc=(
            "Signal-to-noise threshold below which Fourier modes of the measured 2-point correlation"
            " function are set to zero. Only used by the 'empirical-2pcf' optimizer."
        ),
        default=2.5,
    )
    apodize = pexConfig.Field(
        dtype=bool,
        doc=(
            "Whether to apodize the measured 2-point correlation function before taking its Fourier"
            " transform. Only used by the 'empirical-2pcf' optimizer."
        ),
        default=True,
    )
    apodWindow = pexConfig.ChoiceField(
        dtype=str,
        doc="Apodization window function. Only used by the 'empirical-2pcf' optimizer.",
        default="hann",
        allowed={
            "hann": "Hann window; gentler taper at the price of more spectral leakage.",
            "blackman-harris": "Blackman-Harris window, as used in Gomes et al. 2025.",
        },
    )
    apodRadius = pexConfig.Field(
        dtype=float,
        doc=(
            "Radius in degrees where the apodization window reaches zero. If None, the window reaches"
            " zero at correlationSeparationMax. Only used by the 'empirical-2pcf' optimizer."
        ),
        default=0.278,
        optional=True,
    )
    apodAnisotropy = pexConfig.ChoiceField(
        dtype=str,
        doc="Anisotropy of the apodization window. Only used by the 'empirical-2pcf' optimizer.",
        default="auto",
        allowed={
            "auto": (
                "Measure the anisotropy of the 2-point correlation function with adaptive"
                " weighted second moments and use a matched elliptical window."
            ),
            "none": "Isotropic apodization window.",
        },
    )
    apodGScale = pexConfig.Field(
        dtype=float,
        doc=(
            "Factor multiplying the measured anisotropy of the apodization window when"
            " apodAnisotropy='auto'. Only used by the 'empirical-2pcf' optimizer."
        ),
        default=1.0,
    )
    whiteNoise = pexConfig.Field(
        dtype=float,
        doc=(
            "Additional white noise in mas added in quadrature to the residual errors; can regularize"
            " the fit if the Cholesky decomposition fails. Only used by the 'empirical-2pcf' optimizer."
        ),
        default=0.0,
    )
    saveSourceTable = pexConfig.Field(
        dtype=bool,
        doc=(
            "Save the per-source table with positions, residuals, and Gaussian Processes predictions"
            " in tangent plane and detector pixel coordinates."
        ),
        default=False,
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
        optional=True,
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

    def validate(self):
        super().validate()
        if self.optimizer == "empirical-2pcf":
            if self.correlationSeparationMax is None:
                raise pexConfig.FieldValidationError(
                    self.__class__.correlationSeparationMax,
                    self,
                    "correlationSeparationMax must be set for the 'empirical-2pcf' optimizer.",
                )
            if treegp.__version__ != "1.5.0":
                raise NotImplementedError("G25 ++ solver is not supported yet (need treegp==1.5.0).")
            if self.correlationSeparationMax / self.correlationPixelSize < 2:
                raise pexConfig.FieldValidationError(
                    self.__class__.correlationPixelSize,
                    self,
                    "correlationSeparationMax must span at least 2 pixels of correlationPixelSize"
                    " for the 'empirical-2pcf' optimizer.",
                )


class GaussianProcessesTurbulenceFitTask(pipeBase.PipelineTask):
    """Run Gaussian Processes on astrometric residuals with the assumption that
    they are due to atmospheric turbulence.

    Two methods are available, selected with `config.optimizer`: fitting the
    hyperparameters of a parametric kernel on the measured 2-point correlation
    function (Leget et al. 2021, A&A 650, A81), or using the measured 2-point
    correlation function directly as the kernel (Gomes et al. 2025,
    AJ 170:361).
    """

    ConfigClass = GaussianProcessesTurbulenceFitConfig
    _DefaultName = "gaussianProcessesTurbulenceFit"

    def run(self, inputWcs, inputPositions, camera=None):
        """Run Gaussian Processes on position residuals and subtract the fitted
        Gaussian Processes prediction from the WCS to account for atmospheric
        turbulence.

        Parameters
        ----------
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        inputPositions : `astropy.table.Table`
            Catalog of input positions with residuals to the current best fit.
        camera : `lsst.afw.cameraGeom.Camera`, optional
            Camera object, used to transform source positions to the focal
            plane. Only provided when `config.saveSourceTable` is set.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            ``outputWcs`` : `lsst.afw.table.ExposureCatalog`
                Catalog with WCS after inserting the correction for atmospheric
                turbulence.
            ``hyperparameters`` : `astropy.table.Table`
                Table of best-fit hyperparameters in x and y-directions.
            ``sourceTable`` : `astropy.table.Table`
                Per-source table with positions, residuals, and Gaussian
                Processes predictions in tangent plane and detector pixel
                coordinates. Only set when `config.saveSourceTable` is set.
        """

        visit = inputWcs[0]["visit"]

        columns = [
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
        inputPositions = inputPositions.get(parameters={"columns": columns})

        visitPositions = inputPositions[
            (inputPositions["exposureName"] == str(visit)) & ~inputPositions["clip"]
        ]

        gpx, gpy, trainInd, testInd, hyperparameters, allTPCoords = self.runGP(inputWcs, visitPositions)

        self.evaluate(gpx, gpy, visitPositions, trainInd, testInd, inputWcs)

        wcsWithSpline = self.addGPToWcs(gpx, gpy, inputWcs)

        outputs = {"outputWcs": wcsWithSpline, "hyperparameters": hyperparameters}
        if self.config.saveSourceTable:
            outputs["sourceTable"] = self.makeSourceTable(
                gpx, gpy, visitPositions, allTPCoords, trainInd, inputWcs, camera, visit
            )

        return pipeBase.Struct(**outputs)

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
        hyperparameters : `astropy.table.Table`
            Table with an ``x`` and a ``y`` column, one per residual
            direction. For the 'anisotropic' optimizer, they hold the
            best-fit kernel hyperparameters. For the 'empirical-2pcf'
            optimizer, they hold diagnostics of the empirical kernel, in
            the order [xi0, g1 measured, g2 measured, g1 applied,
            g2 applied, pixel size, number of grid pixels], where xi0 is
            the zero-lag variance in mas^2 and the g values describe the
            anisotropy of the apodization window (NaN if not measured).
        allTPCoords : `numpy.ndarray`
            Tangent plane coordinates of the input positions in degrees.
            (n_samples, 2)
        """
        if self.config.optimizer == "empirical-2pcf" and not hasattr(treegp, "empirical_2pcf"):
            raise RuntimeError(
                "The version of treegp in this environment does not support the 'empirical-2pcf'"
                " optimizer (Gomes et al. 2025); it requires the treegp branch tickets/DM-55875."
            )

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

        if self.config.optimizer == "anisotropic":
            gpKwargs = dict(
                kernel=self.config.initKernel,
                optimizer="anisotropic",
                normalize=True,
                nbins=21,
                min_sep=self.config.correlationSeparationMin,
                max_sep=self.config.correlationSeparationMax,
                p0=self.config.initAnisotropicCorrelationLength,
            )
        else:
            gpKwargs = dict(
                optimizer="empirical-2pcf",
                normalize=True,
                max_sep=self.config.correlationSeparationMax,
                pixel_size=self.config.correlationPixelSize,
                power_threshold=self.config.powerThreshold,
                apodize=self.config.apodize,
                apod_window=self.config.apodWindow,
                apod_radius=self.config.apodRadius,
                apod_anisotropy=("auto" if self.config.apodAnisotropy == "auto" else None),
                apod_g_scale=self.config.apodGScale,
                white_noise=self.config.whiteNoise,
            )

        # Solve Gaussian Processes in dx direction.
        gpx = treegp.GPInterpolation(**gpKwargs)
        gpx.initialize(allTPCoords[trainInds], dx[trainInds], y_err=dxErr[trainInds])

        # Solve Gaussian Processes in dy direction.
        gpy = treegp.GPInterpolation(**gpKwargs)
        gpy.initialize(allTPCoords[trainInds], dy[trainInds], y_err=dyErr[trainInds])

        try:
            gpx.solve()
            gpy.solve()
        except np.linalg.LinAlgError as e:
            if "Singular matrix" in str(e):
                error = pipeBase.AnnotatedPartialOutputsError.annotate(
                    SingularMatrixError(len(allTPCoords[trainInds])),
                    self,
                    log=self.log,
                )
                raise error from e
            elif "not positive definite" in str(e):
                error = pipeBase.AnnotatedPartialOutputsError.annotate(
                    NotPositiveDefiniteMatrixError(len(allTPCoords[trainInds])),
                    self,
                    log=self.log,
                )
                raise error from e
            else:
                raise

        if self.config.optimizer == "anisotropic":
            hyperparameters = Table(
                {
                    "x": np.array(gpx._optimizer._results_robust),
                    "y": np.array(gpy._optimizer._results_robust),
                }
            )
        else:
            hyperparameters = Table(
                {
                    "x": self._empiricalDiagnostics(gpx),
                    "y": self._empiricalDiagnostics(gpy),
                }
            )

        return gpx, gpy, trainInds, testInds, hyperparameters, allTPCoords

    @staticmethod
    def _empiricalDiagnostics(gp):
        """Summarize the empirical-2pcf kernel of a solved Gaussian Processes
        interpolator as a fixed-order array of floats.

        Parameters
        ----------
        gp : `treegp.gp_interp.GPInterpolation`
            Solved interpolator with optimizer 'empirical-2pcf'.

        Returns
        -------
        diagnostics : `numpy.ndarray`
            [xi0, g1 measured, g2 measured, g1 applied, g2 applied,
            pixel size, number of grid pixels].
        """
        solver = gp._optimizer
        gMeasured = solver._apod_g_measured
        if gMeasured is None:
            gMeasured = (np.nan, np.nan)
        gApplied = solver._apod_g_applied
        return np.array(
            [
                gp.kernel.xi0,
                gMeasured[0],
                gMeasured[1],
                gApplied[0],
                gApplied[1],
                solver.pixel_size,
                solver.npix,
            ],
            dtype=float,
        )

    def makeSourceTable(self, gpx, gpy, positions, allTPCoords, trainInds, inputWcs, camera, visit):
        """Build the per-source table with positions, residuals, and Gaussian
        Processes predictions in tangent plane and detector pixel coordinates.

        The pixel-frame offsets are expressed in each detector's own pixel
        axes: the tangent plane residuals (and predictions) are mapped back
        to detector pixels through the WCS, so that ``dxPix``/``dyPix`` match
        the difference between the observed pixel position and the pixel
        position of the model.

        Parameters
        ----------
        gpx : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for x-direction residuals.
        gpy : `treegp.gp_interp.GPInterpolation`
            Gaussian Processes interpolator for y-direction residuals.
        positions : `astropy.table.Table`
            Catalog of input positions with residuals to the best fit.
        allTPCoords : `numpy.ndarray`
            Tangent plane coordinates of `positions` in degrees.
            (n_samples, 2)
        trainInds : `numpy.ndarray`
            Array of indices for points used in training.
        inputWcs : `lsst.afw.table.ExposureCatalog`
            Catalog with WCSs for each detector of the input exposure.
        camera : `lsst.afw.cameraGeom.Camera`
            Camera object, used to transform source positions to the focal
            plane.
        visit : `int`
            Visit number.

        Returns
        -------
        sourceTable : `astropy.table.Table`
            Table with one row per source, containing the visit, detector,
            tangent plane coordinates (degrees), measured residuals and
            Gaussian Processes predictions in the tangent plane (mas), the
            training/validation flag, the detector pixel and focal plane
            (mm) positions, and the measured residuals and Gaussian
            Processes predictions in detector pixel coordinates.
        """
        nSources = len(positions)
        isTraining = np.zeros(nSources, dtype=bool)
        isTraining[trainInds] = True

        prediction = np.zeros((nSources, 2))
        focalPlaneCoords = np.zeros((nSources, 2))
        pixelResiduals = np.zeros((nSources, 2))
        pixelPredictions = np.zeros((nSources, 2))

        chunkSize = 10000
        for detector in inputWcs:
            detId = detector["id"]
            detWCS = detector.wcs
            detInd = np.flatnonzero(positions["deviceName"].astype(int) == detId)
            if len(detInd) == 0:
                continue

            # Predict the Gaussian Processes in the tangent plane, in
            # chunks to limit the size of the cross-covariance matrix.
            for start in range(0, len(detInd), chunkSize):
                ind = detInd[start : start + chunkSize]
                prediction[ind, 0] = gpx.predict(allTPCoords[ind])
                prediction[ind, 1] = gpy.predict(allTPCoords[ind])

            pixObserved = np.array([positions["xpix"][detInd], positions["ypix"][detInd]])

            # Positions on the focal plane, in mm.
            pixelsToFocalPlane = camera[detId].getTransform(PIXELS, FOCAL_PLANE).getMapping()
            focalPlaneCoords[detInd] = pixelsToFocalPlane.applyForward(pixObserved).T

            # Express the tangent plane residuals and predictions as
            # offsets in detector pixel coordinates: the pixel position of
            # the model is the observed tangent plane position minus the
            # residual, mapped back through the WCS.
            pixelsToTangentPlane = detWCS.getFrameDict().getMapping("PIXELS", "IWC")
            residDeg = (
                (np.array([positions["xresw"][detInd], positions["yresw"][detInd]]) * u.mas)
                .to(u.degree)
                .value
            )
            # astshim mappings require C-contiguous arrays.
            pixModel = pixelsToTangentPlane.applyInverse(
                np.ascontiguousarray(allTPCoords[detInd].T - residDeg)
            )
            pixelResiduals[detInd] = (pixObserved - pixModel).T

            predictionDeg = (prediction[detInd].T * u.mas).to(u.degree).value
            pixPredicted = pixelsToTangentPlane.applyInverse(
                np.ascontiguousarray(allTPCoords[detInd].T - predictionDeg)
            )
            pixelPredictions[detInd] = (pixObserved - pixPredicted).T

        sourceTable = Table(
            {
                "visit": np.full(nSources, visit, dtype=np.int64),
                "detector": positions["deviceName"].astype(int),
                "xTP": allTPCoords[:, 0],
                "yTP": allTPCoords[:, 1],
                "dxTP": positions["xresw"],
                "dyTP": positions["yresw"],
                "dxTPGP": prediction[:, 0],
                "dyTPGP": prediction[:, 1],
                "isTraining": isTraining,
                "xPix": positions["xpix"],
                "yPix": positions["ypix"],
                "fpX": focalPlaneCoords[:, 0],
                "fpY": focalPlaneCoords[:, 1],
                "dxPix": pixelResiduals[:, 0],
                "dyPix": pixelResiduals[:, 1],
                "dxPixGP": pixelPredictions[:, 0],
                "dyPixGP": pixelPredictions[:, 1],
            }
        )

        return sourceTable

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
            tx, ty = splineX.get_knots()
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
