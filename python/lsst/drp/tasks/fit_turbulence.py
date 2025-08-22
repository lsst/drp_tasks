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
    pipeBase.PipelineTaskConnections, dimensions=("instrument", "visit", "skymap", "tract"),
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
    # outputWcs -> same dimensions as inputWcs, but with GP map added
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
        doc=("These are the initial parameters for fiting the anisotropic correlation length. p0[0] is the "
             "equivalent of the isotropic correlation length, and p0[1]/p0[2] are ellipticity parameters and "
             "are mathematically equivalent to e1/e2 in weak-lensing. p0[1]/p0[2] must be in the range "
             "[-1,1], where 0 means the correlation is isotropic."),
        default=[1, -0.2, -0.2],
    )
    correlationSeparationMin = pexConfig.Field(
        dtype=float,
        doc="Minimum distance separation in arcsec in the computation of the 2-point correlation function.",
        default=0.,
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
    

class GaussianProcessesTurbulenceFitTask(pipeBase.PipelineTask):
    """TODO: Run GP on astrometric residuals (assuming they are due to turbulence)
    """

    ConfigClass = GaussianProcessesTurbulenceFitConfig
    _DefaultName = "gaussianProcessesTurbulenceFit"

    def run(self, inputWcs, inputPositions, inputSources):


        visit = str(inputWcs[0]['visit'])

        visitPositions = inputPositions[inputPositions['exposureName'] == visit]

        x = visitPositions['xworld']
        y = visitPositions['yworld']
        dx = visitPositions['xresw']
        dy = visitPositions['yresw']
        residError = (visitPositions['sigpix']*self.config.pixelSize*u.arcsec).to(u.mas).value

        gpx, gpy, trainInd, testInd = self.runGP(x, y, dx, dy, residError)

        #self.evaluate(gpx, gpy, trainInd, testInd, x, y, dx, dy, residError)
        import ipdb; ipdb.set_trace()
        sourceCatalog = inputSources.get(parameters={'columns': ['x', 'y', 'detector']})

        predictedCatalog = self.predict(gpx, gpy, inputWcs, sourceCatalog)

        return pipeBase.Struct(
            outputSources=predictedCatalog
        )
    
    def runGP(self, x, y, dx, dy, residError):
        """TODO
        """
        # Set up the Gaussian Processes.
        
        gpx = treegp.GPInterpolation(kernel=self.config.initKernel,
                                     optimizer='anisotropic',
                                     normalize=True,
                                     nbins=21,
                                     min_sep=self.config.correlationSeparationMin,
                                     max_sep=self.config.correlationSeparationMax,
                                     p0=self.config.initCorrelationLength)

        coord = np.array([x, y]).T
        rng = np.random.default_rng(1234)
        nPoints = len(coord)
        nTrain = max([nPoints, self.config.maxTrainingPoints])
        perm = rng.permutation(np.arange(nPoints))
        trainInds = perm[:nTrain]
        testInds = perm[nTrain:]

        # Predict on du:
        gpx.initialize(coord[trainInds], dx, y_err=residError)
        gpx.solve()

        # Predict on du:
        gpy = treegp.GPInterpolation(kernel=self.config.initKernel,
                                     optimizer='anisotropic',
                                     normalize=True,
                                     nbins=21,
                                     min_sep=self.config.correlationSeparationMin,
                                     max_sep=self.config.correlationSeparationMax,
                                     p0=self.config.initCorrelationLength)

        gpy.initialize(coord[trainInds], dy, y_err=residError)
        gpy.solve()

        return gpx, gpy, trainInds, testInds
    
    def predict(self, gpx, gpy, inputWcs, sourceCatalog):
        """TODO
        """
        correctedCoordinates = np.zeros((len(sourceCatalog), 2))

        for detector in tqdm(inputWcs):
            detId = detector['id']
            detWCS = detector.wcs
            detInd = sourceCatalog['detector'] == detId
            detectorSources = sourceCatalog[detInd]

            initialSky = detWCS.pixelToSkyArray(detectorSources['x'], detectorSources['y'])
            mapToTangent = detWCS.getFrameDict().getMapping("SKY", "IWC")
            tangentPlaneCoords = mapToTangent.applyForward(np.array(initialSky)).T

            xPrediction = gpx.predict(tangentPlaneCoords)
            yPrediction = gpy.predict(tangentPlaneCoords)

            correctedCoordinates[detInd, 0] = tangentPlaneCoords[0] - xPrediction
            correctedCoordinates[detInd, 1] = tangentPlaneCoords[1] - yPrediction

        return correctedCoordinates

    def plot_visit(self, x, y, dx, dy, residualLimit):

        fig, subs = plt.subplots(1, 3)
        subs[0].scatter(x, y, c=dx, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)
        subs[1].scatter(x, y, c=dy, vmin=-residualLimit, vmax=residualLimit, cmap=plt.cm.seismic, s=1)
        
        subs[0].set_aspect("equal")
        subs[1].set_aspect("equal")
        subs[0].set_xlabel('x (degree)')
        subs[0].set_xlabel('y (degree)')
        subs[0].set_ylabel('y (degree)')


    def evaluate(self, gpx, gpy, trainInd, testInd, x, y, dx, dy, residError):
        """Will do eb correlation before and after here, and validate
        prediction on some of the test data.

        Just copied some stuff over here for now.
        """

        RESDIUAL_LIM = np.nanstd(dx)


        xie, xib, logr = treegp.comp_eb_treecorr(u, v, du, dv, rmin=20/3600, rmax=0.6, dlogr=0.3)

        MAXEB = max([np.max(xie), np.max(xib)])
        MINEB = min([np.min(xie), np.min(xib)])
        MAXEB += MAXEB * 0.1
        MINEB -= MINEB * 0.1

        self.MAXEB = MAXEB
        self.MINEB = MINEB

        self.point_size = 1
        self.RESDIUAL_LIM = np.nanstd(self.dic['dx']) * DX_TO_DU_UNITS
        