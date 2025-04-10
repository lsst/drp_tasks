# This file is part of pipe_tasks.
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

import numpy as np
from astropy.table import Table

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class ComputeObjectEpochsConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "skymap", "patch"),
):
    objectCat = pipeBase.connectionTypes.Input(
        doc="Multiband catalog of positions in each patch.",
        name="deepCoadd_obj",
        storageClass="ArrowAstropy",
        dimensions=["skymap", "tract", "patch"],
        deferLoad=True,
    )

    epochMap = pipeBase.connectionTypes.Input(
        doc="Healsparse map of mean epoch of objectCat in each band.",
        name="deepCoadd_epoch_map_mean",
        storageClass="HealSparseMap",
        dimensions=("skymap", "tract", "band"),
        deferLoad=True,
        multiple=True,
    )

    objectEpochs = pipeBase.connectionTypes.Output(
        doc="Catalog of epochs for objectCat objects.",
        name="object_epoch",
        storageClass="ArrowAstropy",
        dimensions=["skymap", "tract", "patch"],
    )


class ComputeObjectEpochsConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=ComputeObjectEpochsConnections,
):
    bands = pexConfig.ListField(
        doc="Bands to create mean epoch columns for",
        dtype=str,
        default=["u", "g", "r", "i", "z", "y"],
    )


class ComputeObjectEpochsTask(pipeBase.PipelineTask):
    """Collect mean epochs for the observations that went into each object.

    TODO: DM-46202, Remove this task once the object epochs are available
    elsewhere.
    """

    ConfigClass = ComputeObjectEpochsConfig
    _DefaultName = "computeObjectEpochs"

    def computeEpochs(self, cat, epochMapDict):
        """Compute the mean epoch of the visits at each object centroid.

        Parameters
        ----------
        cat : `astropy.table.Table`
            Catalog containing object positions.
        epochMapDict: `dict` [`str`, `DeferredDatasetHandle`]
            Dictionary of handles per band for healsparse maps containing
            the mean epoch for positions in the reference catalog.

        Returns
        -------
        epochTable = `astropy.table.Table`
            Catalog with mean epoch of visits at each object position.
        """
        # The primary key should probably stay id and be standardized in the
        # object table later, but this key was used originally and it would
        # be too disruptive to change now.
        allEpochs = {"objectId": cat["id"]}
        for band in self.config.bands:
            epochs = np.ones(len(cat)) * np.nan
            col_ra, col_dec = (str(("meas", band, f"coord_{coord}")) for coord in ("ra", "dec"))
            if col_ra in cat.columns and col_dec in cat.columns:
                ra, dec = cat[col_ra], cat[col_dec]
                validPositions = np.isfinite(ra) & np.isfinite(dec)
                if validPositions.any():
                    ra, dec = (x[validPositions] * (180.0 / np.pi) for x in (ra, dec))
                    epochMap = epochMapDict[band].get()
                    bandEpochs = epochMap.get_values_pos(ra, dec)
                    epochsValid = epochMap.get_values_pos(ra, dec, valid_mask=True)
                    bandEpochs[~epochsValid] = np.nan
                    epochs[validPositions] = bandEpochs
                    del epochMap
            allEpochs[f"{band}_epoch"] = epochs

        epochTable = Table(allEpochs)
        return epochTable

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        inputs["epochMap"] = {ref.dataId["band"]: ref for ref in inputs["epochMap"]}

        objectCatRef = inputs["objectCat"]
        columns_avail = objectCatRef.get(component="columns")
        columns = [
            column
            for band in self.config.bands
            for coord in ["ra", "dec"]
            if str(column := ("meas", band, f"coord_{coord}")) in columns_avail
        ]
        objectCat = objectCatRef.get(parameters={"columns": columns})
        epochs = self.computeEpochs(objectCat, inputs["epochMap"])
        butlerQC.put(epochs, outputRefs.objectEpochs)
