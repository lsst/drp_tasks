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

__all__ = [
    "AggregateCoaddInputsTractConnections",
    "AggregateCoaddInputsTractConfig",
    "AggregateCoaddInputsTract",
    "AggregateCoaddInputsConnections",
    "AggregateCoaddInputsConfig",
    "AggregateCoaddInputs",
]

from astropy.table import Table
import numpy as np

from lsst.obs.base import TableVStack
import lsst.pipe.base
import lsst.pex.config


class AggregateCoaddInputsTractConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("skymap", "tract", "band"),
):
    coadd_inputs_handles = lsst.pipe.base.connectionTypes.Input(
        name="deep_coadd_predetection.coaddInputs",
        doc="Coadd inputs.",
        storageClass="CoaddInputs",
        dimensions=("skymap", "tract", "patch", "band"),
        deferLoad=True,
        multiple=True,
    )
    aggregate_tract_output = lsst.pipe.base.connectionTypes.Output(
        name="deep_coadd_inputs_aggregate_tract",
        doc="Aggregated coadd inputs (by tract and band).",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "band"),
    )


class AggregateCoaddInputsTractConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=AggregateCoaddInputsTractConnections,
):
    pass


class AggregateCoaddInputsTract(lsst.pipe.base.PipelineTask):
    """Task to aggregate coadd inputs by tract/band."""

    ConfigClass = AggregateCoaddInputsTractConfig
    _DefaultName = "aggreate_coadd_inputs_tract"

    def run(self, *, coadd_inputs_handles):
        """Run the AggregateCoaddInputsTract task.

        Parameters
        ----------
        coadd_inputs : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            List of coadd input handles to aggregate.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct containing:
                ``aggregate_tract_output`` : `astropy.table.Table`
        """
        self.log.info("Aggregating %d coadd input catalogs", len(coadd_inputs_handles))
        n_inputs = 0
        detector_table_dict = {}
        bands = set()
        for coadd_inputs_handle in coadd_inputs_handles:
            data_id = coadd_inputs_handle.dataId
            tract = data_id["tract"]
            patch = data_id["patch"]
            band = data_id["band"]

            bands.add(band)

            coadd_inputs = coadd_inputs_handle.get()

            detector_table_dict[(tract, patch, band)] = coadd_inputs.ccds
            n_inputs += len(coadd_inputs.ccds)

        self.log.info("Found %d coadd input rows to aggregate", n_inputs)

        aggregate_tract_output = Table()
        aggregate_tract_output["tract"] = np.zeros(n_inputs, dtype=np.int32)
        aggregate_tract_output["patch"] = np.zeros(n_inputs, dtype=np.int32)
        aggregate_tract_output["visit"] = np.zeros(n_inputs, dtype=np.int64)
        aggregate_tract_output["detector"] = np.zeros(n_inputs, dtype=np.int32)
        aggregate_tract_output["weight"] = np.zeros(n_inputs)
        aggregate_tract_output["band"] = np.zeros(n_inputs, dtype="U3")

        counter = 0
        for (tract, patch, band), detectors in detector_table_dict.items():
            ndet = len(detectors)

            aggregate_tract_output["tract"][counter: counter + ndet] = tract
            aggregate_tract_output["patch"][counter: counter + ndet] = patch
            aggregate_tract_output["band"][counter: counter + ndet] = band

            aggregate_tract_output["visit"][counter: counter + ndet] = detectors["visit"]
            aggregate_tract_output["detector"][counter: counter + ndet] = detectors["ccd"]
            aggregate_tract_output["weight"][counter: counter + ndet] = detectors["weight"]

            counter += ndet

        return lsst.pipe.base.Struct(aggregate_tract_output=aggregate_tract_output)


class AggregateCoaddInputsConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("skymap",),
):
    aggregated_tract_handles = lsst.pipe.base.connectionTypes.Input(
        name="deep_coadd_inputs_aggregate_tract",
        doc="Aggregated coadd inputs (by tract and band).",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "band"),
        deferLoad=True,
        multiple=True,
    )
    aggregated_coadd_inputs = lsst.pipe.base.connectionTypes.Output(
        name="deep_coadd_inputs_aggregate",
        doc="Aggregated coadd inputs.",
        storageClass="ArrowAstropy",
        dimensions=("skymap",),
    )


class AggregateCoaddInputsConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=AggregateCoaddInputsConnections,
):
    pass


class AggregateCoaddInputs(lsst.pipe.base.PipelineTask):
    """Task to aggregate aggregated coadd inputs over a full run."""

    ConfigClass = AggregateCoaddInputsConfig
    _DefaultName = "aggregate_coadd_inputs"

    def run(self, *, aggregated_tract_handles):
        """Run the AggregateCoaddInputs task.

        Parameters
        ----------
        aggregated_tract_handles : `list`
            [`lsst.daf.butler.DeferredDatasetHandle`]
            List of aggregated coadd nipust to aggregate.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct containing:
                ``aggregated_coadd_inputs`` : `astropy.table.Table`
        """
        self.log.info("Aggregating %d aggregated coadd input catalogs", len(aggregated_tract_handles))

        aggregated_coadd_inputs = TableVStack.vstack_handles(aggregated_tract_handles)

        return lsst.pipe.base.Struct(aggregated_coadd_inputs=aggregated_coadd_inputs)
