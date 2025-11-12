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
    "MakeCoaddInputSummaryTractConnections",
    "MakeCoaddInputSummaryTractConfig",
    "MakeCoaddInputSummaryTract",
    "MakeCoaddInputSummaryConnections",
    "MakeCoaddInputSummaryConfig",
    "MakeCoaddInputSummary",
]

import numpy as np
from astropy.table import Table

import lsst.pex.config
import lsst.pipe.base
from lsst.obs.base import TableVStack


class MakeCoaddInputSummaryTractConnections(
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
    coadd_input_summary_tract = lsst.pipe.base.connectionTypes.Output(
        name="coadd_input_summary_tract",
        doc="Summary table aggregating coaddInputs from multiple coadds (including "
        "all coaddInputs tables in a tract for a single band).",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "band"),
    )


class MakeCoaddInputSummaryTractConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=MakeCoaddInputSummaryTractConnections,
):
    pass


class MakeCoaddInputSummaryTract(lsst.pipe.base.PipelineTask):
    """Task to make coadd input summary by tract/band."""

    ConfigClass = MakeCoaddInputSummaryTractConfig
    _DefaultName = "make_coadd_input_summary_tract"

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
                ``coadd_input_summary_tract`` : `astropy.table.Table`

        Notes
        -----
        The output table contains the the following columns:

        tract : `int`
            The tract number.
        patch : `int`
            The patch number.
        visit : `int`
            The visit number.
        detector : `int`
            The detector number.
        weight : `float`
            The weight that was used as an input to the coadd.
        goodpix : `int`
            The number of pixels from the visit/detector in the patch.
        band : `str`
            The band for the visit.
        """
        self.log.info("Summarizing %d coadd input catalogs", len(coadd_inputs_handles))
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

        self.log.info("Found %d coadd input rows to summarize", n_inputs)

        coadd_input_summary_tract = Table()
        coadd_input_summary_tract["tract"] = np.zeros(n_inputs, dtype=np.int32)
        coadd_input_summary_tract["patch"] = np.zeros(n_inputs, dtype=np.int32)
        coadd_input_summary_tract["visit"] = np.zeros(n_inputs, dtype=np.int64)
        coadd_input_summary_tract["detector"] = np.zeros(n_inputs, dtype=np.int32)
        coadd_input_summary_tract["weight"] = np.zeros(n_inputs)
        coadd_input_summary_tract["goodpix"] = np.zeros(n_inputs, dtype=np.int32)
        coadd_input_summary_tract["band"] = np.zeros(n_inputs, dtype="U3")

        counter = 0
        for (tract, patch, band), detectors in detector_table_dict.items():
            ndet = len(detectors)

            coadd_input_summary_tract["tract"][counter : counter + ndet] = tract
            coadd_input_summary_tract["patch"][counter : counter + ndet] = patch
            coadd_input_summary_tract["band"][counter : counter + ndet] = band

            coadd_input_summary_tract["visit"][counter : counter + ndet] = detectors["visit"]
            coadd_input_summary_tract["detector"][counter : counter + ndet] = detectors["ccd"]
            coadd_input_summary_tract["weight"][counter : counter + ndet] = detectors["weight"]
            coadd_input_summary_tract["goodpix"][counter : counter + ndet] = detectors["goodpix"]

            counter += ndet

        return lsst.pipe.base.Struct(coadd_input_summary_tract=coadd_input_summary_tract)


class MakeCoaddInputSummaryConnections(
    lsst.pipe.base.PipelineTaskConnections,
    dimensions=("skymap",),
):
    coadd_input_summary_tract_handles = lsst.pipe.base.connectionTypes.Input(
        name="coadd_input_summary_tract",
        doc="Summary table aggregating coaddInputs from multiple coadds (including "
        "all coaddInputs tables in a tract for a single band).",
        storageClass="ArrowAstropy",
        dimensions=("skymap", "tract", "band"),
        deferLoad=True,
        multiple=True,
    )
    coadd_input_summary = lsst.pipe.base.connectionTypes.Output(
        name="coadd_input_summary",
        doc="Summary table aggregating coaddInputs from all coadds, including all tracts and bands.",
        storageClass="ArrowAstropy",
        dimensions=("skymap",),
    )


class MakeCoaddInputSummaryConfig(
    lsst.pipe.base.PipelineTaskConfig,
    pipelineConnections=MakeCoaddInputSummaryConnections,
):
    pass


class MakeCoaddInputSummary(lsst.pipe.base.PipelineTask):
    """Task to summarize coadd inputs over a full run."""

    ConfigClass = MakeCoaddInputSummaryConfig
    _DefaultName = "make_coadd_input_summary"

    def run(self, *, coadd_input_summary_tract_handles):
        """Run the MakeCoaddInputSummary task.

        Parameters
        ----------
        coadd_input_summary_tract_handles : `list`
            [`lsst.daf.butler.DeferredDatasetHandle`]
            List of summarized coadd inputs to combine.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Result struct containing:
                ``coadd_input_summary`` : `astropy.table.Table`

        Notes
        -----
        The output table contains the the following columns:

        tract : `int`
            The tract number.
        patch : `int`
            The patch number.
        visit : `int`
            The visit number.
        detector : `int`
            The detector number.
        weight : `float`
            The weight that was used as an input to the coadd.
        goodpix : `int`
            The number of pixels from the visit/detector in the patch.
        band : `str`
            The band for the visit.
        """
        self.log.info("Combining %d summarized coadd input catalogs", len(coadd_input_summary_tract_handles))

        coadd_input_summary = TableVStack.vstack_handles(coadd_input_summary_tract_handles)

        return lsst.pipe.base.Struct(coadd_input_summary=coadd_input_summary)
