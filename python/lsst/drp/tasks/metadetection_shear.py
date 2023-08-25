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

from __future__ import annotations

from lsst.daf.butler import DataCoordinate, DatasetRef
from lsst.pipe.base.connectionTypes import BaseInput, Output

__all__ = ()

from collections.abc import Collection, Mapping, Sequence
from typing import Any, ClassVar

import pyarrow as pa

from lsst.pex.config import ListField
from lsst.pipe.base import (
    InputQuantizedConnection,
    NoWorkFound,
    OutputQuantizedConnection,
    QuantumContext,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
import lsst.pipe.base.connectionTypes as cT
from lsst.cell_coadds import MultipleCellCoadd, SingleCellCoadd


class MetadetectionShearConnections(PipelineTaskConnections, dimensions={"patch"}):
    """Definitions of inputs and outputs for MetadetectionShearTask."""

    input_coadds = cT.Input(
        "DeepCoadd",
        storageClass="MultipleCellCoadd",
        doc="Per-band deep coadds.",
        multiple=True,
        dimensions={"patch", "band"},
    )

    # TODO: If there are image-like or other non-catalog output products (e.g.
    # detection masks), add them here.

    object_catalog = cT.Output(
        "ShearObject",
        storageClass="ArrowTable",
        doc="Output catalog with all quantities measured inside the metacalibration loop.",
        multiple=False,
        dimensions={"patch"},
    )
    object_schema = cT.InitOutput(
        "ShearObject_schema",
        # TODO: It's not currently possible to save ArrowSchema objects on
        # their own, but some combination of Eli and Jim can figure out how to
        # fix that.
        storageClass="ArrowSchema",
        doc="Schema of the output catalog.",
    )

    # TODO: if we want a per-cell output catalog instead of just denormalizing
    # everything into per-object catalogs, add it and its schema here.

    config: MetadetectionShearConfig

    def adjustQuantum(
        self,
        inputs: dict[str, tuple[BaseInput, Collection[DatasetRef]]],
        outputs: dict[str, tuple[Output, Collection[DatasetRef]]],
        label: str,
        data_id: DataCoordinate,
    ) -> tuple[
        Mapping[str, tuple[BaseInput, Collection[DatasetRef]]],
        Mapping[str, tuple[Output, Collection[DatasetRef]]],
    ]:
        # Docstring inherited.
        # This is a hook for customizing what is input and output to each
        # invocation of the task as early as possible, which we override here
        # to make sure we have exactly the required bands, no more, no less.
        connection, original_input_coadds = inputs["input_coadds"]
        bands_missing = set(self.config.required_bands)
        adjusted_input_coadds = []
        for ref in original_input_coadds:
            if ref.dataId["band"] in bands_missing:
                adjusted_input_coadds.append(ref)
                bands_missing.remove(ref.dataId["band"])
        if bands_missing:
            raise NoWorkFound(
                f"Required bands {bands_missing} not present for {label}@{data_id})."
            )
        adjusted_inputs = {"input_coadds": (connection, adjusted_input_coadds)}
        inputs.update(adjusted_inputs)
        super().adjustQuantum(inputs, outputs, label, data_id)
        return adjusted_inputs, {}


class MetadetectionShearConfig(
    PipelineTaskConfig, pipelineConnections=MetadetectionShearConnections
):
    """Configuration definition for MetadetectionShearTask."""

    required_bands = ListField[str](
        "Bands expected to be present.  Cells with one or more of these bands "
        "missing will be skipped.  Bands other than those listed here will "
        "not be processed.",
        default=["g", "r", "i", "z"],
        optional=False,
    )

    # TODO: expose more configuration options here.


class MetadetectionShearTask(PipelineTask):
    """A PipelineTask that measures shear using metadetection."""

    _DefaultName: ClassVar[str] = "metadetectionShear"
    ConfigClass: ClassVar[type[MetadetectionShearConfig]] = MetadetectionShearConfig

    config: MetadetectionShearConfig

    def __init__(self, *, initInputs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(initInputs=initInputs, **kwargs)
        self.object_schema = self.make_object_schema(self.config)

    @classmethod
    def make_object_schema(cls, config: MetadetectionShearConfig) -> pa.Schema:
        """Construct a PyArrow Schema for this task's main output catalog.

        Parameters
        ----------
        config : `MetadetectionShearConfig`
            Configuration that may be used to control details of the schema.

        Returns
        -------
        object_schema : `pyarrow.Schema`
            Schema for the object catalog produced by this task.  Each field's
            metadata should include both a 'doc' entry and a 'unit' entry.
        """
        return pa.schema(
            [
                pa.field(
                    "id",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Unique identifier for a ShearObject, specific "
                            "to a single metacalibration counterfactual image."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "tract",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": "ID of the tract on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "patch_x",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": "Column within the tract of the patch on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "patch_y",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": "Row within the tract of the patch on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "cell_x",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": "Column within the patch of the cell on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "cell_y",
                    pa.uint64(),
                    nullable=False,
                    metadata={
                        "doc": "Row within the patch of the cell on which this measurement was made.",
                        "unit": "",
                    },
                ),
                # TODO: add more field definitions here
            ]
        )

    def runQuantum(
        self,
        qc: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # Docstring inherited.
        # Read the coadds and put them in the order defined by
        # config.required_bands (note that each MultipleCellCoadd object also
        # knows its own band, if that's needed).
        coadds_by_band = {
            ref.dataId["band"]: qc.get(ref) for ref in inputRefs.input_coadds
        }
        outputs = self.run([coadds_by_band[b] for b in self.config.required_bands])
        qc.put(outputs, outputRefs)

    def run(self, patch_coadds: Sequence[MultipleCellCoadd]) -> Struct:
        """Run metadetection on a patch.

        Parameters
        ----------
        patch_coadds : `~collections.abc.Sequence` [ \
                `~lsst.cell_coadds.MultipleCellCoadd` ]
            Per-band, per-patch coadds, in the order specified by
            `MetadetectionShearConfig.required_bands`.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Structure with the following attributes:

            - ``object_catalog`` [ `pyarrow.Table` ]: the output object
              catalog for the patch, with schema equal to `object_schema`.
        """
        single_cell_tables: list[pa.Table] = []
        for single_cell_coadds in zip(*patch_coadds, strict=True):
            single_cell_tables.append(self.process_cell(single_cell_coadds))
        # TODO: if we need to do any cell-overlap-region deduplication here
        # (instead of purely in science analysis code), this is where'd it'd
        # happen.
        return Struct(
            object_catalog=pa.concat_tables(single_cell_tables),
        )

    def process_cell(self, single_cell_coadds: Sequence[SingleCellCoadd]) -> pa.Table:
        """Run metadetection on a single cell.

        Parameters
        ----------
        single_cell_coadds : `~collections.abc.Sequence` [ \
                `~lsst.cell_coadds.SingleCellCoadd` ]
            Per-band, per-cell coadds, in the order specified by
            `MetadetectionShearConfig.required_bands`.

        Returns
        -------
        object_catalog : `pyarrow.Table`
            Output object catalog for the cell, with schema equal to
            `object_schema`.
        """
        rows: list[dict[str, Any]] = []
        # TODO: run metadetection on the cell, filling in 'rows' with
        # measurements.  Or replace 'rows' with a 'columns' dict of numpy array
        # columns and call 'from_pydict' instead of 'from_pylist' below, if
        # that's more convenient.
        return pa.Table.from_pylist(rows, self.object_schema)
