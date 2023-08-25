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

__all__ = (
    "MetadetectionProcessingError",
    "MetadetectionShearConfig",
    "MetadetectionShearTask",
)

from collections.abc import Collection, Mapping, Sequence
from itertools import product
from typing import Any, ClassVar

import esutil as eu
import numpy as np
import pyarrow as pa
from metadetect.lsst.masking import apply_apodized_bright_masks_mbexp, apply_apodized_edge_masks_mbexp
from metadetect.lsst.metacal_exposures import STEP as SHEAR_STEP
from metadetect.lsst.metadetect import MetadetectTask
from metadetect.lsst.util import extract_multiband_coadd_data

import lsst.pipe.base.connectionTypes as cT
from lsst.afw.image import ExposureF
from lsst.afw.table import SimpleCatalog
from lsst.cell_coadds import MultipleCellCoadd, StitchedCoadd
from lsst.daf.butler import DataCoordinate, DatasetRef
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader
from lsst.meas.base import FullIdGenerator, SkyMapIdGeneratorConfig
from lsst.pex.config import ConfigField, ConfigurableField, Field, FieldValidationError, ListField
from lsst.pipe.base import (
    AlgorithmError,
    InputQuantizedConnection,
    NoWorkFound,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base.connectionTypes import BaseInput, Output
from lsst.skymap import Index2D


class MetadetectionProcessingError(AlgorithmError):
    """Exception raised when metadetection processing fails."""

    @property
    def metadata(self) -> dict:
        return {}


class MetadetectionShearConnections(PipelineTaskConnections, dimensions={"patch"}):
    """Definitions of inputs and outputs for MetadetectionShearTask."""

    input_coadds = cT.Input(
        "deep_coadd_cell_predetection",
        storageClass="MultipleCellCoadd",
        doc="Per-band deep coadds.",
        multiple=True,
        dimensions={"patch", "band"},
    )

    ref_cat = cT.PrerequisiteInput(
        doc="Reference catalog used to mask bright objects.",
        name="the_monster_20250219",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        deferLoad=True,
        multiple=True,
    )

    metadetect_catalog = cT.Output(
        "object_shear_patch",
        storageClass="ArrowTable",
        doc="Output catalog with all quantities measured inside the metacalibration loop.",
        multiple=False,
        dimensions={"patch"},
    )

    metadetect_schema = cT.InitOutput(
        "object_shear_schema",
        storageClass="ArrowSchema",
        doc="Schema of the output catalog.",
    )

    config: MetadetectionShearConfig

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config:
            return None

        if not config.do_mask_bright_objects:
            del self.ref_cat

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
        bands_missing = set(self.config.photometry_bands)
        adjusted_input_coadds = []
        for ref in original_input_coadds:
            if ref.dataId["band"] in self.config.photometry_bands:
                adjusted_input_coadds.append(ref)
                bands_missing.remove(ref.dataId["band"])
        if missing_shear_bands := bands_missing.intersection(self.config.metadetect.shear_bands):
            raise NoWorkFound(f"Required bands {missing_shear_bands} not present for {label}@{data_id}).")
        adjusted_inputs = {"input_coadds": (connection, adjusted_input_coadds)}
        inputs.update(adjusted_inputs)
        super().adjustQuantum(inputs, outputs, label, data_id)
        return adjusted_inputs, {}


class MetadetectionShearConfig(PipelineTaskConfig, pipelineConnections=MetadetectionShearConnections):
    """Configuration definition for MetadetectionShearTask."""

    metadetect = ConfigurableField(
        target=MetadetectTask,
        doc="Configuration for metadetection.",
    )

    photometry_bands = ListField[str](
        "Bands expected to be present. Cells with one or more of these bands "
        "missing will be skipped. Bands other than those listed here will "
        "not be processed.",
        default=["g", "r", "i", "z"],
    )

    do_mask_bright_objects = Field[bool](
        doc="Mask bright objects in coadds?",
        default=False,
    )

    ref_loader = ConfigField(
        dtype=LoadReferenceObjectsConfig,
        doc="Reference object loader used for bright-object masking.",
    )

    ref_loader_filter_name = Field[str](
        "Filter name from ref_loader used for bright-object masking.",
        default="monster_DES_r",
    )

    border = Field[int](
        "Border to apply to single cell images",
        default=50,
    )

    id_generator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.metadetect.shear_bands = ["r", "i", "z"]
        self.metadetect.metacal.types = ["noshear", "1p", "1m", "2p", "2m"]

    def validate(self):
        super().validate()
        if (shear_bands := self.metadetect.shear_bands) is not None and not set(shear_bands).issubset(
            self.photometry_bands
        ):
            raise FieldValidationError(
                self.__class__.metadetect,
                self,
                "photometry_bands must be a list of bands that is a superset of metadetect.shear_bands",
            )


class MetadetectionShearTask(PipelineTask):
    """A PipelineTask that measures shear using metadetection."""

    _DefaultName: ClassVar[str] = "metadetectionShear"
    ConfigClass: ClassVar[type[MetadetectionShearConfig]] = MetadetectionShearConfig

    config: MetadetectionShearConfig

    def __init__(self, *, initInputs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(initInputs=initInputs, **kwargs)
        self.metadetect_schema = self.make_metadetect_schema(self.config)
        self.makeSubtask("metadetect")

    @classmethod
    def make_metadetect_schema(cls, config: MetadetectionShearConfig) -> pa.Schema:
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
        pa_schema = pa.schema(
            [
                # Fields from pipeline bookkeeping.
                pa.field(
                    "shearObjectId",
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
                    "patch",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "ID of the patch within the tract on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "cell_x",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Column of the cell within the patch on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "cell_y",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Row of the cell within the patch on which this measurement was made.",
                        "unit": "",
                    },
                ),
                # Fields from metadetection (generic).
                pa.field(
                    "metaStep",
                    pa.string(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Type of artificial shear applied to image. "
                            "One of: 'ns', '1p', '1m', '2p', '2m'."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "stamp_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for the stamp on which this measurement was made.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "x",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Centroid (tract, x-axis) of the detected ShearObject.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "y",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Centroid (tract, y-axis) of the detected ShearObject.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "coord_ra",
                    pa.float64(),
                    nullable=False,
                    metadata={
                        "doc": "Detected Right Ascension of the ShearObject.",
                        "unit": "degrees",
                    },
                ),
                pa.field(
                    "coord_dec",
                    pa.float64(),
                    nullable=False,
                    metadata={
                        "doc": "Detected Declination of the ShearObject.",
                        "unit": "degrees",
                    },
                ),
                # Original PSF measurements
                pa.field(
                    "psfOriginal_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for the original PSF measurement.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfOriginal_e1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Distortion-style e1 of the original PSF from adaptive moments.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfOriginal_e2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Distortion-style e2 of the original PSF from adaptive moments.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfOriginal_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Trace (<x^2> + <y^2>) measurement of the original PSF from adaptive moments.",
                        "unit": "arcseconds squared",
                    },
                ),
                pa.field(
                    "bmask",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "`bmask` flags for the ShearObject",
                        "unit": "",
                    },
                ),
                pa.field(
                    "ormask",
                    pa.int32(),
                    nullable=False,
                    metadata={
                        "doc": "`ored` mask flags for the ShearObject",
                        "unit": "",
                    },
                ),
                pa.field(
                    "mfrac",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Gaussian-weighted masked fraction for the ShearObject.",
                        "unit": "",
                    },
                ),
                # Fields that come only from gauss algorithm.
                # Reconvolved PSF measurements (gauss)
                pa.field(
                    "gauss_psfReconvolved_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for reconvolved PSF (measured with gauss algorithm).",
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_psfReconvolved_g1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Reduced-shear g1 of the reconvolved PSF (measured with gauss algorithm).",
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_psfReconvolved_g2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "Reduced-shear g2 of the reconvolved PSF (measured with gauss algorithm).",
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_psfReconvolved_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Trace (<x^2> + <y^2>) of the reconvolved PSF (measured with gauss algorithm)."
                        ),
                        "unit": "arcseconds squared",
                    },
                ),
                # Object measurements (gauss algorithm).
                pa.field(
                    "gauss_g1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Reduced-shear g1 measurement of the ShearObject "
                            "(measured with gauss algorithm)."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_g2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Reduced-shear g2 measurement of the ShearObject "
                            "(measured with gauss algorithm)."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_g1_g1_Cov",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Auto-covariance of g1 measurement of the ShearObject "
                            "(measured with gauss algorithm)."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_g1_g2_Cov",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Cross-covariance of g1 and g2 measurement of the ShearObject "
                            "(measured with gauss algorithm)."
                        ),
                        "unit": "",
                    },
                ),
                pa.field(
                    "gauss_g2_g2_Cov",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Auto-covariance of g2 measurement of the ShearObject "
                            "(measured with gauss algorithm)."
                        ),
                        "unit": "",
                    },
                ),
            ],
            metadata={
                "shear_step": str(SHEAR_STEP),
                "shear_bands": "".join(sorted(config.metadetect.shear_bands)),
            },
        )

        for alg_name in ("gauss", "pgauss"):
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_snr",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Signal-to-noise ratio measure of the ShearObject "
                            f"(measured with {alg_name} algorithm)."
                        ),
                        "unit": "",
                    },
                ),
            )
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Trace (<x^2> + <y^2>) measurement of the ShearObject "
                            f"(measured with {alg_name} algorithm)."
                        ),
                        "unit": "arcseconds squared",
                    },
                ),
            )
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_TErr",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Uncertainty in the trace measurement of the ShearObject "
                            f"(measured with {alg_name} algorithm)."
                        ),
                        "unit": "arcseconds squared",
                    },
                ),
            )
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_T_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": (
                            "Flags for the trace (<x^2> + <y^2>) measurement of the ShearObject "
                            f"(measured with {alg_name} algorithm)."
                        ),
                        "unit": "",
                    },
                ),
            )
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_object_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": f"Flags for the ShearObject measurement (measured with {alg_name} algorithm).",
                        "unit": "",
                    },
                ),
            )
            pa_schema = pa_schema.append(
                pa.field(
                    f"{alg_name}_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": f"Overall flags for {alg_name} measurement algorithm.",
                        "unit": "",
                    },
                ),
            )

            # Per-band quantities, typically fluxes and associated quantites.
            for b in config.photometry_bands:
                pa_schema = pa_schema.append(
                    pa.field(
                        f"{b}_{alg_name}Flux_flags",
                        pa.uint32(),
                        nullable=False,
                        metadata={
                            "doc": f"Flags set for flux in {b} band measured with {alg_name} algorithm.",
                            "unit": "",
                        },
                    ),
                )
                pa_schema = pa_schema.append(
                    pa.field(
                        f"{b}_{alg_name}Flux",
                        pa.float32(),
                        nullable=b not in config.metadetect.shear_bands,
                        metadata={
                            "doc": f"Flux in {b} band (measured with {alg_name} algorithm).",
                            "unit": "",
                        },
                    ),
                )
                pa_schema = pa_schema.append(
                    pa.field(
                        f"{b}_{alg_name}FluxErr",
                        pa.float32(),
                        nullable=b not in config.metadetect.shear_bands,
                        metadata={
                            "doc": f"Flux uncertainty in {b} band (measured with {alg_name} algorithm).",
                            "unit": "",
                        },
                    ),
                )

        return pa_schema

    def runQuantum(
        self,
        qc: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # Docstring inherited.

        id_generator = self.config.id_generator.apply(qc.quantum.dataId)

        if self.config.do_mask_bright_objects:
            ref_loader = ReferenceObjectLoader(
                dataIds=[ref.datasetRef.dataId for ref in inputRefs.ref_cat],
                refCats=[qc.get(ref) for ref in inputRefs.ref_cat],
                name=self.config.connections.ref_cat,
                config=self.config.ref_loader,
                log=self.log,
            )
            ref_cat = ref_loader.loadRegion(
                qc.quantum.dataId.region, filterName=self.config.ref_loader_filter_name
            )
        else:
            ref_cat = None

        # Read the coadds and put them in the order defined by
        # config.photometry_bands (note that each MultipleCellCoadd object also
        # knows its own band, if that's needed).

        coadds_by_band = {
            ref.dataId["band"]: qc.get(ref)
            for ref in inputRefs.input_coadds
            if ref.dataId["band"] in self.config.photometry_bands
        }

        outputs = self.run(
            patch_coadds=coadds_by_band,
            id_generator=id_generator,
            ref_cat=ref_cat,
        )
        qc.put(outputs, outputRefs)

    def run(
        self,
        *,
        patch_coadds: Mapping[str, MultipleCellCoadd],
        id_generator: FullIdGenerator,
        ref_cat: SimpleCatalog | None,
    ) -> Struct:
        """Run metadetection on a patch.

        Parameters
        ----------
        patch_coadds : `~collections.abc.Mapping` [ \
                `~lsst.cell_coadds.MultipleCellCoadd` ]
            Per-band, per-patch coadds, in the order specified by
            `MetadetectionShearConfig.photometry_bands`.
        id_generator : `~lsst.meas.base.FullIdGenerator`
            Generator for object IDs and to seed the random number generator.
        ref_cat : `lsst.afw.table.SimpleCatalog`, optional
            Reference catalog to use when masking bright stars.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Structure with the following attributes:

            - ``metadetect_catalog`` [ `pyarrow.Table` ]: the output object
              catalog for the patch, with schema equal to `metadetect_schema`.
        """
        seed = id_generator.catalog_id
        self.rng = np.random.RandomState(seed)
        idstart = 0

        grid = patch_coadds[self.config.photometry_bands[0]].grid
        nx_cells, ny_cells = grid.shape
        single_cell_tables: list[pa.Table] = []
        for nx, ny in product(range(nx_cells), range(ny_cells)):
            cell_id = Index2D(nx, ny)
            bbox = grid.bbox_of(cell_id)
            cell_coadds = [patch_coadd.stitch(bbox) for patch_coadd in patch_coadds.values()]
            self.log.debug("Processing cell %s %s", nx, ny)

            try:
                res = self.process_cell(cell_coadds, cell_id=cell_id)
            except Exception as e:
                self.log.error("Failed to process cell %s %s: %s", nx, ny, e)
                continue

            if len(res) > 0:
                res["id"] = id_generator.arange(idstart, idstart + len(res))
                # TODO: Avoid back and forth conversion between array and dict.
                da = self._dictify(
                    res,
                    tract=id_generator.data_id.tract.id,
                    patch=id_generator.data_id.patch.id,
                )
                table = pa.Table.from_pydict(da, self.metadetect_schema)

                single_cell_tables.append(table)
                idstart += len(res)

        if not single_cell_tables:
            raise MetadetectionProcessingError("No objects found in any cell")

        # TODO: DM-53796 De-duplicate objects before concatenation.
        return Struct(
            metadetect_catalog=pa.concat_tables(single_cell_tables),
        )

    def process_cell(
        self,
        cell_coadds: Sequence[StitchedCoadd],
        cell_id: Index2D,
    ) -> pa.Table:
        """Run metadetection on a single cell.

        Parameters
        ----------
        cell_coadds : `~collections.abc.Sequence` [ \
                `~lsst.cell_coadds.StitchedCoadd` ]
            Per-band, per-cell coadds, in the order specified by
            `MetadetectionShearConfig.photometry_bands`.
        cell_id : `~lsst.skymap.Index2D`
            The cell ID for the cell being processed.

        Returns
        -------
        metadetect_catalog : `pyarrow.Table`
            Output object catalog for the cell, with schema equal to
            `metadetect_schema`.
        """

        coadd_data = self._cell_to_coadd_data(cell_coadds)
        # TODO get bright star etc. info as input
        bright_info = []

        apply_apodized_edge_masks_mbexp(**coadd_data)

        if len(bright_info) > 0:
            apply_apodized_bright_masks_mbexp(bright_info=bright_info, **coadd_data)

        mask_frac = _get_mask_frac(
            coadd_data["mfrac_mbexp"],
            trim_pixels=0,
        )

        res = self.metadetect.run(rng=self.rng, **coadd_data)

        comb_res = _make_comb_data(
            cell_coadd=cell_coadds[0],
            res=res,
            mask_frac=mask_frac,
            bands=[cell_coadd.band for cell_coadd in cell_coadds],
            cell_id=cell_id,
        )

        return comb_res

    @staticmethod
    def _cell_to_coadd_data(cell_coadds: Sequence[StitchedCoadd]):
        coadd_data_list = []
        for cell_coadd in cell_coadds:
            coadd_data = {}
            coadd_data["coadd_exp"] = cell_coadd.asExposure()
            coadd_data["coadd_noise_exp"] = cell_coadd.asExposure(noise_index=0)
            coadd_data["coadd_mfrac_exp"] = ExposureF(coadd_data["coadd_exp"], deep=True)
            coadd_data["coadd_mfrac_exp"].image = cell_coadd.mask_fractions
            coadd_data_list.append(coadd_data)

        return extract_multiband_coadd_data(coadd_data_list)

    def _dictify(self, data, tract: int, patch: int):
        output = {}
        # TODO: Move this to a better location after DP2.
        mapping = {
            "bmask": "bmask",
            "cell_x": "cell_x",
            "cell_y": "cell_y",
            "col": "x",
            "col_diff": "x_offset",  # dropped.
            "dec": "coord_dec",
            "gauss_flags": "gauss_flags",
            "gauss_g_1": "gauss_g1",
            "gauss_g_2": "gauss_g2",
            "gauss_g_cov_11": "gauss_g1_g1_Cov",
            "gauss_g_cov_12": "gauss_g1_g2_Cov",  # same as 21.
            "gauss_g_cov_22": "gauss_g2_g2_Cov",
            "gauss_obj_flags": "gauss_object_flags",
            "gauss_psf_flags": "gauss_psfReconvolved_flags",
            "gauss_psf_g_1": "gauss_psfReconvolved_g1",
            "gauss_psf_g_2": "gauss_psfReconvolved_g2",
            "gauss_psf_T": "gauss_psfReconvolved_T",
            "gauss_s2n": "gauss_snr",
            "gauss_T": "gauss_T",
            "gauss_T_err": "gauss_TErr",
            "gauss_T_flags": "gauss_T_flags",
            "gauss_T_ratio": "gauss_T_ratio",  # dropped.
            "id": "shearObjectId",
            "mfrac": "mfrac",
            "ormask": "ormask",
            "pgauss_flags": "pgauss_flags",
            "pgauss_obj_flags": "pgauss_object_flags",
            "pgauss_s2n": "pgauss_snr",
            "pgauss_T": "pgauss_T",
            "pgauss_T_err": "pgauss_TErr",
            "pgauss_T_flags": "pgauss_T_flags",
            "pgauss_T_ratio": "pgauss_T_ratio",  # dropped.
            "psfrec_flags": "psfOriginal_flags",
            "psfrec_g_1": "psfOriginal_e1",
            "psfrec_g_2": "psfOriginal_e2",
            "psfrec_T": "psfOriginal_T",
            "ra": "coord_ra",
            "row": "y",
            "row_diff": "y_offset",  # dropped.
            "shear_type": "metaStep",
            "stamp_flags": "stamp_flags",
        }

        for b, alg_name in product(self.config.photometry_bands, ("gauss", "pgauss")):
            mapping[f"{alg_name}_band_flux_{b}"] = f"{b}_{alg_name}Flux"
            mapping[f"{alg_name}_band_flux_err_{b}"] = f"{b}_{alg_name}FluxErr"
            mapping[f"{alg_name}_band_flux_flags_{b}"] = f"{b}_{alg_name}Flux_flags"

        for name in mapping:
            if name in data.dtype.names:
                output[mapping.get(name, name)] = data[name]
            else:
                if "flags" in name.lower():
                    output[mapping.get(name, name)] = np.ones_like(data["id"], dtype=np.int32)
                else:
                    output[mapping.get(name, name)] = np.ones_like(data["id"], dtype=np.float32)
                    output[mapping.get(name, name)] *= np.nan

        output["tract"] = tract * np.ones_like(data["id"], dtype=np.uint64)
        output["patch"] = patch * np.ones_like(data["id"], dtype=np.uint32)

        return output


def _make_comb_data(
    cell_coadd,
    res,
    mask_frac,
    bands,
    cell_id,
):
    idinfo = cell_coadd.identifiers

    copy_dt = [
        # we will copy out of arrays to these
        ("psfrec_g_1", "f4"),
        ("psfrec_g_2", "f4"),
        ("gauss_psf_g_1", "f4"),
        ("gauss_psf_g_2", "f4"),
        ("gauss_g_1", "f4"),
        ("gauss_g_2", "f4"),
        ("gauss_g_cov_11", "f4"),
        ("gauss_g_cov_12", "f4"),
        ("gauss_g_cov_21", "f4"),
        ("gauss_g_cov_22", "f4"),
    ]

    for b in bands:
        copy_dt.append(("gauss_band_flux_flags_%s" % b, "i4"))
        copy_dt.append(("gauss_band_flux_%s" % b, "f4"))
        copy_dt.append(("gauss_band_flux_err_%s" % b, "f4"))
        copy_dt.append(("pgauss_band_flux_flags_%s" % b, "i4"))
        copy_dt.append(("pgauss_band_flux_%s" % b, "f4"))
        copy_dt.append(("pgauss_band_flux_err_%s" % b, "f4"))

    add_dt = [
        ("id", "u8"),
        ("tract", "u4"),
        ("patch_x", "u1"),
        ("patch_y", "u1"),
        ("cell_x", "u1"),
        ("cell_y", "u1"),
        ("shear_type", "U2"),
        ("mask_frac", "f4"),
        ("primary", bool),
    ] + copy_dt

    if not hasattr(res, "keys"):
        res = {"noshear": res}

    dlist = []
    for stype in res.keys():
        data = res[stype]
        if data is not None:
            newdata = eu.numpy_util.add_fields(data, add_dt)
            newdata["psfrec_g_1"] = newdata["psfrec_g"][:, 0]
            newdata["psfrec_g_2"] = newdata["psfrec_g"][:, 1]

            newdata["gauss_psf_g_1"] = newdata["gauss_psf_g"][:, 0]
            newdata["gauss_psf_g_2"] = newdata["gauss_psf_g"][:, 1]
            newdata["gauss_g_1"] = newdata["gauss_g"][:, 0]
            newdata["gauss_g_2"] = newdata["gauss_g"][:, 1]

            newdata["gauss_g_cov_11"] = newdata["gauss_g_cov"][:, 0, 0]
            newdata["gauss_g_cov_12"] = newdata["gauss_g_cov"][:, 0, 1]
            newdata["gauss_g_cov_21"] = newdata["gauss_g_cov"][:, 1, 0]
            newdata["gauss_g_cov_22"] = newdata["gauss_g_cov"][:, 1, 1]

            # To-do make compatible with a single band better than this.
            if len(bands) > 1:
                for i, b in enumerate(bands):
                    newdata["gauss_band_flux_flags_%s" % b] = newdata["gauss_band_flux_flags"][:, i]
                    newdata["gauss_band_flux_%s" % b] = newdata["gauss_band_flux"][:, i]
                    newdata["gauss_band_flux_err_%s" % b] = newdata["gauss_band_flux_err"][:, i]
                    newdata["pgauss_band_flux_flags_%s" % b] = newdata["pgauss_band_flux_flags"][:, i]
                    newdata["pgauss_band_flux_%s" % b] = newdata["pgauss_band_flux"][:, i]
                    newdata["pgauss_band_flux_err_%s" % b] = newdata["pgauss_band_flux_err"][:, i]
                    newdata["gauss_band_flux_flags_%s" % b] = newdata["gauss_band_flux_flags"][:, i]
                    newdata["gauss_band_flux_%s" % b] = newdata["gauss_band_flux"][:, i]
                    newdata["gauss_band_flux_err_%s" % b] = newdata["gauss_band_flux_err"][:, i]
            else:
                b = bands[0]
                newdata["gauss_band_flux_flags_%s" % b] = newdata["gauss_band_flux_flags"]
                newdata["gauss_band_flux_%s" % b] = newdata["gauss_band_flux"]
                newdata["gauss_band_flux_err_%s" % b] = newdata["gauss_band_flux_err"]
                newdata["pgauss_band_flux_flags_%s" % b] = newdata["pgauss_band_flux_flags"]
                newdata["pgauss_band_flux_%s" % b] = newdata["pgauss_band_flux"]
                newdata["pgauss_band_flux_err_%s" % b] = newdata["pgauss_band_flux_err"]
                newdata["gauss_band_flux_flags_%s" % b] = newdata["gauss_band_flux_flags"]
                newdata["gauss_band_flux_%s" % b] = newdata["gauss_band_flux"]
                newdata["gauss_band_flux_err_%s" % b] = newdata["gauss_band_flux_err"]

            newdata["tract"] = idinfo.tract
            newdata["patch_x"] = idinfo.patch.x
            newdata["patch_y"] = idinfo.patch.y
            newdata["cell_x"] = cell_id.x
            newdata["cell_y"] = cell_id.y

            if stype == "noshear":
                newdata["shear_type"] = "ns"
            else:
                newdata["shear_type"] = stype

            dlist.append(newdata)

    if len(dlist) > 0:
        output = eu.numpy_util.combine_arrlist(dlist)
    else:
        output = []

    return output


def _get_mask_frac(mfrac_mbexp, trim_pixels=0):
    """
    get the average mask frac for each band and then return the max of those
    """

    mask_fracs = []
    for mfrac_exp in mfrac_mbexp:
        mfrac = mfrac_exp.image.array
        dim = mfrac.shape[0]
        mfrac = mfrac[
            trim_pixels : dim - trim_pixels - 1,
            trim_pixels : dim - trim_pixels - 1,
        ]
        mask_fracs.append(mfrac.mean())

    return max(mask_fracs)
