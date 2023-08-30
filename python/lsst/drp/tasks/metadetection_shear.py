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
import numpy as np

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
        "cellCoadd",
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
    # TODO make this exist
    # object_schema = cT.InitOutput(
    #     "ShearObject_schema",
    #     # TODO: It's not currently possible to save ArrowSchema objects on
    #     # their own, but some combination of Eli and Jim can figure out
    #     # how to # fix that.
    #     storageClass="ArrowSchema",
    #     doc="Schema of the output catalog.",
    # )

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

    from lsst.meas.base import SkyMapIdGeneratorConfig

    required_bands = ListField[str](
        "Bands expected to be present.  Cells with one or more of these bands "
        "missing will be skipped.  Bands other than those listed here will "
        "not be processed.",
        # TODO learn how to set in a config file
        # default=["g", "r", "i", "z"],
        default=["r"],
        optional=False,
    )

    idGenerator = SkyMapIdGeneratorConfig.make_field()

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

                pa.field(
                    "wmom_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Overall flags for wmom measurement.",
                        "unit": "",
                    },
                ),

                # Original PSF measurements
                pa.field(
                    "psfrec_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for admom PSF measurement.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfrec_g_1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "admom g1 measurement for PSF.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfrec_g_2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "admom g2 measurement for PSF.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "psfrec_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "admom wmom T (<x^2> + <y^2>) measurement for PSF.",
                        "unit": "",
                    },
                ),

                # reconvolved PSF measurements
                pa.field(
                    "wmom_psf_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for wmom reconvolved PSF measurement.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_psf_g_1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom g1 measurement for reconvolved PSF.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_psf_g_2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom g2 measurement for reconvolved PSF.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_psf_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom T (<x^2> + <y^2>) measurement for reconvolved PSF.",
                        "unit": "",
                    },
                ),

                # Object measurements
                pa.field(
                    "wmom_obj_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for wmom object measurement.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_s2n",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom object s2n measurement.",
                        "unit": "",
                    },
                ),

                pa.field(
                    "wmom_g_1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom object g1 measurement.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_g_2",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom object g2 measurement.",
                        "unit": "",
                    },
                ),

                pa.field(
                    "wmom_T_flags",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "Flags for wmom T measurement for object.",
                        "unit": "",
                    },
                ),

                pa.field(
                    "wmom_T",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom T (<x^2> + <y^2>) measurement for object.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_T_err",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom T uncertainty for object.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_T_ratio",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom T/Tpsf for object.",
                        "unit": "",
                    },
                ),

                pa.field(
                    "wmom_band_flux_flags_1",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom flux measurement flags for object.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_band_flux_1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom flux for object.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "wmom_band_flux_err_1",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "wmom flux uncertainty for object.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "shear_bands",
                    pa.string(),
                    nullable=False,
                    metadata={
                        "doc": "bands used for shear measurement.",
                        "unit": "",
                    },
                ),

                # pa.field(
                #     "row0",
                #     pa.float32(),
                #     nullable=False,
                #     metadata={
                #         "doc": "row start for stamp.",
                #         "unit": "",
                #     },
                # ),
                # pa.field(
                #     "col0",
                #     pa.float32(),
                #     nullable=False,
                #     metadata={
                #         "doc": "column start for stamp.",
                #         "unit": "",
                #     },
                # ),
                pa.field(
                    "row",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "detected row for object, within image.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "col",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "detected column for object, within image.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "row_diff",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "difference of measured row from detected row.",
                        "unit": "",
                    },
                ),
                pa.field(
                    "col_diff",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "difference of measured column from detected row.",
                        "unit": "",
                    },
                ),

                pa.field(
                    "ra",
                    pa.float64(),
                    nullable=False,
                    metadata={
                        "doc": "detected ra for object.",
                        "unit": "degrees",
                    },
                ),
                pa.field(
                    "dec",
                    pa.float64(),
                    nullable=False,
                    metadata={
                        "doc": "detected dec for object.",
                        "unit": "degrees",
                    },
                ),

                pa.field(
                    "bmask",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "bmask flags for object",
                        "unit": "",
                    },
                ),
                pa.field(
                    "ormask",
                    pa.uint32(),
                    nullable=False,
                    metadata={
                        "doc": "ored mask flags for object",
                        "unit": "",
                    },
                ),

                pa.field(
                    "mfrac",
                    pa.float32(),
                    nullable=False,
                    metadata={
                        "doc": "gaussian weighted masked fraction for object.",
                        "unit": "",
                    },
                ),

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

        # TODO make this work
        idGenerator = self.config.idGenerator.apply(qc.quantum.dataId)
        seed = idGenerator.catalog_id

        coadds_by_band = {
            ref.dataId["band"]: qc.get(ref) for ref in inputRefs.input_coadds
        }
        outputs = self.run(
            [coadds_by_band[b] for b in self.config.required_bands], seed,
        )
        qc.put(outputs, outputRefs)

    def run(self, patch_coadds: Sequence[MultipleCellCoadd], seed: int) -> Struct:
        """Run metadetection on a patch.

        Parameters
        ----------
        patch_coadds : `~collections.abc.Sequence` [ \
                `~lsst.cell_coadds.MultipleCellCoadd` ]
            Per-band, per-patch coadds, in the order specified by
            `MetadetectionShearConfig.required_bands`.
        seed: int
            A seed for random number generator, used for simulation, and/or
            fitting.

        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Structure with the following attributes:

            - ``object_catalog`` [ `pyarrow.Table` ]: the output object
              catalog for the patch, with schema equal to `object_schema`.
        """

        # TODO figure out how to use a config to switch to simulation mode
        # in that case also need to fake the id info
        rng = np.random.RandomState(seed)
        idstart = 0

        single_cell_tables: list[pa.Table] = []
        for single_cell_coadds in zip(
            *[patch_coadd.cells.values() for patch_coadd in patch_coadds], strict=True
        ):
            self._log_ids(single_cell_coadds[0])

            res = self.process_cell(single_cell_coadds, rng, simulate=True)

            # TODO figure out how to make the object ids
            if len(res) > 0:
                res['id'] = np.arange(idstart, idstart + len(res))
                da = _dictify(res)
                table = pa.Table.from_pydict(da, self.object_schema)

                single_cell_tables.append(table)
                idstart += len(res)
            break

        # TODO: if we need to do any cell-overlap-region deduplication here
        # (instead of purely in science analysis code), this is where'd it'd
        # happen.
        return Struct(
            object_catalog=pa.concat_tables(single_cell_tables),
        )

    def process_cell(
        self, single_cell_coadds: Sequence[SingleCellCoadd], rng,
        simulate=False,
    ) -> pa.Table:
        """Run metadetection on a single cell.

        Parameters
        ----------
        single_cell_coadds : `~collections.abc.Sequence` [ \
                `~lsst.cell_coadds.SingleCellCoadd` ]
            Per-band, per-cell coadds, in the order specified by
            `MetadetectionShearConfig.required_bands`.

        rng: `np.random.RandomState`
            Random number generator.
        simulate: bool, optional
            If set to True, simulate the data

        Returns
        -------
        object_catalog : `pyarrow.Table`
            Output object catalog for the cell, with schema equal to
            `object_schema`.
        """
        from metadetect.lsst.metadetect import run_metadetect
        from metadetect.lsst.metadetect import get_config as get_mdet_config

        # rows: list[dict[str, Any]] = []
        # TODO: run metadetection on the cell, filling in 'rows' with
        # measurements.  Or replace 'rows' with a 'columns' dict of numpy array
        # columns and call 'from_pydict' instead of 'from_pylist' below, if
        # that's more convenient.

        coadd_data = _simulate_coadd(rng)

        mask_frac = _get_mask_frac(
            coadd_data['mfrac_mbexp'],
            trim_pixels=0,
        )

        mdet_config = get_mdet_config()
        # TODO how to get the id/tract/patch_x etc. currently in schema
        res = run_metadetect(
            rng=rng,
            config=mdet_config,
            **coadd_data,
        )
        comb_res = _make_comb_data(
            single_cell_coadd=single_cell_coadds[0],
            res=res,
            meas_type=mdet_config['meas_type'],
            mask_frac=mask_frac,
            full_output=True,
        )

        return comb_res

    def _log_ids(self, single_cell_coadd):
        idinfo = single_cell_coadd.identifiers
        tract = idinfo.tract
        patch_x = idinfo.patch.x
        patch_y = idinfo.patch.y
        cell_x = idinfo.cell.x
        cell_y = idinfo.cell.y
        mess = (
            f'tract: {tract} patch: {patch_x},{patch_y} cell: {cell_x},{cell_y}'
        )
        self.log.info(mess)


def _dictify(data):
    output = {}
    for name in data.dtype.names:
        output[name] = data[name]
    return output


def _make_comb_data(
    single_cell_coadd,
    res,
    meas_type,
    mask_frac,
    full_output=False,
):
    import esutil as eu

    idinfo = single_cell_coadd.identifiers

    copy_dt = [
        # we will copy out of arrays to these
        ('psfrec_g_1', 'f4'),
        ('psfrec_g_2', 'f4'),
        ('wmom_psf_g_1', 'f4'),
        ('wmom_psf_g_2', 'f4'),
        ('wmom_g_1', 'f4'),
        ('wmom_g_2', 'f4'),
        ('wmom_band_flux_flags_1', 'i4'),
        ('wmom_band_flux_1', 'f4'),
        ('wmom_band_flux_err_1', 'f4'),
    ]

    add_dt = [
        ('id', 'u8'),
        ('tract', 'u4'),
        ('patch_x', 'u1'),
        ('patch_y', 'u1'),
        ('cell_x', 'u1'),
        ('cell_y', 'u1'),
        ('shear_type', 'U2'),
        ('mask_frac', 'f4'),
        ('primary', bool),
    ] + copy_dt

    if not hasattr(res, 'keys'):
        res = {'noshear': res}

    dlist = []
    for stype in res.keys():
        data = res[stype]
        if data is not None:

            if not full_output:
                data = _trim_output_columns(data, meas_type)

            newdata = eu.numpy_util.add_fields(data, add_dt)
            newdata['psfrec_g_1'] = newdata['psfrec_g'][:, 0]
            newdata['psfrec_g_2'] = newdata['psfrec_g'][:, 1]
            newdata['wmom_psf_g_1'] = newdata['wmom_psf_g'][:, 0]
            newdata['wmom_psf_g_2'] = newdata['wmom_psf_g'][:, 1]
            newdata['wmom_g_1'] = newdata['wmom_g'][:, 0]
            newdata['wmom_g_2'] = newdata['wmom_g'][:, 1]
            newdata['wmom_band_flux_flags_1'] = newdata['wmom_band_flux_flags']
            newdata['wmom_band_flux_1'] = newdata['wmom_band_flux']
            newdata['wmom_band_flux_err_1'] = newdata['wmom_band_flux_err']

            newdata['tract'] = idinfo.tract
            newdata['patch_x'] = idinfo.patch.x
            newdata['patch_y'] = idinfo.patch.y
            newdata['cell_x'] = idinfo.cell.x
            newdata['cell_y'] = idinfo.cell.y

            if stype == 'noshear':
                newdata['shear_type'] = 'ns'
            else:
                newdata['shear_type'] = stype

            dlist.append(newdata)

    if len(dlist) > 0:
        output = eu.numpy_util.combine_arrlist(dlist)
        # output['id'] = np.arange(output.size)
    else:
        output = []

    return output


def _trim_output_columns(data, meas_type):
    # TODO decide what to keep
    raise NotImplementedError('implement trim output')

    # if meas_type == 'admom':
    #     meas_type = 'am'
    #
    # # note the bmask/ormask compress to nothing
    # cols2keep_orig = [
    #     get_flags_name(data=data, meas_type=meas_type),
    #     'bmask',
    #     'ormask',
    #     'row', 'row0',
    #     'col', 'col0',
    #     'mfrac',
    #     '%s_s2n' % meas_type,
    #     '%s_T_ratio' % meas_type,
    #     '%s_g' % meas_type,
    #     '%s_g_cov' % meas_type,
    # ]
    #
    # cols2keep = []
    # for col in cols2keep_orig:
    #     if col in data.dtype.names:
    #         cols2keep.append(col)
    #
    # return eu.numpy_util.extract_fields(data, cols2keep)


def _get_mask_frac(mfrac_mbexp, trim_pixels=0):
    """
    get the average mask frac for each band and then return the max of those
    """

    mask_fracs = []
    for mfrac_exp in mfrac_mbexp:
        mfrac = mfrac_exp.image.array
        dim = mfrac.shape[0]
        mfrac = mfrac[
            trim_pixels:dim - trim_pixels - 1,
            trim_pixels:dim - trim_pixels - 1,
        ]
        mask_fracs.append(mfrac.mean())

    return max(mask_fracs)


def _simulate_coadd(rng):
    from descwl_shear_sims.sim import (
        make_sim,
        get_sim_config,
        get_se_dim,
    )
    from descwl_shear_sims.galaxies import make_galaxy_catalog
    from descwl_shear_sims.psfs import make_fixed_psf, make_ps_psf, make_rand_psf
    # from descwl_shear_sims.stars import make_star_catalog
    from descwl_coadd.coadd_nowarp import make_coadd_nowarp
    from metadetect.lsst.util import extract_multiband_coadd_data

    g1, g2 = 0.02, 0.00

    sim_config = get_sim_config()
    if sim_config['se_dim'] is None:
        sim_config['se_dim'] = get_se_dim(
            coadd_dim=sim_config['coadd_dim'],
            dither=sim_config['dither'],
            rotate=sim_config['rotate'],
        )

    if sim_config['gal_type'] != 'wldeblend':
        gal_config = sim_config.get('gal_config', None)
    else:
        sim_config["layout"] = None
        gal_config = None

    galaxy_catalog = make_galaxy_catalog(
        rng=rng,
        gal_type=sim_config['gal_type'],
        coadd_dim=sim_config['coadd_dim'],
        buff=sim_config['buff'],
        layout=sim_config['layout'],
        sep=sim_config['sep'],  # for layout='pair'
        gal_config=gal_config,
    )
    if sim_config['psf_type'] == 'ps':
        psf = make_ps_psf(
            rng=rng,
            dim=sim_config['se_dim'],
            variation_factor=sim_config['psf_variation_factor'],
        )
    elif sim_config['randomize_psf']:
        psf = make_rand_psf(
            psf_type=sim_config["psf_type"], rng=rng,
        )
    else:
        psf = make_fixed_psf(psf_type=sim_config["psf_type"])

    sim_data = make_sim(
        rng=rng,
        galaxy_catalog=galaxy_catalog,
        coadd_dim=sim_config['coadd_dim'],
        se_dim=sim_config['se_dim'],
        g1=g1,
        g2=g2,
        psf=psf,
        draw_stars=sim_config['draw_stars'],
        psf_dim=sim_config['psf_dim'],
        dither=sim_config['dither'],
        rotate=sim_config['rotate'],
        bands=sim_config['bands'],
        epochs_per_band=sim_config['epochs_per_band'],
        noise_factor=sim_config['noise_factor'],
        cosmic_rays=sim_config['cosmic_rays'],
        bad_columns=sim_config['bad_columns'],
        star_bleeds=sim_config['star_bleeds'],
        sky_n_sigma=sim_config['sky_n_sigma'],
    )

    bands = list(sim_data['band_data'].keys())
    exps = sim_data['band_data'][bands[0]]
    assert (
        not sim_config['dither']
        and not sim_config['rotate']
        and len(exps) == 1
    )

    coadd_data_list = [
        make_coadd_nowarp(
            exp=exps[0],
            psf_dims=sim_data['psf_dims'],
            rng=rng,
            remove_poisson=False,
        )
        for band in bands
    ]

    return extract_multiband_coadd_data(coadd_data_list)
