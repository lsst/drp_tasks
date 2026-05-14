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


import unittest
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
from felis.datamodel import Schema

import lsst.utils.tests
from lsst.drp.tasks.metadetection_shear import MetadetectionShearConfig, MetadetectionShearTask
from lsst.pipe.base import InvalidQuantumError
from lsst.resources import ResourcePath
from lsst.skymap import RingsSkyMapConfig


class ShearObjectSchemaConsistencyTestCase(unittest.TestCase):
    """Verify that the ShearObject table in lsstcam.yaml is a subset of
    the schema produced by MetadetectionShearTask.make_metadetect_schema.
    """

    def assertFieldsEqual(self, field1, field2):
        """Assert that the two fields are identical.

        Specifically, it checks that the following are identical:
        - name
        - type width (if not string)
        """
        self.assertEqual(field1.name, field2.name)
        if field1.type != "string":
            self.assertEqual(field1.type.byte_width, field2.type.byte_width)

    def _read_shearobject_columns_from_yaml(self) -> pa.Schema:
        dtypes = {
            "char": "U2",
            "int": np.int32,
            "uint": np.uint32,
            "float": np.float32,
            "double": np.float64,
        }
        # Load the lsstcam.yaml schema resource from sdm_schemas

        resource = ResourcePath("resource://lsst.sdm.schemas/lsstcam.yaml")
        schema = Schema.from_uri(resource, context={"id_generation": True})
        # Find ShearObject table and collect column names
        for table in schema.tables:
            if table.name == "ShearObject":
                cols = pa.schema(
                    [
                        pa.field(
                            col.name,
                            pa.from_numpy_dtype(dtypes.get(col.datatype, col.datatype)),
                        )
                        for col in table.columns
                    ]
                )
                return cols

    def test_yaml_is_subset_of_generated_schema(self) -> None:
        sdm_schema = self._read_shearobject_columns_from_yaml()
        # Generate schema from task
        config = MetadetectionShearTask.ConfigClass()
        # Ensure defaults are applied (e.g., shear_bands)
        config.setDefaults()
        task_schema = MetadetectionShearTask.make_metadetect_schema(config)

        missing_field_names = set(sdm_schema.names).difference(task_schema.names)
        self.assertFalse(missing_field_names, f"Missing fields in generated schema: {missing_field_names}")

        for field in sdm_schema:
            with self.subTest(field_name=field.name):
                self.assertFieldsEqual(field, task_schema.field(field.name))


@dataclass(frozen=True)
class SkymapConfigs:
    """Immutable configuration container for standard skymap settings."""

    lsst_cells_v1: RingsSkyMapConfig
    lsst_cells_v2: RingsSkyMapConfig


class CountCellsAlongEdgesTestCase(unittest.TestCase):
    """Test the count_cells_along_edges static method."""

    @classmethod
    def setUpClass(cls) -> None:
        # Docstring inherited.

        ring_skymap_config = RingsSkyMapConfig()
        ring_skymap_config.tractBuilder.name = "cells"
        ring_skymap_config.numRings = 120
        ring_skymap_config.pixelScale = 0.2
        ring_skymap_config.projection = "TAN"
        ring_skymap_config.raStart = 0.0
        ring_skymap_config.rotation = 0.0
        ring_skymap_config.tractOverlap = 0.016666666666666666

        cell_config = ring_skymap_config.tractBuilder["cells"]

        cell_config.cellInnerDimensions = [150, 150]
        cell_config.numCellsInPatchBorder = 1
        cell_config.numCellsPerPatchInner = 20

        cls.standard_skymap_configs = SkymapConfigs(
            lsst_cells_v1=ring_skymap_config.copy(),
            lsst_cells_v2=ring_skymap_config.copy(),
        )
        cls.standard_skymap_configs.lsst_cells_v1.tractBuilder["cells"].cellBorder = 50
        cls.standard_skymap_configs.lsst_cells_v2.tractBuilder["cells"].cellBorder = 0
        cls.standard_skymap_configs.lsst_cells_v1.freeze()
        cls.standard_skymap_configs.lsst_cells_v2.freeze()

        cls.custom_skymap_config = ring_skymap_config

    def setUp(self):
        # Docstring inherited.
        self.custom_skymap_config = self.standard_skymap_configs.lsst_cells_v1.copy()
        self.metadetection_shear_config = MetadetectionShearConfig()

    def test_count_cells_along_edges(self):
        """Test count_cells_along_edges"""

        lsst_cells_v1 = self.standard_skymap_configs.lsst_cells_v1
        num_cells = MetadetectionShearTask.count_cells_along_edges(lsst_cells_v1)
        self.assertEqual(num_cells, 0)

        lsst_cells_v2 = self.standard_skymap_configs.lsst_cells_v2
        num_cells = MetadetectionShearTask.count_cells_along_edges(lsst_cells_v2)
        self.assertEqual(num_cells, 1)

    def test_validate_skymap_config_lsst_cells_v1(self):
        """Test validation passes with lsst_cells_v1 skymap."""
        self.metadetection_shear_config.border = None
        task = MetadetectionShearTask(config=self.metadetection_shear_config)

        # This should not raise an exception
        task.validate_skymap_config(self.standard_skymap_configs.lsst_cells_v1)

    def test_validate_skymap_config_lsst_cells_v2(self):
        """Test validation passes with lsst_cells_v2 skymap."""
        self.metadetection_shear_config.border = 50
        task = MetadetectionShearTask(config=self.metadetection_shear_config)

        # This should not raise an exception
        task.validate_skymap_config(self.standard_skymap_configs.lsst_cells_v2)

    def test_validate_skymap_config_invalid_tract_builder(self):
        """Test validation fails with non-cells tract builder."""
        self.custom_skymap_config.tractBuilder.name = "legacy"

        task = MetadetectionShearTask(config=self.metadetection_shear_config)

        with self.assertRaises(InvalidQuantumError, msg="requires a cell-based skymap"):
            task.validate_skymap_config(self.custom_skymap_config)

    def test_validate_skymap_config_no_border_in_both_configs(self):
        """Test validation fails when border is None in task config and
        cellBorder is 0 in skymap.
        """
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 0

        config = MetadetectionShearConfig()
        config.border = None

        task = MetadetectionShearTask(config=config)

        with self.assertRaises(
            InvalidQuantumError, msg="requires a border to be set either in the skymap config"
        ):
            task.validate_skymap_config(self.custom_skymap_config)

    def test_validate_skymap_config_border_in_both_configs(self):
        """Test validation fails when border is set in both task config and
        cellBorder in skymap.
        """
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 2

        config = MetadetectionShearConfig()
        config.border = 10

        task = MetadetectionShearTask(config=config)

        with self.assertRaises(InvalidQuantumError) as cm:
            task.validate_skymap_config(self.custom_skymap_config)

        self.assertIn("requires a border to be set either in the skymap config", str(cm.exception))

    def test_validate_skymap_config_border_too_large(self):
        """Test validation fails when border value exceeds maximum allowed."""
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 0  # No cell border

        config = MetadetectionShearConfig()
        config.border = 5000  # Very large border

        task = MetadetectionShearTask(config=config)

        with self.assertRaises(InvalidQuantumError) as cm:
            task.validate_skymap_config(self.custom_skymap_config)

        self.assertIn("border value is too large", str(cm.exception))

    def test_validate_skymap_config_pixel_scale_mismatch(self):
        """Test validation fails when pixel scale calculations don't match."""
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 0
        self.custom_skymap_config.pixelScale = 0.1  # Very small pixel scale
        self.custom_skymap_config.tractOverlap = 0.001  # Small tract overlap

        config = MetadetectionShearConfig()
        config.border = 10

        task = MetadetectionShearTask(config=config)

        with self.assertRaises(InvalidQuantumError) as cm:
            task.validate_skymap_config(self.custom_skymap_config)

        self.assertIn("pixel scale in the skymap config does not match", str(cm.exception))

    def test_validate_skymap_config_success_with_task_border(self):
        """Test validation succeeds when border is set in task config only."""
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 0  # No cell border

        config = MetadetectionShearConfig()
        config.border = 10  # Reasonable border

        task = MetadetectionShearTask(config=config)

        # Should not raise an exception
        task.validate_skymap_config(self.custom_skymap_config)

    def test_validate_skymap_config_success_with_skymap_border(self):
        """Test validation succeeds when border is set in skymap config
        only.
        """
        self.custom_skymap_config.tractBuilder["cells"].cellBorder = 2  # Has cell border
        self.custom_skymap_config.tractBuilder["cells"].numCellsInPatchBorder = 1

        config = MetadetectionShearConfig()
        config.border = None  # No task border

        task = MetadetectionShearTask(config=config)

        # Should not raise an exception
        task.validate_skymap_config(self.custom_skymap_config)


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
