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

import numpy as np
import pyarrow as pa
from felis.datamodel import Schema

import lsst.utils.tests
from lsst.drp.tasks.metadetection_shear import MetadetectionShearTask
from lsst.resources import ResourcePath


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


def setup_module(module):
    lsst.utils.tests.init()


class MatchMemoryTestCase(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
