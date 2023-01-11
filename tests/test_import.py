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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.

import importlib.resources
import os.path
import unittest

from lsst.utils import doImport


class ImportTestCase(unittest.TestCase):
    """Test that every file can be imported.

    drp_tasks does not import all the task code by default and not
    every file will always have a test.
    """

    def test_import(self):
        self.assertImport("lsst.drp.tasks")

    def assertImport(self, root_pkg):
        for file in importlib.resources.contents(root_pkg):
            if not file.endswith(".py"):
                continue
            if file.startswith("__"):
                continue
            root, ext = os.path.splitext(file)
            module_name = f"{root_pkg}.{root}"
            with self.subTest(module=module_name):
                try:
                    doImport(module_name)
                except ImportError as e:
                    raise AssertionError(f"Error importing module {module_name}: {e}") from e


if __name__ == "__main__":
    unittest.main()
