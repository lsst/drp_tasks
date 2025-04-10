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

import astropy.table
import numpy as np

import lsst.utils.tests
from lsst.drp.tasks.compute_object_epochs import ComputeObjectEpochsConfig, ComputeObjectEpochsTask


class ComputeObjectEpochsTestCase(lsst.utils.tests.TestCase):
    def test_computeObjectEpochs(self):
        """Test basic MakePsfMatchedWarpTask

        This constructs a direct_warp using `MakeDirectWarpTask` and then
        runs `MakePsfMatchedWarpTask` on it.
        """
        cat = astropy.table.Table(
            {
                "id": np.arange(5),
                "ra": np.full(5, np.nan),
                "dec": np.full(5, np.nan),
            }
        )

        bands = ("g", "r")

        task = ComputeObjectEpochsTask(config=ComputeObjectEpochsConfig(bands=bands))
        # TODO: DM-46202 Pass at least one healSparseMap if this task remains.
        result = task.computeEpochs(cat, {})

        self.assertListEqual(result.colnames, ["objectId"] + [f"{band}_epoch" for band in bands])


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
