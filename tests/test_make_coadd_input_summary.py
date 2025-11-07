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

import unittest

import numpy as np

from lsst.drp.tasks.make_coadd_input_summary import (
    MakeCoaddInputSummary,
    MakeCoaddInputSummaryTract,
)
from lsst.pipe.base import InMemoryDatasetHandle
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask


class MakeCoaddInputSummaryTestCase(unittest.TestCase):
    def setUp(self):
        self.recorder_task = CoaddInputRecorderTask(name="recorder")

    def _make_coadd_input_handle(self, tract, patch, band, n_input):
        coadd_inputs = self.recorder_task.makeCoaddInputs()

        coadd_inputs.ccds.resize(n_input)
        coadd_inputs.ccds["id"] = np.arange(n_input)
        coadd_inputs.ccds["ccd"] = np.arange(n_input)
        coadd_inputs.ccds["visit"] = np.arange(100, 100 + n_input)
        coadd_inputs.ccds["goodpix"][:] = 1000
        coadd_inputs.ccds["weight"][:] = 1.0

        handle = InMemoryDatasetHandle(
            coadd_inputs,
            dataId={"tract": tract, "patch": patch, "band": band},
            storageClass="CoaddInputs",
        )

        return handle

    def test_make_coadd_input_summary_tract(self):
        handles = [self._make_coadd_input_handle(100, patch, "g", 10) for patch in [0, 10, 20]]

        task = MakeCoaddInputSummaryTract()
        summary_tract = task.run(coadd_inputs_handles=handles).coadd_input_summary_tract

        self.assertEqual(len(summary_tract), 10 * 3)
        np.testing.assert_array_equal(summary_tract["tract"], 100)
        np.testing.assert_array_equal(summary_tract["band"], "g")

        index = 0
        for handle in handles:
            inputs = handle.get().ccds

            np.testing.assert_array_equal(
                summary_tract["patch"][index : index + len(inputs)],
                handle.dataId["patch"],
            )
            np.testing.assert_array_equal(
                summary_tract["visit"][index : index + len(inputs)],
                inputs["visit"],
            )
            np.testing.assert_array_equal(
                summary_tract["detector"][index : index + len(inputs)],
                inputs["ccd"],
            )
            np.testing.assert_array_equal(
                summary_tract["goodpix"][index : index + len(inputs)],
                inputs["goodpix"],
            )
            np.testing.assert_array_almost_equal(
                summary_tract["weight"][index : index + len(inputs)],
                inputs["weight"],
            )
            index += len(inputs)

        def test_make_coadd_input_summary(self):
            task1 = MakeCoaddInputSummaryTract()

            summary_tract_handles = []

            for tract in [100, 200]:
                handles = [self._make_coadd_input_handle(tract, patch, "g", 10) for patch in [0, 10, 20]]

                summary_tract = task1.run(coadd_inputs_handles=handles).coadd_input_summary_tract

                summary_tract_handles.append(
                    InMemoryDatasetHandle(
                        summary_tract,
                        dataId={"tract": tract, "band": "g"},
                        storageClass="ArrowAstropy",
                    ),
                )

            task2 = MakeCoaddInputSummary()
            summary = task2.run(coadd_input_summary_tract_handles=summary_tract_handles).coadd_input_summary

            self.assertEqual(len(summary), 2 * 10 * 3)

            index = 0
            for handle in summary_tract_handles:
                summary_tract = handle.get()

                for name in ["tract", "patch", "visit", "detector", "goodpix"]:
                    np.testing.assert_array_equal(
                        summary[name][index : index + len(summary_tract)],
                        summary_tract[name],
                    )

                np.testing.assert_array_almost_equal(
                    summary["weight"][index : index + len(summary_tract)],
                    summary_tract["weight"],
                )

                index += len(summary_tract)


if __name__ == "__main__":
    unittest.main()
