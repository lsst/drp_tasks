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

__all__ = ()

import dataclasses
from collections.abc import Mapping
from typing import ClassVar

import astropy.io.fits
import astropy.table
import numpy as np
import scipy.linalg

from lsst.afw.cameraGeom import FOCAL_PLANE, Camera
from lsst.afw.math import ChebyshevBoundedField, ChebyshevBoundedFieldControl
from lsst.geom import Box2D, Box2I
from lsst.pex.config import Field
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT


@dataclasses.dataclass
class WarpDiffMatrixProblem:
    chebyshev_order: int
    visit_ids: np.ndarray
    matrix: np.ndarray
    vector: np.ndarray

    @classmethod
    def readFits(cls, filename: str) -> WarpDiffMatrixProblem:
        with astropy.io.fits.open(filename) as fits:
            chebyshev_order = fits["MODEL"].header["CHBORDER"]
            visit_ids = np.asarray(fits["MODEL"].data["visit_id"], dtype=np.uint64)
            matrix = np.asarray(fits["MATRIX"].data, dtype=np.float64)
            vector = np.asarray(fits["VECTOR"].data, dtype=np.float64)
        return cls(
            chebyshev_order=chebyshev_order,
            visit_ids=visit_ids,
            matrix=matrix,
            vector=vector,
        )

    def writeFits(self, filename: str) -> None:
        fits = astropy.io.fits.HDUList()
        visit_id_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("visit_id", format="K", array=self.visit_ids)]
        )
        visit_id_hdu.header["EXTNAME"] = "MODEL"
        visit_id_hdu.header["CHBORDER"] = self.chebyshev_order
        fits.append(visit_id_hdu)
        fits.append(astropy.io.fits.ImageHDU(self.matrix, name="MATRIX"))
        fits.append(astropy.io.fits.ImageHDU(self.vector, name="VECTOR"))
        fits.writeto(filename)


class BuildWarpDiffMatrixConnections(PipelineTaskConnections, dimensions=["patch", "band"]):
    diff_table = cT.Input("warp_diff_binned", storageClass="ArrowAstropy", dimensions=["patch", "band"])
    camera = cT.PrerequisiteInput(
        "camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    matrix_problem = cT.Output(
        "warp_diff_matrix_problem", storageClass="WarpDiffMatrixProblem", dimensions=["patch", "band"]
    )


class BuildWarpDiffMatrixConfig(PipelineTaskConfig, pipelineConnections=BuildWarpDiffMatrixConnections):
    covariance_rcond = Field[float](
        "Relative condition number for constructing a basis with nonsingular covariance for each bin.",
        dtype=float,
        default=1e-8,
    )
    chebyshev_order = Field[int](
        "Order of the Chebyshev polynomial fit across the focal-plane for each visit.",
        dtype=int,
        default=4,
        check=lambda v: v >= 0,
    )


class BuildWarpDiffMatrixTask(PipelineTask):
    _DefaultName: ClassVar[str] = "buildWarpDiffMatrix"
    ConfigClass: ClassVar[type[BuildWarpDiffMatrixConfig]] = BuildWarpDiffMatrixConfig
    config: BuildWarpDiffMatrixConfig

    def __init__(self, *, config=None, log=None, initInputs=None, **kwargs):
        super().__init__(config=config, log=log, initInputs=initInputs, **kwargs)
        # We always drop the zeroth-order Chebyshev term since it's degenerate
        # with detector pisons.
        self.n_chebyshev = ((self.config.chebyshev_order + 2) * (self.config.chebyshev_order + 1)) // 2 - 1

    def run(
        self,
        *,
        camera: Camera,
        diff_table: astropy.table.Table,
    ):
        visit_indices = {visit_id: index for index, visit_id in enumerate(self.unique_visits(diff_table))}
        n_visits = len(visit_indices)
        n_parameters = (len(camera) + self.n_chebyshev) * n_visits
        F = np.zeros((n_parameters, n_parameters), dtype=float)
        g = np.zeros(n_parameters, dtype=float)
        diff_table = diff_table.group_by(["bin_row", "bin_column"])
        camera_bbox = self._compute_camera_bbox(camera)
        detector_indices = {detector_id: index for index, detector_id in enumerate(camera)}
        for begin, end in zip(diff_table.groups.indices[:-1], diff_table.groups.indices[1:]):
            bin_diff_table = diff_table[begin:end]
            # The bins that have the same (row, column) in patch coordinates
            # are correlated, because many of them involved the same warp and
            # hence the same pixel values.  We need to incorporate these
            # correlations in our fitting.
            # We start by constructing a 'projection matrix' A that maps the
            # warps to their differences.  It has shape (n_diffs, n_visits),
            # and each row has a 1 for the positive visit and -1 for the
            # negative visit.
            n_diffs = end - begin
            A = np.zeros((n_diffs, n_visits), dtype=float)
            for i in range(n_diffs):
                A[i, visit_indices[bin_diff_table["positive_visit_id"][i]]] = 1.0
                A[i, visit_indices[bin_diff_table["negative_visit_id"][i]]] = -1.0
            # If we had computed the original variances of the visit warps in
            # the same bins V with the same clipping, we could compute the
            # covariance matrix of the diffs via
            #
            #   C = A V A^T
            #
            # We didn't do that, but we did save something almost as good (and
            # a lot more efficient to gather): the variances of the diffs
            # we actually measured, i.e. the diagonal of C.  And thanks to the
            # structure of A, we know those diagonal values are just the sums
            # of the variances of the two visits that went into the diff
            # (because when you subtract values, you add their variances).
            # That means that as long as n_diffs >= n_visits, we can recover
            # V from the diagonal of C, and we can do it by solving:
            #
            #  diag(C) = abs(A) diag(V)
            #
            # Because different diffs involving the same visit might have
            # different clipping, this won't factor exactly, so we use least
            # squares to effectively average over different combinations of
            # diffs giving slightly inconsistent estimates of the visit-level
            # variances.
            V, _, _, _ = scipy.linalg.lstsq(np.abs(A), bin_diff_table["bin_variance"])
            C = np.dot(np.dot(A, V), A.transpose())
            # We can't use our new covariance directly, however, because it's
            # almost certainly singular: we've fundamentally only got at most
            # (n_visits - 1) independent sets of pixels, but we've constructed
            # n_diffs data points from those.  To fix this, we compute the
            # spectral decomposition of C:
            #
            # Q S Q^T
            #
            # and select out the significantly positive eigenvalues and
            # corresponding columns of Q:
            S, Q = scipy.linalg.eigh(C)
            keep = S > np.max(S) * self.config.covariance_rcond
            S = S[keep]
            Q = Q[:, keep]
            # Q still satisfies Q Q^T = I, but not Q^T Q = I.  We can still use
            # it to project our diffs vector 'z' to a smaller-dimensional space
            # with a nonsingular (and diagonal) covariance matrix S:
            #
            #   z -> Q^T z = v
            #
            #   C -> Q^T C Q = Q^T (Q S Q^T) Q = S
            #
            v = np.dot(Q.transpose(), bin_diff_table["bin_value"])
            # Our projected data points are now independent, and they're
            # independent from the data points from any bin (row, column),
            # as well as independent of any the data points from any other
            # patch.
            # The ultimate goal is to find the minimum-norm solution to the the
            # normal equations:
            #
            #  (B^T Q S^{-1} Q^T B) \alpha = B^T Q S^{-1} v
            #
            #  F \alpha = g
            #
            # where \alpha are the background model parameters (for all visits)
            # and B is the matrix that evaluates and subtracts the models at
            # bin centers. Because S is diagonal, we can construct the matrix
            # and right-hand side vector by summing the contributions from each
            # bin 'i':
            #
            #   F = \sum_i F_i = \sum_i B_i^T Q_i S_i^{-1} Q_i^T B_i
            #   b = \sum_i g_i = \sum_i B_i^T Q_i S_i^{-1} v
            #
            S_inv_sqrt = S ** (-0.5)
            B = self.make_model_matrix(diff_table[begin:end], visit_indices, detector_indices, camera_bbox)
            F_i_sqrt = np.dot(S_inv_sqrt[:, np.newaxis] * Q.transpose(), B)
            F += np.dot(F_i_sqrt.transpose(), F_i_sqrt)
            g += np.dot(F_i_sqrt, S_inv_sqrt * v)
        return Struct(
            matrix_problem=WarpDiffMatrixProblem(
                chebyshev_order=self.config.chebyshev_order,
                visit_ids=np.array(visit_indices.keys(), dtype=np.uint64),
                matrix=F,
                vector=g,
            )
        )

    def make_model_matrix(
        self,
        bin_diff_table: astropy.table.Table,
        all_visits: Mapping[int, int],
        all_detectors: Mapping[int, int],
        camera_bbox: Box2I,
    ) -> np.ndarray:
        visits_in_bin = self.unique_visits(bin_diff_table)
        n_visit_parameters = self.n_chebyshev + len(all_detectors)
        n_parameters = n_visit_parameters * len(all_visits)
        ctrl = ChebyshevBoundedFieldControl()
        ctrl.orderX = self.chebyshev_order
        ctrl.orderY = self.chebyshev_order
        B = np.zeros((len(bin_diff_table), n_parameters), dtype=float)
        positive_chebyshev_matrix = ChebyshevBoundedField.makeFitMatrix(
            camera_bbox, bin_diff_table["positive_x"], bin_diff_table["positive_y"], ctrl
        )[:, 1:]
        positive_chebyshev_matrix *= bin_diff_table["positive_scaling"][:, np.newaxis]
        negative_chebyshev_matrix = ChebyshevBoundedField.makeFitMatrix(
            camera_bbox, bin_diff_table["negative_x"], bin_diff_table["negative_y"], ctrl
        )[:, 1:]
        negative_chebyshev_matrix *= bin_diff_table["negative_scaling"][:, np.newaxis]
        for visit_id in visits_in_bin:
            positive_mask = bin_diff_table["positive_visit_id"] == visit_id
            negative_mask = bin_diff_table["negative_visit_id"] == visit_id
            j = all_visits[visit_id] * n_visit_parameters
            B[positive_mask, j : j + self.n_chebyshev] += positive_chebyshev_matrix[positive_mask, :]
            B[negative_mask, j : j + self.n_chebyshev] += negative_chebyshev_matrix[negative_mask, :]
        for i in range(len(bin_diff_table)):
            j = (
                all_visits[bin_diff_table["positive_visit_id"][i]] * n_visit_parameters
                + self.n_chebyshev
                + all_detectors[bin_diff_table["positive_detector_id"][i]]
            )
            B[i, j] = bin_diff_table["positive_scaling"].data[i]
            j = (
                all_visits[bin_diff_table["negative_visit_id"][i]] * n_visit_parameters
                + self.n_chebyshev
                + all_detectors[bin_diff_table["negative_detector_id"][i]]
            )
            B[i, j] = -bin_diff_table["negative_scaling"].data[i]
        return B

    @staticmethod
    def unique_visits(diff_table: astropy.table.Table) -> list[int]:
        return sorted(set(diff_table["positive_visit_id"]) | set(diff_table["negative_visit_id"]))

    @staticmethod
    def unique_detectors(diff_table: astropy.table.Table) -> list[int]:
        return sorted(set(diff_table["positive_detector_id"]) | set(diff_table["negative_detector_id"]))

    @staticmethod
    def _compute_camera_bbox(camera: Camera) -> Box2I:
        """Compute the bounding box of a camera in focal-plane coordinates."""
        bbox = Box2D()
        for detector in camera:
            for corner in detector.getCorners(FOCAL_PLANE):
                bbox.include(corner)
        return Box2I(bbox)
