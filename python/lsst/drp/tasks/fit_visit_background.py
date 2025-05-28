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
import uuid
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import ClassVar, cast

import numpy as np
import pydantic

from lsst.afw.cameraGeom import FOCAL_PLANE, PIXELS, Camera, Detector
from lsst.afw.image import MaskedImageF
from lsst.afw.math import (
    ApproximateControl,
    BackgroundList,
    BackgroundMI,
    ChebyshevBoundedField,
    ChebyshevBoundedFieldControl,
    Interpolate,
    UndersampleStyle,
)
from lsst.geom import Box2D, Box2I, Point2I
from lsst.pex.config import Field
from lsst.pipe.base import (
    InputQuantizedConnection,
    InvalidQuantumError,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    QuantumContext,
    Struct,
)
from lsst.pipe.base import connectionTypes as cT


@dataclasses.dataclass(frozen=True)
class FitInputData:
    """Data points input to a focal-plane background fit.

    All arrays are 1-d and have same length, and are considered read-only.
    """

    x: np.ndarray
    """Focal plane X coordinates of the centers of bins."""

    y: np.ndarray
    """Focal plane Y coordinates of the centers of bins."""

    z: np.ndarray
    """Values of background-estimation bins."""

    w: np.ndarray
    """Weights of background-estimation bins, from variance**(-1/2)."""

    sky_frame: np.ndarray | None = None
    """Binned sky frame values.

    This will be `None` if the sky frame is not included as a basis function.
    """

    detector_slices: dict[int, slice] = dataclasses.field(default_factory=dict)
    """A mapping from detector ID to a slice that corresponds to the points
    in that detector in the arrays.

    This may be empty for single-detector objects.
    """

    def __post_init__(self) -> None:
        # Make arrays immutable to avoid bugs, especially with caching.
        self.x.flags.writeable = False
        self.y.flags.writeable = False
        self.z.flags.writeable = False
        self.w.flags.writeable = False
        if self.sky_frame is not None:
            self.sky_frame.flags.writeable = False

    def __len__(self) -> int:
        return len(self.z)

    def masked(self, mask: np.ndarray) -> FitInputData:
        """Returned a masked copy of the data.

        Parameters
        ----------
        mask : `numpy.ndarray`
            Boolean or index array that selects the data to keep.

        Returns
        -------
        masked : `FitInputData`
            Data with mask applied.  Detector slices are dropped.
        """
        return FitInputData(
            x=self.x[mask],
            y=self.y[mask],
            z=self.z[mask],
            w=self.w[mask],
            sky_frame=self.sky_frame[mask] if self.sky_frame is not None else None,
        )

    def fit(self, matrix: np.ndarray) -> VisitBackgroundFit:
        """Perform a linear least-squares fit with this data.

        Parameters
        ----------
        matrix : `np.ndarray`
            Design matrix to solve.  First-dimension shape should equal
            ``len(self)``.

        Returns
        -------
        fit : `VisitBackgroundFit`
            Object holding the best-fit coefficients that also remembers the
            matrix and the input data.
        """
        if self.sky_frame is not None:
            full_matrix = np.zeros((matrix.shape[0], matrix.shape[1] + 1), dtype=float)
            full_matrix[:, :-1] = matrix
            full_matrix[:, -1] = self.sky_frame
        else:
            full_matrix = matrix
        coefficients, _, _, _ = np.linalg.lstsq(full_matrix * self.w[:, np.newaxis], self.z * self.w)
        sky_frame_factor: float | None = None
        if self.sky_frame is not None:
            sky_frame_factor = coefficients[-1]
            coefficients = coefficients[:-1]
        return VisitBackgroundFit(
            coefficients=coefficients, matrix=matrix, sky_frame_factor=sky_frame_factor, data=self
        )

    @classmethod
    def concatenate(cls, data: Mapping[int, FitInputData]) -> FitInputData:
        """Combine input data from multiple detectors.

        Parameters
        ----------
        data : `~collections.abc.Mapping` [ `int`, `FitInputData` ]
            Mapping from detector ID to the input data from that detector.

        Returns
        -------
        concatenated : `FitInputData`
            Concatenated input data.
        """
        i = 0
        has_sky: bool = False
        detector_slices: dict[int, slice] = {}
        for detector_id, detector_data in data.items():
            detector_slices[detector_id] = slice(i, i + len(detector_data))
            i += len(detector_data)
            has_sky = detector_data.sky_frame is not None
        return cls(
            x=np.concatenate([d.x for d in data.values()]),
            y=np.concatenate([d.y for d in data.values()]),
            z=np.concatenate([d.z for d in data.values()]),
            w=np.concatenate([d.w for d in data.values()]),
            sky_frame=(
                np.concatenate([cast(np.ndarray, d.sky_frame) for d in data.values()]) if has_sky else None
            ),
            detector_slices=detector_slices,
        )


@dataclasses.dataclass(frozen=True)
class VisitBackgroundFit:
    """Results of a linear fit to a focal-plane background."""

    coefficients: np.ndarray
    """Coefficients of the fit (i.e. model parameters)."""

    matrix: np.ndarray
    """Matrix that evaluates the model at the data points.

    Shape is ``(len(data), len(self.coefficients))``.
    """

    sky_frame_factor: float | None
    """Coefficient for the sky frame basis function."""

    data: FitInputData
    """Data that went into the fit."""

    def __post_init__(self) -> None:
        # Make arrays immutable to avoid bugs, especially with caching.
        self.coefficients.flags.writeable = False
        self.matrix.flags.writeable = False

    def masked(self, mask: np.ndarray) -> VisitBackgroundFit:
        """Apply a mask to the data and matrix, and re-fit the coefficients
        to return a new instance.

        Parameters
        ----------
        mask : `numpy.ndarray`
            Boolean or index array that selects the data to keep.

        Returns
        -------
        masked_fit : `VisitBackgroundFit`
            New fit with mask applied to data.
        """
        data = self.data.masked(mask)
        matrix = self.matrix[mask, :]
        return data.fit(matrix)

    @cached_property
    def model(self) -> np.ndarray:
        """The model evaluated at the data points."""
        result = np.dot(self.matrix, self.coefficients)
        if self.sky_frame_factor is not None and self.data.sky_frame is not None:
            result += self.sky_frame_factor * self.data.sky_frame
        return result

    @cached_property
    def residuals(self) -> np.ndarray:
        """The difference between the data and the model."""
        return self.data.z - self.model


class ChebyshevBasis(pydantic.BaseModel):
    """A persistable description of a 2-d Chebyshev polynonmial basis."""

    x_min: pydantic.StrictInt
    x_max: pydantic.StrictInt
    y_min: pydantic.StrictInt
    y_max: pydantic.StrictInt
    x_order: pydantic.StrictInt
    y_order: pydantic.StrictInt

    @classmethod
    def from_bbox(cls, bbox: Box2I, x_order: int, y_order: int) -> ChebyshevBasis:
        """Construct from a box object and polynomial orders.

        Parameters
        ----------
        bbox : `lsst.geom.Box2I`
            Bounding box used to scale the range of the Chebyshev polynomials
            from (-1, 1).
        x_order : `int`
            Polynomial order (inclusive) in the X direction.
        y_order : `int`
            Polynomial order (inclusive) in the Y direction.
        """
        return cls(
            x_min=bbox.x.min,
            x_max=bbox.x.max,
            y_min=bbox.y.min,
            y_max=bbox.y.max,
            x_order=x_order,
            y_order=y_order,
        )

    @property
    def bbox(self) -> Box2I:
        """Bounding box used to scale the range of the Chebyshev polynomials
        from (-1, 1).
        """
        return Box2I(minimum=Point2I(self.x_min, self.y_min), maximum=Point2I(self.x_max, self.y_max))


class VisitBackgroundModel(pydantic.BaseModel):
    """A persistable background model for a full visit."""

    version: int = 1
    """File format version number."""

    chebyshev_basis: ChebyshevBasis
    """Chebyshev basis in focal-plane coordinates."""

    detector_ids: list[pydantic.StrictInt] | None = None
    """IDs of detectors that have pistons.

    When this is not `None`, `detector_types` must be `None` and the
    zeroth-order Chebyshev term is dropped.
    """

    detector_types: list[pydantic.StrictStr] | None = None
    """Detector physical types that have pistons.

    When this is not `None`, `detector_ids` must be `None` and the zeroth-order
    Chebyshev term is dropped.
    """

    sky_frame_datasets: dict[int, uuid.UUID] | None = None
    """UUIDs of the binned sky frame datasets that were used as a basis
    function in this fit, keyed by detector.

    This is only set by `FitBackgroundTask.runQuantum`; it may be `None` even
    if the sky frame was fit.
    """

    sky_frame_factor: float | None = None
    """Factor the sky frame is multiplied by in this model."""

    coefficients: list[pydantic.StrictFloat] = pydantic.Field(default_factory=list)
    """Model parameters, not including the sky frame factor.

    This may be empty if the model has not been fit yet.  Chebyshev
    coefficients are first, followed optionally by either ``detector_ids``
    pistons or ``detector_types`` pistons, in the order set by those
    attributes.
    """

    def set_detector_types(self, detectors: Iterable[Detector]) -> None:
        """Set the `detector_types` attribute with all of the unique physical
        types of the given detectors, in deterministic order.
        """
        self.detector_types = sorted(set(detector.getPhysicalType() for detector in detectors))

    def make_matrix(
        self,
        x: np.ndarray,
        y: np.ndarray,
        detector_slices: Mapping[int, slice],
        detectors: Mapping[int, Detector],
        drop_chebyshev0: bool,
    ) -> np.ndarray:
        """Construct a matrix that can be used to fit for the coefficients of
        this model or evaluate the model given coefficients.

        Like the coefficients, this matrix never includes the sky frame term,
        which must be applied separately.

        Parameters
        ----------
        x : `numpy.ndarray`
            Focal plane X coordinates of the data points.
        y : `numpy.ndarray`
            Focal plane Y coordinates of the data points.
        detector_slices : `~collections.abc.Mapping` [`int`, `slice`]
            Mapping from detector ID to the slice that corresponds to that
            detector in ``x`` and ``y``.
        detectors : `~collections.abc.Mapping` [`int`,\
                `lsst.afw.cameraGeom.Detector`]
            Descriptions of all detectors that the matrix will evaluate on.
        drop_chebyshev0 : `bool`
            Drop the zeroth-order Chebyshev polynomial term.  This is useful
            for models where some other combination of terms would be fully
            degenerate with it.

        Returns
        -------
        matrix : `numpy.ndarray`
            A 2-d matrix with shape ``(len(x), len(coefficients))``, where the
            latter is the length the ``coefficients`` *will* have after the
            fit if the array is empty.
        """
        ctrl = ChebyshevBoundedFieldControl()
        ctrl.orderX = self.chebyshev_basis.x_order
        ctrl.orderY = self.chebyshev_basis.y_order
        chebyshev_matrix = ChebyshevBoundedField.makeFitMatrix(self.chebyshev_basis.bbox, x, y, ctrl)
        if drop_chebyshev0:
            chebyshev_matrix = chebyshev_matrix[:, 1:]
        n_data = chebyshev_matrix.shape[0]
        n_chebyshev = chebyshev_matrix.shape[1]
        n_coeffs = n_chebyshev
        if self.detector_ids is not None:
            n_coeffs = n_chebyshev + len(self.detector_ids)
            matrix = np.zeros((n_data, n_coeffs), dtype=float)
            matrix[:, :n_chebyshev] = chebyshev_matrix
            for j, detector_id in enumerate(self.detector_ids, start=n_chebyshev):
                detector_slice = detector_slices.get(detector_id, slice(0, 0))
                matrix[detector_slice, j] = 1.0
        elif self.detector_types is not None:
            n_coeffs = n_chebyshev + len(self.detector_types)
            matrix = np.zeros((n_data, n_coeffs), dtype=float)
            matrix[:, :n_chebyshev] = chebyshev_matrix
            for detector_id, detector in detectors.items():
                j = n_chebyshev + self.detector_types.index(detector.getPhysicalType())
                matrix[detector_slices[detector_id], j] = 1.0
        else:
            matrix = chebyshev_matrix
        return matrix


class FitVisitBackgroundConnections(PipelineTaskConnections, dimensions=["visit"]):
    camera = cT.PrerequisiteInput(
        "camera",
        storageClass="Camera",
        dimensions=["instrument"],
        isCalibration=True,
    )
    input_backgrounds = cT.Input(
        "preliminary_visit_image_background",
        storageClass="Background",
        multiple=True,
        dimensions=["visit", "detector"],
    )
    sky_frame_backgrounds = cT.PrerequisiteInput(
        "sky_frame_background",
        storageClass="Background",
        multiple=True,
        dimensions=["detector", "physical_filter"],
        isCalibration=True,
    )
    output_backgrounds = cT.Output(
        "visit_background_diff",
        storageClass="Background",
        multiple=True,
        dimensions=["visit", "detector"],
    )
    model = cT.Output(
        "visit_background_model",
        storageClass="VisitBackgroundModel",
        dimensions=["visit"],
    )

    def __init__(self, *, config=None):
        assert isinstance(config, FitVisitBackgroundConfig)
        if not config.fit_sky_frame:
            del self.sky_frame_binned

    def adjustQuantum(self, inputs, outputs, label, data_id):
        if self.config.fit_sky_frame:
            # Make sure we have a sky frame for every detector, and trim out
            # any we don't need.
            _, input_refs = inputs["input_backgrounds"]
            sky_connection, sky_refs = inputs["sky_frame_backgrounds"]
            input_refs_by_detector = {ref.dataId["detector"]: ref for ref in input_refs}
            sky_refs_by_detector = {ref.dataId["detector"]: ref for ref in sky_refs}
            if not (sky_refs_by_detector.keys() >= input_refs_by_detector.keys()):
                missing = input_refs_by_detector.keys() - sky_refs_by_detector.keys()
                raise FileNotFoundError(
                    f"No binned sky frame for detector(s) {missing} in visit {data_id['visit']}."
                )
            inputs["sky_frame_backgrounds"] = (
                sky_connection,
                [sky_refs_by_detector[d] for d in input_refs_by_detector.keys()],
            )
        return super().adjustQuantum(inputs, outputs, label, data_id)


class FitVisitBackgroundConfig(PipelineTaskConfig, pipelineConnections=FitVisitBackgroundConnections):
    x_order = Field[int](
        "Chebyshev polynomial order in focal-plane X.",
        dtype=int,
        default=6,
    )
    y_order = Field[int](
        "Chebyshev polynomial order in focal-plane Y.",
        dtype=int,
        default=6,
    )
    fit_detector_pistons = Field[bool](
        "If True, fit a constant to each detector in instead of the zeroth-order Chebyshev term.",
        dtype=bool,
        default=False,
    )
    fit_type_pistons = Field[bool](
        "If True, fit a constant to each detector type (as defined by Detector.getPhysicalType()) "
        "instead of the zeroth-order Chebyshev term.",
        dtype=bool,
        default=False,
    )
    fit_sky_frame = Field[bool](
        "If True, fit with the binned sky frame as another basis function.",
        dtype=bool,
        default=True,
    )
    clip_positive_threshold = Field[float](
        "Clip sample points with (data - model) greater than this threshold and re-fit.",
        dtype=float,
        optional=True,
        default=50.0,
    )
    clip_negative_threshold = Field[float](
        "Clip sample points with (data - model) less than this threshold and re-fit.",
        dtype=float,
        optional=True,
        default=None,
    )
    clip_iterations = Field[int](
        "Number of clip-and-refit iterations to perform.  Ignored if no clipping is enabled.",
        dtype=int,
        default=10,
    )

    def validate(self):
        if self.fit_detector_pistons and self.fit_type_pistons:
            raise ValueError("Only one of 'fit_detector_pistons' and 'fit_type_pistons' may be True.")
        return super().validate()


class FitVisitBackgroundTask(PipelineTask):
    """A task that fits a visit-level background by fitting a larger-scale
    model to the binned values originally used in per-detector backgrounds.

    Notes
    -----
    This task requires input `lsst.afw.math.BackgroundList` instances in which
    all layers (i.e. from different rounds of subtraction) either use the same
    bin size or have a single bin.
    """

    ConfigClass: ClassVar[type[FitVisitBackgroundConfig]] = FitVisitBackgroundConfig

    _DefaultName: ClassVar[str] = "fitVisitBackground"

    config: FitVisitBackgroundConfig

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        camera = butlerQC.get(inputRefs.camera)
        input_backgrounds = {
            ref.dataId["detector"]: butlerQC.get(ref)
            for ref in sorted(inputRefs.input_backgrounds, key=lambda ref: ref.dataId["detector"])
        }
        sky_frame_backgrounds: dict[int, BackgroundList] | None = None
        if self.config.fit_sky_frame:
            sky_frame_backgrounds = {
                ref.dataId["detector"]: butlerQC.get(ref)
                for ref in sorted(inputRefs.sky_frame_backgrounds, key=lambda ref: ref.dataId["detector"])
            }
        outputs = self.run(
            camera=camera, input_backgrounds=input_backgrounds, sky_frame_backgrounds=sky_frame_backgrounds
        )
        # Add provenance that `run` doesn't have access to.
        outputs.model.sky_frame_datasets = {
            ref.dataId["detector"]: ref.id for ref in inputRefs.sky_frame_backgrounds
        }
        for ref in outputRefs.output_backgrounds:
            if (bg := outputs.output_backgrounds.get(ref.dataId["detector"])) is not None:
                butlerQC.put(bg, ref)
        butlerQC.put(outputs.model, outputRefs.model)

    def run(
        self,
        *,
        camera: Camera,
        input_backgrounds: Mapping[int, BackgroundList],
        sky_frame_backgrounds: Mapping[int, BackgroundList] | None = None,
    ) -> Struct:
        """Fit a Chebyshev and optional detector pistons to pre-binned
        background statistics images.

        Parameters
        ----------
        camera : `lsst.afw.cameraGeom.Camera`
            Description of the camera.
        input_backgrounds : `~collections.abc.Mapping` [
                `int`, `lsst.afw.math.BackgroundList`]
            Input background models.
        sky_frame_backgrounds : `~collections.abc.Mapping` [
                `int`, `lsst.afw.math.BackgroundList`]
            Background models subtracted from the sky frame.

        Returns
        -------
        outputs : `lsst.pipe.base.Struct`
            Result struct with the following attributes:

            - ``output_backgrounds``: `dict` mapping detector ID to the new
              `lsst.afw.math.BackgroundList`.  These should be applied to
              images that already have the original input background
              subtracted.
            - ``fit``: `VisitBackgroundFit` instance with the details of the
              final fit.
            - ``model``: `VisitBackgroundModel` instance that allows the
              full model to be serialized.
        """
        self.log.info("Extracting input samples from %s detectors.", len(input_backgrounds))
        per_detector_data: dict[int, FitInputData] = {}
        detectors: dict[int, Detector] = {}
        for detector_id, bg_list in input_backgrounds.items():
            self.log.verbose("Extracting input samples for detector=%s.", detector_id)
            detector = camera[detector_id]
            sky_frame_bg_list = None
            if self.config.fit_sky_frame:
                sky_frame_bg_list = sky_frame_backgrounds[detector_id]
            per_detector_data[detector_id] = self._gather_input_samples(
                detector, bg_list, sky_frame_bg_list=sky_frame_bg_list
            )
            detectors[detector_id] = detector
        full_data = FitInputData.concatenate(per_detector_data)
        self.log.info(
            "Fitting focal-plane model to %s data points from %s detectors.",
            len(full_data),
            len(detectors),
        )
        model = VisitBackgroundModel(
            chebyshev_basis=ChebyshevBasis.from_bbox(
                self._compute_camera_bbox(camera),
                self.config.x_order,
                self.config.y_order,
            )
        )
        if self.config.fit_detector_pistons:
            model.detector_ids = list(full_data.detector_slices.keys())
        elif self.config.fit_type_pistons:
            model.set_detector_types(detectors.values())
        matrix = model.make_matrix(
            full_data.x,
            full_data.y,
            full_data.detector_slices,
            detectors=detectors,
            drop_chebyshev0=(self.config.fit_detector_pistons or self.config.fit_type_pistons),
        )
        fit = full_data.fit(matrix)
        self.log.info(
            "Fit %s-parameter focal-plane model to %s data points.",
            len(fit.coefficients) + (fit.sky_frame_factor is not None),
            len(fit.data),
        )
        if self.config.clip_positive_threshold is not None or self.config.clip_negative_threshold is not None:
            for n_iter in range(self.config.clip_iterations):
                if self.config.clip_positive_threshold is not None:
                    bad = fit.residuals > self.config.clip_positive_threshold
                    if self.config.clip_negative_threshold is not None:
                        bad = np.logical_or(bad, fit.residuals < self.config.clip_negative_threshold)
                else:
                    bad = fit.residuals < self.config.clip_negative_threshold
                n_clipped = sum(bad)
                if not n_clipped:
                    break
                fit = fit.masked(np.logical_not(bad))
                self.log.info("Clipped %s data points and re-fit for iteration %s", sum(bad), n_iter)
        model.coefficients = fit.coefficients.tolist()
        model.sky_frame_factor = fit.sky_frame_factor
        self.log.info("Projecting focal-plane model back to detectors.")
        output_backgrounds = {
            detector_id: self._make_output_background(
                detector,
                model,
                input_backgrounds[detector_id],
                sky_frame_bg_list=(
                    sky_frame_backgrounds[detector_id] if sky_frame_backgrounds is not None else None
                ),
            )
            for detector_id, detector in detectors.items()
        }
        return Struct(output_backgrounds=output_backgrounds, model=model, fit=fit)

    @staticmethod
    def _compute_camera_bbox(camera: Camera) -> Box2I:
        """Compute the bounding box of a camera in focal-plane coordinates."""
        bbox = Box2D()
        for detector in camera:
            for corner in detector.getCorners(FOCAL_PLANE):
                bbox.include(corner)
        return Box2I(bbox)

    def _make_bin_grid_fp_coordinates(
        self, detector: Detector, background: BackgroundMI
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct arrays of focal-plane coordinates that correspond to the
        bin centers of the given background object.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            Detector the background is for.
        background : `lsst.afw.math.BackgroundMI`
            A single layer of a background model.  All other models for this
            detector are assumed to have the same bins.

        Returns
        -------
        x_fp : `numpy.ndarray`
            Array of focal-plane X coordinates.
        y_fp : `numpy.ndarray`
            Array of focal-plane Y coordinates.
        """
        pix_to_fp = detector.getTransform(PIXELS, FOCAL_PLANE)
        x2_pix, y2_pix = np.meshgrid(background.getBinCentersX(), background.getBinCentersY())
        xy_fp = pix_to_fp.getMapping().applyForward(np.vstack((x2_pix.ravel(), y2_pix.ravel())))
        return xy_fp[0], xy_fp[1]

    def _gather_input_samples(
        self, detector: Detector, bg_list: BackgroundList, sky_frame_bg_list: BackgroundList | None = None
    ) -> FitInputData:
        """Transform input background-estimation bin values to focal-plane
        coordinates.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            Detector the background is for.
        bg_list : `lsst.afw.math.BackgroundList`
            Input background model for this detector.
        sky_frame_bg_list : `lsst.afw.math.BackgroundList`, optional
            Background model constructed from the sky frame for this detector.
            Ignored unless `~FitVisitBackgroundConfig.fit_sky_frame` is `True`.

        Returns
        -------
        data : `FitInputData`
            Struct containing finiteness-filtered bin values and their
            focal plane coordinates.
        """
        sum_image, bg_for_coordinates = self._sum_bg_list(bg_list)
        x_fp, y_fp = self._make_bin_grid_fp_coordinates(detector, bg_for_coordinates)
        z = sum_image.image.array.ravel().astype(float)
        w = sum_image.variance.array.ravel().astype(float) ** (-0.5)
        sky_frame: np.ndarray | None = None
        if self.config.fit_sky_frame:
            assert sky_frame_bg_list is not None, "Should be guaranteed by caller."
            sky_frame_sum_image, _ = self._sum_bg_list(sky_frame_bg_list)
            if sky_frame_sum_image.getBBox() != sum_image.getBBox():
                raise InvalidQuantumError(
                    f"Binned sky frame for detector {detector.getId()} has "
                    f"{sky_frame_sum_image.getWidth()}×{sky_frame_sum_image.getHeight()} bins, but "
                    f"background has {sum_image.getWidth()}×{sum_image.getHeight()}."
                )
            sky_frame = sky_frame_sum_image.image.array.ravel().astype(float)
        return FitInputData(x=x_fp, y=y_fp, z=z, w=w, sky_frame=sky_frame).masked(
            np.logical_and(w > 0.0, np.isfinite(z))
        )

    def _sum_bg_list(self, bg_list: BackgroundList) -> tuple[MaskedImageF, BackgroundMI]:
        sum_image: MaskedImageF | None = None
        sum_constant_value: float = 0.0
        sum_constant_variance: float = 0.0
        bg_for_coordinates: BackgroundMI | None = None
        for bg, *_ in bg_list:
            layer_image = bg.getStatsImage()
            if layer_image.getBBox().getArea() == 1:
                sum_constant_value += layer_image.image.array[0, 0]
                sum_constant_variance += layer_image.variance.array[0, 0]
            elif sum_image is not None:
                sum_image += layer_image
            else:
                sum_image = layer_image.clone()
                bg_for_coordinates = bg
        sum_image.image.array[:, :] += sum_constant_value
        sum_image.variance.array[:, :] += sum_constant_variance
        return sum_image, bg_for_coordinates

    def _make_output_background(
        self,
        detector: Detector,
        model: VisitBackgroundModel,
        original_bg_list: BackgroundList,
        sky_frame_bg_list: BackgroundList | None,
    ) -> BackgroundMI:
        """Convert the focal-plane background model into a per-detector
        background objet.

        Parameters
        ----------
        detector : `lsst.afw.cameraGeom.Detector`
            Detector the background is for.
        model : `VisitBackgroundModel`
            Focal-plane background model.
        original_bg_list : `lsst.afw.math.BackgroundList`
            Input background model for this detector.  THESE MODELS ARE
            MODIFIED IN PLACE AND SHOULD BE CONSIDERED CONSUMED.
        sky_frame_bg_list : `lsst.afw.math.BackgroundList`, optional
            Background model constructed from the sky frame for this detector.
            Ignored unless `~FitVisitBackgroundConfig.fit_sky_frame` is `True`.
            THESE MODELS ARE MODIFIED IN PLACE AND SHOULD BE CONSIDERED
            CONSUMED.

        Returns
        -------
        bg_list : `lsst.afw.math.BackgroundList`
            Updated background model for this detector.  This reverts the
            input background before applying the new one.
        """
        # Make the background's "stats image" plenty big to exactly constrain
        # a perfect Chebyshev fit (should actually be ~2x as many points as
        # we need), since the background object insists on re-fitting a
        # Chebyshev to the stats image every time you make a full-size image.
        n_bins_x = self.config.x_order + 1
        n_bins_y = self.config.y_order + 1
        stats_image = MaskedImageF(n_bins_x, n_bins_y)
        new_bg = BackgroundMI(detector.getBBox(), stats_image)
        x_fp, y_fp = self._make_bin_grid_fp_coordinates(detector, new_bg)
        matrix = model.make_matrix(
            x_fp,
            y_fp,
            detector_slices={detector.getId(): slice(0, None)},
            detectors={detector.getId(): detector},
            drop_chebyshev0=(self.config.fit_detector_pistons or self.config.fit_type_pistons),
        )
        z = np.dot(matrix, model.coefficients)
        stats_image.image.array[:, :] = z.reshape(n_bins_y, n_bins_x)
        stats_image.variance.array[:, :] = 1.0  # just to keep everything finite; we won't use weights
        ctrl = new_bg.getBackgroundControl()
        ctrl.setInterpStyle(Interpolate.Style.CUBIC_SPLINE)
        ctrl.setUndersampleStyle(UndersampleStyle.THROW_EXCEPTION)
        actrl = ApproximateControl(
            ApproximateControl.Style.UNKNOWN,
            self.config.x_order,
            self.config.y_order,
            False,
        )
        ctrl.setApproximateControl(actrl)
        result = original_bg_list
        for revert_bg, *_ in result:
            stats_image = revert_bg.getStatsImage()
            stats_image *= -1
        if sky_frame_bg_list:
            for sky_frame_bg, *rest in sky_frame_bg_list:
                sky_frame_stats_image = sky_frame_bg.getStatsImage()
                sky_frame_stats_image *= model.sky_frame_factor
                result.append((sky_frame_bg, *rest))
        result.append(
            (
                new_bg,
                ctrl.getInterpStyle(),
                ctrl.getUndersampleStyle(),
                actrl.getStyle(),
                actrl.getOrderX(),
                actrl.getOrderY(),
                False,
            )
        )
        return result
