.. lsst-task-topic:: lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask

#######################
ReprocessVisitImageTask
#######################

``ReprocessVisitImageTask`` applies calibration-model objects produced by upstream tasks to single-visit images, and performs the final round of detection, deblending, and measurement that will populate the Source table.

.. _lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask-summary:

Processing summary
==================

``ReprocessVisitImageTask`` reads in the one or two post-ISR single-exposure images that correspond to a single ``{visit, detector}`` combination, combines them into a single image (via :lsst-task:`~lsst.pipe.base.snapCombine.SnapCombineTask`).
It then applies calibration-model objects produced upstream.
These include:

- the point-spread function model (`lsst.afw.detection.Psf`)
- aperture corrections (`lsst.afw.image.ApCorrMap`)
- the astrometric solution (`lsst.afw.geom.SkyWcs`)
- the photometric solution (`lsst.afw.image.PhotoCalib`)
- a background model (`lsst.afw.math.BackgroundList`)

``ReprocessVisitImageTask`` then detects new sources (:lsst-task:`~lsst.meas.algorithms.detection.SourceDetectionTask`), deblends them (:lsst-task:`~lsst.meas.deblender.SourceDeblendTask`), measures them (:lsst-task:`~lsst.meas.base.SingleFrameMeasurementTask`), and applies the aperture corrections loaded previously (:lsst-task:`~lsst.meas.algorithms.ApplyApCorrTaskTask`).
Additional subtasks or configurations of :lsst-task:`~lsst.meas.base.SingleFrameMeasurementTask` may be used to ensure the calibration models are correctly reflected in all measurements, but these should be considered implementation details, despite being visible (and configurable) as regular subtasks.

The task's outputs are an `lsst.afw.image.ExposureF` that has been reprocessed and calibrated with the input calibration models and the `~lsst.afw.table.SourceCatalog` of single-visit measurements on that exposure.
The background model is subtracted from the image and the photometric calibration is used to scale it to ``nJy`` pixel values, while the other calibration model objects are just attached to the stored image.

See connection and ``run`` argument documentation for details.

.. _lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask

.. _lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask-butler:

Butler datasets
===============

When run through the `~lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask.runQuantum` method, ``ReprocessVisitImageTask`` obtains datasets from the input Butler data repository and persists outputs to the output Butler data repository.

In this mode, the inputs are:

- a single-exposure, single-detector post-ISR image (``postISRCCD``, by default)
- a ``visit``\-level catalog with per-``detector`` rows (``finalVisitSummary``) that aggregates most calibration model objects.
- two background models (`calexpBackground``, ``skyCorr``), to be concatenated to form the full background model.

The outputs are:

- the final image, including all calibration models (``pvi``)
- the catalog of measurements (``sources_detector``)
- a separate catalog of deblended footprints (``sources_detector_footprints``)
- a final background model, including both the input background models and a small adjustment that uses the new detections for masking (``pvi_background``).

.. _lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask

.. _lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.drp.tasks.reprocess_visit_image.ReprocessVisitImageTask
