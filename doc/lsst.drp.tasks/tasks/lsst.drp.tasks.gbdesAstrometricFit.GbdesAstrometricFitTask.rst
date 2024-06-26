.. lsst-task-topic:: lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask

#######################
GbdesAstrometricFitTask
#######################

``GbdesAstrometricFitTask`` fits an astrometric solution, or world coordinate system (WCS), for a series of visits.
Sources in science images are matched with a reference catalog, which allows the fitting algorithm to break degeneracies in the astrometric model.
The fit model is a series of mappings, which can include (but are not limited to) a per-detector polynomial mapping and a per-visit polynomial mapping.
The fit is done using the `WCSFit` class from the `gbdes` package (https://github.com/lsst/gbdes/tree/lsst-dev), which is an implementation of the method described in Bernstein et al. (2017).

This task requires source catalogs for the input images, a reference catalog, and a visit summary table, which provides a starting point for the WCS fit.
The task produces the fitted WCSs, which are described in AST format and are held in `lsst.afw.table.ExposureCatalog`\s called `outputWcs`.
The task also outputs a table for debugging, `outputCatalog`, which contains the sources used in the fit, along with their residuals with the best-fit model.

.. _lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask-summary:

Processing summary
==================

``GbdesAstrometricFitTask`` runs this sequence of operations:

#. Gathers information on the visit epochs, RAs and Decs, etc.

#. Associates sources in science images and the reference catalog to make matched objects using a friends-of-friends algorithm.

#. Uses the associated objects to fit the astrometric solution, which is a configurable sequence of mappings from the detector pixels to the sky.
This uses `WCSFit` from the `gbdes` package (https://github.com/lsst/gbdes/tree/lsst-dev).

#. Converts the internal astrometric mappings to AST format.
The output is in the form of one `lsst.afw.table.ExposureCatalog` for each visit, with a row for each detector.
The fit WCS is accessed by calling `getWCS()` on the catalog row corresponding to the desired visit and detector.

.. _lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFit-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask

.. _lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask

.. _lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.drp.tasks.gbdesAstrometricFit.GbdesAstrometricFitTask
