.. lsst-task-topic:: lsst.drp.tasks.WCSFitTask

##########
WCSFitTask
##########

``WCSFitTask`` fits an astrometric solution, or world coordinate system (WCS), for a series of visits. Sources in science images are matched with a reference catalog, which allows the fitting algorithm to break degeneracies in the astrometric model. The fit model is a series of mappings, which can include (but are not limited to) a per-detector polynomial mapping and a per-visit polynomial mapping.

This task requires source catalogs for the input images, a reference catalog, and a visit summary table, which provides a starting point for the WCS-fit. The task produces the fitted WCSs, which are described in AST format and are held in `lsst.afw.table.ExposureCatalog`\s called `outputWcs`. The task also outputs a table for debugging, `outputCatalog`, which contains the sources used in the fit, alongs with their residuals with the best-fit model.

.. _lsst.drp.tasks.WCSFitTask-summary:

Processing summary
==================

``WCSFitTask`` runs this sequence of operations:

#. Gathers information on the visit epochs, RAs and Decs, etc.

#. Associates sources in science images and the reference catalog to make matched objects using a friends-of-friends algorithm.

#. Uses the associated objects to fit the astrometric solution, which is a configurable sequence of mappings from the detector pixels to the sky. This uses the `wcsfit` package from https://github.com/lsst/gbdes/tree/lsst-dev.

#. Converts the internal astrometric mappings to AST format. The output is in the form of one `lsst.afw.table.ExposureCatalog` for each visit, with a row for each detector. The fit WCS is accessed by calling `getWCS()` on the catalog row corresponding to the desired visit and detector.

.. _lsst.drp.tasks.WCSFitTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.drp.tasks.WCSFitTask

.. _lsst.drp.tasks.WCSFitTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.drp.tasks.WCSFitTask

.. _lsst.drp.tasks.WCSFitTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.drp.tasks.WCSFitTask
