.. lsst-task-topic:: lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask

##################
MakeDirectWarpTask
##################

Warp single visit images (calexps or PVIs) onto a common projection by performing the following operations:

- Group the single-visit images by visit/run
- For each visit, generate a Warp by calling method @ref run.

.. _lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask-api:

Python API summary
==================

.. lsst-task-api-summary:: lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask

.. _lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask-subtasks:

Retargetable subtasks
=====================

.. lsst-task-config-subtasks:: lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask

.. _lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask-configs:

Configuration fields
====================

.. lsst-task-config-fields:: lsst.drp.tasks.make_direct_warp.MakeDirectWarpTask
