.. py:currentmodule:: lsst.drp.tasks

.. _lsst.drp.tasks:

##############
lsst.drp.tasks
##############

PipelineTasks that are used only in data release pipelines (not alert or calibration production).
Note that many tasks that fit this description remain in `pipe_tasks <https://github.com/lsst/pipe_tasks>`_ for historical reasons.

.. _lsst.drp.tasks-using:

Using lsst.drp.tasks
====================

.. toctree linking to topics related to using the module's APIs.

.. toctree::
   :maxdepth: 1

Task reference
==============

.. _lsst.drp.tasks-pipeline-tasks:

Pipeline tasks
--------------

.. lsst-pipelinetasks::
   :root: lsst.drp.tasks

.. _lsst.drp.tasks-contributing:

Contributing
============

``lsst.drp.tasks`` is developed at https://github.com/lsst/drp_tasks.
You can find Jira issues for this module under the `drp_tasks <https://rubinobs.atlassian.net/issues/?jql=project%20%3D%20%22DM%22%20AND%20component%20%3D%20%22drp_tasks%22>`_ component.

.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.drp.tasks-scripts:

.. Script reference
.. ================

.. .. TODO: Add an item to this toctree for each script reference topic in the scripts subdirectory.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.drp.tasks-pyapi:

Python API reference
====================

.. automodapi:: lsst.drp.tasks.assemble_cell_coadd
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.assemble_chi2_coadd
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.assemble_coadd
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.build_camera
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.compute_object_epochs
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.dcr_assemble_coadd
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.forcedPhotCoadd
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.gbdesAstrometricFit
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.make_direct_warp
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.make_psf_matched_warp
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.reprocess_visit_image
   :no-inheritance-diagram:

.. automodapi:: lsst.drp.tasks.update_visit_summary
   :no-inheritance-diagram:
