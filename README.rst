#########
drp_tasks
#########

``drp_tasks`` is a package in the `LSST Science Pipelines <https://pipelines.lsst.io>`_.

PipelineTasks that are used only in data release pipelines (not alert or calibration production).
Note that many tasks that fit this description remain in `pipe_tasks <https://github.com/lsst/pipe_tasks>`_. for historical reasons.

The package namespace itself is mostly empty. Each specific processing tool must be imported directly from the module; for instance,

``from lsst.drp.tasks.assemble_coadd import AssembleCoaddTask``
