.. pipeline documentation master file

Pipeline: Efficient Parallelism for Training
=====================================

.. image:: _static/logo.webp
   :alt: Pipeline Logo
   :align: right
   :width: 100px
   :class: rounded-corners

Welcome to the documentation for Pipeline, a library for efficient parallelism in deep learning.

.. raw:: html

   <style>
   .rounded-corners {
       border-radius: 10px;
   }
   </style>

Features
--------

- Automatic model partitioning
- Support for various scheduling algorithms
- Easy integration with existing PyTorch models
- Profiling with Nsight Systems

Quick Start
-----------

.. code-block:: python

   from elf import Pipeline
   
   model = YourLargeModel()
   pipe = Pipeline(model, sample)

   output, loss = pipe(inputs, targets, loss_fn)


API Reference
-------------

Pipeline Object
~~~~~~~~~~~~~~~

.. autoclass:: pipeline.Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst
   :recursive:

   pipeline
   pipeline.partitioners

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
