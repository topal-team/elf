.. pipeline documentation master file

ELF: Efficient deep Learning Framework
======================================

.. image:: _static/logo.webp
   :alt: Pipeline Logo
   :align: right
   :width: 100px
   :class: rounded-corners

Welcome to the documentation for ELF, a library for efficient parallelism in deep learning.

.. raw:: html

   <style>
   .rounded-corners {
       border-radius: 10px;
   }
   </style>

Features
--------

- Automatic model partitioning
- Support for various state-of-the-art scheduling algorithms
- Arbitrary skip connections
- Easy integration with existing PyTorch models
- Profiling with Nsight Systems

Quick Start
-----------

.. code-block:: python

   from elf import Pipeline
   
   model = YourLargeModel()
   pipe = Pipeline(model, sample)

   output, loss = pipe(inputs, targets, loss_fn)


User Guide
----------

.. toctree::
   :maxdepth: 2

   advanced_usage


API Reference
-------------

Pipeline Object
~~~~~~~~~~~~~~~

.. autoclass:: elf.Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated
   :template: custom-module-template.rst
   :recursive:

   elf

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
