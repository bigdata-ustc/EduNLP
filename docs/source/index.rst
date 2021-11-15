.. EduNLP documentation master file, created by
   sphinx-quickstart on Sat Aug  7 19:55:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===============================================
Welcome to EduNLP's Tutorials and Documentation
===============================================
.. Logo
.. image:: _static/EduNLP.png
   :width: 200px
   :align: center

.. Badges
.. image:: https://img.shields.io/pypi/pyversions/longling
   :target: https://pypi.python.org/pypi/longling
   :alt: VERSION

.. image:: https://img.shields.io/pypi/v/EduNLP.svg
   :target: https://pypi.python.org/pypi/EduNLP
   :alt: PyPI

.. image:: https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml/badge.svg?branch=master
   :target: https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml
   :alt: test

.. image:: https://codecov.io/gh/bigdata-ustc/EduNLP/branch/master/graph/badge.svg?token=B7gscOGQLD
   :target: https://codecov.io/gh/bigdata-ustc/EduNLP
   :alt: codecov

.. image:: https://readthedocs.org/projects/edunlp/badge/?version=latest
   :target: https://edunlp.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/dm/EduNLP.svg?style=flat
   :target: https://pypi.python.org/pypi/EduNLP
   :alt: Download

.. image:: https://zenodo.org/badge/332661206.svg
   :target: https://zenodo.org/badge/latestdoi/332661206
   :alt: DOI

.. image:: https://img.shields.io/github/license/bigdata-ustc/EduNLP
   :target: https://github.com/bigdata-ustc/EduNLP/blob/master/LICENSE
   :alt: License


`EduNLP <https://github.com/bigdata-ustc/EduNLP>`_ is a library for advanced Natural Language Processing in Python and is one of the projects of `EduX <https://github.com/bigdata-ustc/EduX>`_ plan of BDAA.
It's built on the very latest research, and was designed from day one to be used in real educational products.

EduNLP now comes with pretrained pipelines and currently supports segment, tokenization and vertorization. It supports varies of preprocessing for NLP in educational scenario, such as formula parsing, multi-modal segment.

EduNLP is commercial open-source software, released under the `Apache-2.0 license <https://github.com/bigdata-ustc/EduNLP/blob/master/LICENSE>`_.

Install
---------
EduNLP requires Python version 3.6, 3.7, 3.8 or 3.9. EduNLP use PyTorch as the backend tensor library.

We recommend installing EduNLP by ``pip``:

::

   # basic installation
   pip install EduNLP
   
   # full installation
   pip install EduNLP[full]


But you can also install from source:

::

   git clone https://github.com/bigdata-ustc/EduNLP.git
   cd EduNLP

   # basic installation
   pip install .
   
   # full installation
   pip install .[full]



Getting Started
------------------

One basic usage of EduNLP is to convert an item into a vector, i.e.,

.. code-block:: python

   from EduNLP import get_pretrained_i2v
   i2v = get_pretrained_i2v("d2v_all_256", "./model")
   item_vector, token_vector = i2v(["the content of item 1", "the content of item 2"])


For absolute beginners, start with the :doc:`Tutorial to EduNLP <tutorial/en/index>` :doc:`(中文版) <tutorial/zh/index>`.
It covers the basic concepts of EduNLP and
a step-by-step on training, loading and using the language models.

Resource
--------------

We will continuously publish new datasets in `Standard Item Format (SIF) <https://github.com/bigdata-ustc/EduNLP/blob/master/docs/SIF4TI_CH.md>`_ to encourage the relevant research works. The data resources can be accessed via another EduX project `EduData <https://github.com/bigdata-ustc/EduData>`_

Contribution
--------------
EduNLP is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0.
We welcome contributions. Join us on GitHub and check out our `contribution guidelines <https://github.com/bigdata-ustc/EduNLP/blob/master/CONTRIBUTE.md>`_ `(中文版) <https://github.com/bigdata-ustc/EduNLP/blob/master/CONTRIBUTE_CH.md>`_.

Citation
--------------

If this repository is helpful for you, please cite our work





::

  @misc{bigdata2021edunlp,
   title={EduNLP},
   author={bigdata-ustc},
   publisher = {GitHub},
   journal = {GitHub repository},
   year = {2021},
   howpublished = {\url{https://github.com/bigdata-ustc/EduNLP}},
 }


.. toctree::
   :caption: Introduction
   :hidden:

   self

.. toctree::
   :maxdepth: 1
   :caption: Tutorial
   :hidden:
   :glob:

   tutorial/en/index
   tutorial/en/sif
   tutorial/en/parse
   tutorial/en/seg
   tutorial/en/tokenize
   tutorial/en/pretrain
   tutorial/en/vectorization

.. toctree::
   :maxdepth: 1
   :caption: 用户指南
   :hidden:

   tutorial/zh/index
   tutorial/zh/sif
   tutorial/zh/parse
   tutorial/zh/seg
   tutorial/zh/tokenize
   tutorial/zh/pretrain
   tutorial/zh/vectorization


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :glob:

   api/sif
   api/utils
   api/formula
   api/tokenizer
   api/pretrain
   api/ModelZoo
   api/i2v
   api/vector
   
