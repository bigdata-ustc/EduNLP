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
.. image:: https://img.shields.io/pypi/v/EduNLP.svg
   :target: https://pypi.python.org/pypi/EduNLP

.. image:: https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml/badge.svg?branch=master
   :target: https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml

.. todo: add all badges in EduNLP/REAMD.md

`EduNLP <https://github.com/bigdata-ustc/EduNLP>`_ is a library for advanced Natural Language Processing in Python and is one of the projects of `EduX <https://github.com/bigdata-ustc/EduX>`_ plan of BDAA.
It's built on the very latest research, and was designed from day one to be used in real educational products.

EduNLP now comes with pretrained pipelines and currently supports segment, tokenization and vertorization. It supports varies of preprocessing for NLP in educational scenario, such as formula parsing, multi-modal segment.

EduNLP is commercial open-source software, released under the `Apache-2.0 license <https://github.com/bigdata-ustc/EduNLP/blob/master/LICENSE>`_.

Install
---------
EduNLP requires Python version 3.6, 3.7, 3.8 or 3.9. EduNLP use PyTorch as the backend tensor library.

We recommend installing EduNLP by ``pip``:

::

   pip install EduNLP

But you can also install from source:

::

   git clone https://github.com/bigdata-ustc/EduNLP.git
   cd EduNLP
   pip install .



Getting Started
------------------
For absolute beginners, start with the :doc:`Tutorial to EduNLP <tutorial/en/index>` :doc:`(中文版) <tutorial/zh/index>`.
It covers the basic concepts of EduNLP and
a step-by-step on training, loading and using the language models.


Contribution
--------------
EduNLP is free software; you can redistribute it and/or modify it under the terms of the Apache License 2.0.
We welcome contributions. Join us on GitHub and check out our `contribution guidelines <https://github.com/bigdata-ustc/EduNLP/blob/master/CONTRIBUTE.md>`_ `(中文版) <https://github.com/bigdata-ustc/EduNLP/blob/master/CONTRIBUTE_CH.md>`_.

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

.. toctree::
   :maxdepth: 1
   :caption: 用户指南
   :hidden:

   tutorial/zh/index
   tutorial/zh/sif
   tutorial/zh/seg
   tutorial/zh/parse
   tutorial/zh/tokenize
   tutorial/zh/vectorization
   tutorial/zh/pretrain


.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:
   :glob:

   api/index
   api/i2v
   api/sif
   api/formula
