# EduNLP

[![PyPI](https://img.shields.io/pypi/v/EduNLP.svg)](https://pypi.python.org/pypi/EduNLP)
[![Build Status](https://www.travis-ci.org/bigdata-ustc/EduNLP.svg?branch=master)](https://www.travis-ci.org/bigdata-ustc/EduNLP)
[![codecov](https://codecov.io/gh/bigdata-ustc/EduNLP/branch/master/graph/badge.svg?token=B7gscOGQLD)](https://codecov.io/gh/bigdata-ustc/EduNLP)
[![Download](https://img.shields.io/pypi/dm/EduNLP.svg?style=flat)](https://pypi.python.org/pypi/EduNLP)
[![License](https://img.shields.io/github/license/bigdata-ustc/EduNLP)](LICENSE)
[![DOI](https://zenodo.org/badge/332661206.svg)](https://zenodo.org/badge/latestdoi/332661206)

NLP tools for Educational data (e.g., exercise, papers)

## Introduction
EduNLP is a library for advanced Natural Language Processing in Python and is one of the projects of EduX plan of BDAA. It's built on the very latest research, and was designed from day one to be used in real educational products.

EduNLP now comes with pretrained pipelines and currently supports segment, tokenization and vertorization. It supports varies of preprocessing for NLP in educational scenario, such as formula passing, multi-modal segment.

EduNLP is commercial open-source software, released under the Apache-2.0 license.

### Installation

Git and install by pip
```
pip install -e .
```
or install from pypi:
```
pip install EduNLP
```

### Resource
We will continously publish new datasets in [Standard Item Format (SIF)](https://github.com/bigdata-ustc/EduNLP/blob/master/docs/SIF4TI_CH.md) to encourage the relavant research works. The data resourses can be accessed via another EduX project [EduData](https://github.com/bigdata-ustc/EduData)

### Tutorial

* Overview (TBA)
* [Formula Parsing](https://github.com/bigdata-ustc/EduNLP/blob/master/examples/formula/formula.ipynb)
* [Segment and Tokenization](https://github.com/bigdata-ustc/EduNLP/blob/master/examples/sif/sif.ipynb)
* [Vectorization](https://github.com/bigdata-ustc/EduNLP/tree/master/examples/pretrain)
* Pretrained Model (TBA)

## Contribute

EduNLP is still under development. More algorithms and features are going to be added and we always welcome contributions to help make EduNLP better. If you would like to contribute, please follow this [guideline](CONTRIBUTE.md).

## Citation

If this repository is helpful for you, please cite our work

```
@misc{bigdata2021edunlp,
  title={EduNLP},
  author={bigdata-ustc},
  publisher = {GitHub},
  journal = {GitHub repository},
  year = {2021},
  howpublished = {\url{https://github.com/bigdata-ustc/EduNLP}},
}
```
