<p align="center">
  <img width="300" src="docs/EduNLP.png">
</p>

# EduNLP

[![PyPI](https://img.shields.io/pypi/v/EduNLP.svg)](https://pypi.python.org/pypi/EduNLP)
[![test](https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml/badge.svg?branch=master)](https://github.com/bigdata-ustc/EduNLP/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/gh/bigdata-ustc/EduNLP/branch/master/graph/badge.svg?token=B7gscOGQLD)](https://codecov.io/gh/bigdata-ustc/EduNLP)
[![Documentation Status](https://readthedocs.org/projects/edunlp/badge/?version=latest)](https://edunlp.readthedocs.io/en/latest/?badge=latest)
[![Download](https://img.shields.io/pypi/dm/EduNLP.svg?style=flat)](https://pypi.python.org/pypi/EduNLP)
[![License](https://img.shields.io/github/license/bigdata-ustc/EduNLP)](LICENSE)
[![DOI](https://zenodo.org/badge/332661206.svg)](https://zenodo.org/badge/latestdoi/332661206)


EduNLP is a library for advanced Natural Language Processing in Python and is one of the projects of [EduX]((https://github.com/bigdata-ustc/EduX)) plan of [BDAA](https://github.com/bigdata-ustc). It's built on the very latest research, and was designed from day one to be used in real educational products.

EduNLP now comes with pretrained pipelines and currently supports segment, tokenization and vertorization. It supports varies of preprocessing for NLP in educational scenario, such as formula parsing, multi-modal segment.

EduNLP is commercial open-source software, released under the [Apache-2.0 license](LICENSE).

## Quickstart

### Installation

Git and install by pip
``` sh
# basic installation
pip install .

# full installation
pip install .[full]
```
or install from pypi:
```
# basic installation
pip install EduNLP

# full installation
pip install EduNLP[full]
```

### Tutorial

For more details, please refer to the full documentation ([latest](https://edunlp.readthedocs.io/en/latest) | [stable](https://edunlp.readthedocs.io/en/stable)).

### Resource
We will continuously publish new datasets in [Standard Item Format (SIF)](https://github.com/bigdata-ustc/EduNLP/blob/master/docs/SIF4TI_CH.md) to encourage the relevant research works. The data resources can be accessed via another EduX project [EduData](https://github.com/bigdata-ustc/EduData)

## Contribute

EduNLP is still under development. More algorithms and features are going to be added and we always welcome contributions to help make EduNLP better. If you would like to contribute, please follow this [guideline](CONTRIBUTE.md)([开发指南](CONTRIBUTE_CH.md)).

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
