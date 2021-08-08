import logging
from setuptools import setup, find_packages

tutor_deps = [
    "pillow",
    "tqdm",
    "ipython"
]
test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]
docs_deps = [
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx_toggleprompt',
    'sphinx-gallery>=0.6',
    'nbsphinx'
]

dev_deps = ["requests"] + docs_deps + test_deps

try:
    import torch

    ml_pytorch_deps = []
except ModuleNotFoundError:
    import sys

    if 5 <= sys.version_info[1]:
        ml_pytorch_deps = ["torch"]
    else:
        ml_pytorch_deps = []
        logging.warning("Current python version %s is not supported by pytorch", str(sys.version_info[:2]))

vec_deps = ['gensim'] + ml_pytorch_deps

setup(
    name='EduNLP',
    version='0.0.4',
    extras_require={
        'test': test_deps,
        'doc': docs_deps,
        'tutor': tutor_deps,
        'dev': dev_deps,
        'vec': vec_deps,
        'full': vec_deps + tutor_deps
    },
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy>=1.17.0',
        'jieba',
        'js2py',
        'EduData>=0.0.16',
        'PyBaize>=0.0.3'
    ],  # And any other dependencies foo needs
    entry_points={
        "console_scripts": [
            "edunlp = EduNLP.main:cli",
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
