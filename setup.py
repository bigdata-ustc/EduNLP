from setuptools import setup, find_packages

pretrain_deps = [
    "gensim"
]
tutor_deps = [
    "pillow",
    "tqdm"
]
test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
    "pillow",
    "gensim"
]

dev_deps = [
    "requests"
]

setup(
    name='EduNLP',
    version='0.0.4',
    extras_require={
        'test': test_deps,
        'tutor': tutor_deps,
        'pretrain': pretrain_deps,
        "dev": dev_deps
    },
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy>=1.17.0',
        'jieba',
        'js2py',
        'torch',
        'EduData>=0.0.16',
        'PyBaize[torch]>=0.0.3'
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
        "License :: OSI Approved :: Apache License 2.0 (Apache 2.0)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
