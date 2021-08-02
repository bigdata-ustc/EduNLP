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
    version='0.0.3',
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
        'EduData>=0.0.16'
    ],  # And any other dependencies foo needs
    entry_points={
        "console_scripts": [
            "edunlp = EduNLP.main:cli",
        ],
    },
)
