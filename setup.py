from setuptools import setup

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
]

setup(
    name='EduNLP',
    version='0.0.1',
    extras_require={
        'test': test_deps,
    },
    install_requires=[
        'networkx',
        'numpy'
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
