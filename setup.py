from setuptools import setup, find_packages

test_deps = [
    'pytest>=4',
    'pytest-cov>=2.6.0',
    'pytest-flake8',
    "pillow"
]

tutor_deps = [
    "pillow",
]

setup(
    name='EduNLP',
    version='0.0.2',
    extras_require={
        'test': test_deps,
        "tutor": tutor_deps
    },
    packages=find_packages(),
    install_requires=[
        'networkx',
        'numpy',
        'jieba',
        'js2py'
    ],  # And any other dependencies foo needs
    entry_points={
    },
)
