import os.path

from setuptools import find_packages, setup

sklearn_dependency = "scikit-learn>=1.0.0"


def get_version():
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, "bofire/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    return ""


root_dir = os.path.dirname(__file__)
with open(os.path.join(root_dir, "README.md"), "r") as f:
    long_description = f.read()


setup(
    name="bofire",
    description="",
    author="",
    license="BSD-3",
    url="https://github.com/experimental-design/bofire",
    keywords=[
        "Bayesian optimization",
        "Multi-objective optimization",
        "Experimental design",
    ],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=get_version(),
    python_requires=">=3.9",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "pydantic>=1.10.0,<2.0",
        "scipy>=1.7",
    ],
    extras_require={
        "optimization": [
            "torch>=1.12",
            "botorch>=0.8.4",
            "multiprocess",
            "plotly",
            "formulaic>=0.6.0",
            "cloudpickle>=2.0.0",
            sklearn_dependency,
        ],
        "cheminfo": ["rdkit", sklearn_dependency],
        "tests": [
            "mock",
            "mopti",
            "pyright==1.1.305",
            "pytest",
            "pytest-cov",
            "papermill",
            "jupyter",
            "matplotlib",
        ],
        "docs": [
            "mkdocs",
            "mkdocs-material",
            "mkdocs-jupyter",
            "mkdocstrings>=0.18",
            "mkdocstrings-python-legacy",
            "mike",
        ],
    },
)
