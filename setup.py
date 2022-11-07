import os.path

from setuptools import find_packages, setup


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

with open(os.path.join(root_dir, "requirements.txt"), "r") as f:
    stripped_lines = (line.strip() for line in f)
    install_requires = [
        line for line in stripped_lines if not line.startswith("#") and len(line) > 0
    ]

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
    python_requires=">=3.7",
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    extras_require={"testing": ["pytest", "mopti"]},
)
