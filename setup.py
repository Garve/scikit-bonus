"""Setup file."""

import setuptools

import skbonus

base_packages = [
    "scikit-learn>=0.24.0",
    "pandas>=1.2.1",
    "numpy>=1.18.5",
    "scipy>=1.5.0",
]

test_packages = [
    "flake8>=3.8.4",
    "pytest>=6.2.2",
    "black>=20.8b1",
    "pre-commit>=2.10.0",
]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scikit-bonus",
    version=skbonus.__version__,
    author="Robert Kübler",
    author_email="xgarve@gmail.com",
    description="Extending scikit-learn with various useful things.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Garve/scikit-bonus",
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=base_packages,
    extras_require={"test": test_packages},
)
