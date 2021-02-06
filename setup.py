import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scikit-bonus",
    version="0.0.4.1",
    author="Robert Kübler",
    author_email="xgarve@gmail.com",
    description="Extending scikit-learn with useful things",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Garve/scikit-bonus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
