from setuptools import setup, find_packages

VERSION = "0.1"

setup(
    name="openlabel-datasets",
    version=VERSION,
    packages=find_packages(),
    install_requires=[],
    author="Alan Garcia",
    author_email="agarciaj@vicomtech.org",
    description="A package for creating BEVDatasets from multiple sources",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.com/agarciaj1/bev2seg_2",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)