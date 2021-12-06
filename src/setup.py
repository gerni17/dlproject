#!/usr/bin/env python

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="segmentation",
    version="0.0",
    description="First Network for segmentation",
    install_requires=requirements,
    packages=find_packages(),
)
