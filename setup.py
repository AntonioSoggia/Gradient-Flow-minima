#!/usr/bin/env python3
"""
Setup script for Gradient Flow package.
"""

from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="gradient_flow",
    version="0.1.0",
    description="Gradient flow for finding closed-form minima in noisy line images",
    author="Antonio",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="gradient-descent optimization symbolic-computation minima",
)
