"""Installable package definition for CHG-Net."""

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent.resolve()
README = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="chgnet",
    version="0.1.0",
    description="CHG-Net: direction-aware heterogeneous graph trajectory prediction (PyTorch)",
    long_description=README,
    long_description_content_type="text/markdown",
    author="CHG-Net contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*", "scripts"]),
    install_requires=[
        "torch>=2.0.0,<3.0.0",
        "numpy>=1.24.0,<3.0.0",
        "pandas>=2.0.0,<3.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "PyYAML>=6.0,<7.0",
        "tqdm>=4.65.0,<5.0.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0,<9.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
