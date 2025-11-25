"""
Setup script for mood detection system
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mood-detection-system",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Multimodal mood detection system using face and voice",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mood-detection-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "opencv-python>=4.8.0",
        "librosa>=0.10.0",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "soundfile>=0.12.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "mood-detect=main:main",
        ],
    },
)