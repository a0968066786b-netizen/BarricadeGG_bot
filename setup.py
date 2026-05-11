"""
Setup configuration for Quoridor AI Training
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if fh else ""

setup(
    name="quoridor-ai",
    version="1.0.0",
    author="BarricadeGG Bot Team",
    description="Quoridor AI Agent training using Stable Baselines3 and Gymnasium",
    long_description=long_description if long_description else "Quoridor AI training framework",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.27.0",
        "sb3-contrib>=2.0.0",
        "numpy>=1.21.0",
        "tensorboard>=2.10.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
        ],
        "gpu": [
            "torch>=1.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "quoridor-train=scripts.train:main",
            "quoridor-evaluate=scripts.evaluate:main",
        ],
    },
)
