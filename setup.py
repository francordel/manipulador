#!/usr/bin/env python3
"""
Setup script for Manipulador: An Agentic Red-Teaming Framework for LLM Vulnerability Discovery
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="manipulador",
    version="1.0.0",
    author="Anonymous Authors",
    author_email="anonymous@example.com",
    description="An Agentic Red-Teaming Framework for LLM Vulnerability Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/Manipulador-OSS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
        ],
        "api": [
            "openai>=1.0.0",
            "anthropic>=0.7.0",
            "together>=0.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "manipulador-eval=scripts.run_real_measured_evaluation:main",
            "manipulador-eval-2step=scripts.run_real_2step_evaluation:main",
            "manipulador-comprehensive=scripts.run_comprehensive_evaluation_legacy:main",
            "manipulador-clean=scripts.run_clean_evaluation:main",
            "manipulador-analyze=scripts.analyze_results:main",
        ],
    },
    include_package_data=True,
    package_data={
        "manipulador": [
            "configs/**/*.yaml",
            "configs/**/*.json", 
            "data/**/*.csv",
            "data/**/*.json",
        ],
    },
    keywords="red-teaming, llm, security, ai-safety, vulnerability-assessment, adversarial-attacks",
    project_urls={
        "Bug Reports": "https://github.com/your-org/Manipulador-OSS/issues",
        "Source": "https://github.com/your-org/Manipulador-OSS",
        "Documentation": "https://github.com/your-org/Manipulador-OSS/blob/main/docs/",
        "Paper": "https://arxiv.org/abs/2025.XXXXX",
    },
)