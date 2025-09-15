
from setuptools import find_packages, setup

setup(
    name="trending_deploy",
    description="Deploy Trending Models on Hugging Face Inference Endpoints with CPU hardware",
    version="0.1.0",
    url="https://github.com/huggingface/trending-deploy",
    project_urls={
        "Source Code": "https://github.com/huggingface/trending-deploy",
        "Issue Tracker": "https://github.com/huggingface/trending-deploy/issues",
    },
    license="Apache License, Version 2.0",
    maintainer="Hugging Face Team",
    author="Hugging Face Team",
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "huggingface_hub",
    ],
    packages=find_packages(),
)