from setuptools import setup, find_packages

setup(
    name="UWDiffusion",
    version="0.0.1",
    description="A Python package for running diffusion simulations using Underworld3.",
#     long_description=open("README.md").read(),
#     long_description_content_type="text/markdown",
    author="bknight1",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "underworld3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)