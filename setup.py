from setuptools import setup
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="ifsFractals",
    version="0.0.13",
    author="Peter Francis",
    author_email="franpe02@gettysburg.edu",
    description="For Generating Fast IFS Fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/francisp336/IFSFGl",
    py_modules=["ifsFractals"],
    classifiers=[],
    python_requires='>=3.0.0',
    install_requires=[
        'imageio',
        'matplotlib',
        'numba==0.45.1',
        'numpy',
        'Pillow',
        'scipy',
        'system',
        'termcolor',
        'typing'
    ],
    keywords='Fractal Generator'
)
