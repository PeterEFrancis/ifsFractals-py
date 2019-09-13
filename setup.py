import setuptools
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="ifsFractals",
    version="0.0.6",
    author="Peter Francis",
    author_email="franpe02@gettysburg.edu",
    description="For Generating Fast IFS Fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/francisp336/IFSFGl",
    py_modules=["ifsFractals"],
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # python_requires='>=3.0.0',
    install_requires=[
        'imageio',
        'matplotlib',
        'numba',
        'numpy',
        'Pillow',
        'scipy',
        'system',
        'termcolor',
        'typing'
    ],
    keywords='Fractal Generator'
)
