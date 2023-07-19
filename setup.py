from setuptools import setup
from os import path

setup(
    name="ifsFractals",
    packages=["ifsFractals"],
    version="1.0.4",
    author="Peter Francis",
    author_email="peter.e.francis@stonybrook.edu",
    description="For Fast Generation of IFS Fractals",
    long_description="""
    # ifsFractals

    A python module for fast Iterated Function System Fractal generation

    See the [GitHub Repo](https://github.com/PeterEFrancis/ifsFractals-py) for more information.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/PeterEFrancis/ifsFractals-py",
    classifiers=[],
    python_requires='>=3.0.0',
    install_requires=[
        'imageio',
        'matplotlib',
        'numpy',
        'Pillow',
        'IPython'
    ],
    keywords='Fractal Generator'
)
