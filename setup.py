from setuptools import setup
from os import path

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="ifsFractals",
    packages=["ifsFractals"],
    version="1.0.2",
    author="Peter Francis",
    author_email="franpe02@gettysburg.edu",
    description="For Generating Fast IFS Fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PeterEFrancis/ifsFractals",
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
