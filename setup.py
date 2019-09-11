import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ifsFractals",
    version="0.0.1",
    author="Peter Francis",
    author_email="franpe02@gettysburg.edu",
    description="For Generating Fast IFS Fractals",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/francisp336/IFSFGl",
    py_modules=["ifsFractals"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.7.3',
    install_requires=[
        'imageio>=2.5.0',
        'matplotlib>=3.1.1',
        'numba>=0.45.1',
        'numpy>=1.17.0',
        'Pillow>=6.1.0',
        'scipy>=1.3.1',
        'system>=0.1.16',
        'termcolor>=1.1.0',
        'typing>=3.7.4'],

        # 'imageio', 'matplotlib', 'numba', 'numpy', 'Pillow', 'scipy', 'system', 'termcolor', 'typing'],
    keywords='sample setuptools development'
)
