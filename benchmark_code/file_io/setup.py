from setuptools import setup

with open("README.md","r") as fh:
    long_description=fh.read()

setup(
    name="file_io",
    version="0.0.1",
    author="guoren",
    description="A package for mass spectrometry date file process",
    classfiers=[
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    py_modules=[
        "file_io.mgf_file",
        "file_io.msp_file",
        "file_io.mzml_file",
        "file_io.spec_file"
        ]
)