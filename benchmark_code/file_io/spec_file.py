import pickle
from pathlib import Path
import numpy as np
import zipfile
from typing import Union
from . import msp_file, mgf_file, mzml_file

def read_all_spectra(filename_input:str, file_type:object=None):
    if file_type is None:
        file_type=guess_file_type_from_file_name(filename_input)


def guess_file_type_from_file_name(filename_input:str):
    file_type = None
    if filename_input[-4:].lower() == ".zip":
        fzip_all = zipfile.ZipFile(filename_input)
        fzip_list = zipfile.ZipFile.namelist(fzip_all)
        # Only select the first file for the zip file
        if fzip_list[0][-4:].lower() == ".msp":
            file_type = ".msp"
        elif fzip_list[0][-4:].lower() == ".mgf":
            file_type = ".mgf"
    elif filename_input[-3:].lower() == ".gz":
        if filename_input[-8:-3].lower() == ".mzml":
            file_type = ".mzml"
    else:
        if filename_input[-4:].lower() == ".msp":
            file_type = ".msp"
        elif filename_input[-4:].lower() == ".mgf":
            file_type = ".mgf"
        elif filename_input[-5:].lower() == ".mzml":
            file_type = ".mzml"
        elif filename_input.split(".")[-2].lower() == "msp":
            file_type = ".msp"
        elif filename_input.split(".")[-2].lower() == "mgf":
            file_type = ".mgf"
        elif filename_input.split(".")[-2].lower() == "mzml":
            file_type = ".mzml"
    return file_type

def read_one_spectrum(
        filename_input: Union[str, Path],
        file_type: object = None, 
        ms2_only=False, 
        **kwargs: object):
    """

    :param filename_input:
    :param file_type:
    :param ms2_only: Only output MS/MS spectra
    :param kwargs:
    :return: a dictionary contains the following items:
        ms_level: 1, 2
        peaks: list or numpy array, For mzML file, it is a numpy array.
        precursor_mz: float or None
        _scan_number: start from 1, int
        rt: retention time, float or None. For mzML file, it is in seconds. For msp/mgf formats, it is the same expression as in the original file.
    """
    if isinstance(filename_input, Path):
        filename_input = str(filename_input)

    if file_type is None:
        file_type = guess_file_type_from_file_name(filename_input)

    if file_type == ".msp":
        spectral_generator = msp_file.read_one_spectrum(filename_input=filename_input, **kwargs)
    elif file_type == ".mgf":
        spectral_generator = mgf_file.read_one_spectrum(filename_input=filename_input, **kwargs)
    elif file_type == ".mzml":
        spectral_generator = mzml_file.read_one_spectrum(filename_input=filename_input, **kwargs)

    else:
        raise NotImplementedError()

    for i, spec in enumerate(spectral_generator):
        i += 1
        spec_template = {
            '_scan_number': i,
            "_ms_level": 2,
        }


        spec_template.update(spec)
        spec = spec_template

        if ms2_only and spec["_ms_level"] != 2:
            continue

        if len(spec["peaks"]) == 0:
            spec["peaks"] = np.empty((0, 2), dtype=np.float32)
        else:
            spec["peaks"] = np.asarray(spec["peaks"], dtype=np.float32)

        yield spec
