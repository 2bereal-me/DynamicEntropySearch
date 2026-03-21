import numpy as np


def read_one_spectrum(filename_input, ms2_only=False):
    import pymzml

    """
    Read information from .mzml file.
    :param filename_input: a .mzML file.
    :return: a dict contains a list with key 'spectra'.
        The list contains multiple dict, one dict represent a single spectrum's informaiton.
    """
    run = pymzml.run.Reader(filename_input, obo_version="4.1.33")
    for n, spec_raw in enumerate(run):
        if ms2_only and spec_raw.ms_level != 2:
            continue
        spec = np.asarray(spec_raw.peaks("raw"), dtype=np.float32, order="C")
        spec_info = {
            "ms_level": spec_raw.ms_level,
            "_scan_number": n + 1,
            "file": str(filename_input).split("/")[-1],
            "peaks": spec,
            "rt": spec_raw.scan_time_in_minutes() * 60,
            "precursor_mz": spec_raw.selected_precursors[0].get("mz", None) if len(spec_raw.selected_precursors) > 0 else None,
            "charge": spec_raw.selected_precursors[0].get("charge", None) if len(spec_raw.selected_precursors) > 0 else None,
            "precursor_type": (
                spec_raw.selected_precursors[0].get("precursor_type", None) if len(spec_raw.selected_precursors) > 0 else None
            ),
            "name": spec_raw.index,
        }
        if spec_info["charge"] is None:
            if spec_raw["negative scan"]:
                spec_info["charge"] = -1
            elif spec_raw["positive scan"]:
                spec_info["charge"] = 1
        else:
            try:
                spec_info["charge"] = int(spec_info["charge"])
                if spec_raw["negative scan"]:
                    spec_info["charge"] = -abs(spec_info["charge"])
                elif spec_raw["positive scan"]:
                    spec_info["charge"] = abs(spec_info["charge"])
            except ValueError:
                print("Warning: precursor charge is not an integer: {}".format(spec_info["charge"]))

        yield spec_info
