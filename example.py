#!/usr/bin/env python3
from ms_entropy import read_one_spectrum
from dynamic_entropy_search.repository_search import RepositorySearch


def main():
    search_engine = RepositorySearch(path_data="data/repository_search_test")
    file_mzml_1 = "data/balf_pos-267_pos_C18.mzML"
    file_mzml_2 = "data/plasma_pos-SE-011_pos_C18.mzML"
    file_msp = "data/example.msp"

    build_index(search_engine, file_msp, iter_spectra_reader=read_spectrum_from_msp_file)
    build_index(search_engine, file_mzml_1)
    build_index(search_engine, file_mzml_2)

    spec = {
        "charge": 1,
        "peaks": [[58.0646, 1894], [86.095, 98105]],
        "precursor_mz": 183.987125828,
    }
    search_result = search_spectrum(
        search_engine,
        charge=spec["charge"],
        precursor_mz=spec["precursor_mz"],
        peaks=spec["peaks"],
        method="hybrid",
    )

    # Get 1st spectrum data
    spec_data = get_spectrum_data(search_engine, spec["charge"], search_result[0].pop("spec_idx"))
    spec_data.update(search_result[0])
    print(f"Top match spectrum data: {spec_data}")
    pass


def search_spectrum(
    search_engine: RepositorySearch,
    charge,
    precursor_mz,
    peaks,
    method="open",
    ms1_tolerance_in_da=0.01,
    ms2_tolerance_in_da=0.02,
):
    search_result = search_engine.search_topn_matches(
        charge=charge,
        precursor_mz=precursor_mz,
        peaks=peaks,
        method=method,
        ms1_tolerance_in_da=ms1_tolerance_in_da,
        ms2_tolerance_in_da=ms2_tolerance_in_da,
        output_full_spectrum=False,
    )
    return search_result


def get_spectrum_data(search_engine: RepositorySearch, charge, spec_idx):
    spec = search_engine.get_spectrum(charge, spec_idx)
    spec.pop("scan", None)
    return spec


def build_index(search_engine: RepositorySearch, file_ms, iter_spectra_reader=None):
    search_engine.add_ms_file(file_ms, iter_spectra_reader)
    search_engine.build_index()
    search_engine.write()


def read_spectrum_from_msp_file(filename):
    for spec in read_one_spectrum(filename):
        spec["charge"] = int(spec["charge"])
        spec["precursor_mz"] = float(spec["precursormz"])
        yield spec


if __name__ == "__main__":
    main()
