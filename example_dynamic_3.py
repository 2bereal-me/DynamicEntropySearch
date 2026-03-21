from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch
import numpy as np
from pathlib import Path
from ms_entropy import clean_spectrum

spectra_1_for_library = [{
    "id": "Demo spectrum 1",
    "precursor_mz": 150.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]], dtype=np.float32), 
}, {
    "id": "Demo spectrum 2",
    "precursor_mz": 200.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32),
    "metadata": "ABC"
}, {
    "id": "Demo spectrum 3",
    "precursor_mz": 250.0,
    "peaks": np.array([[200.0, 1.0], [101.0, 1.0], [202.0, 1.0]], dtype=np.float32),
    "XXX": "YYY",
}, {
    "precursor_mz": 350.0,
    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [302.0, 1.0]], dtype=np.float32),
},
    ]

spectra_2_for_library = [{
    "id": "Demo spectrum 5",
    "precursor_mz": 234.452,
    "peaks": np.array([[55.0, 217.0], [192.345, 81.0], [198.47, 21.0], [203.0, 66.064]], dtype=np.float32), 
}, {
    "id": "Demo spectrum 6",
    "precursor_mz": 1298.45409,
    "peaks": np.array([[77.46, 31.8], [516.0, 8.0], [1022.0, 47.5536], [313.67845, 742.706], [1299.45409, 1.0], [313.67745, 71.0]], dtype=np.float32),
    "metadata": "plasma"
}]

spectra_3_for_library = [{
    "precursor_mz": 423.4,
    "peaks": np.array([[88.28, 132.0], [102.9, 28.8], [17.753, 51.02]], dtype=np.float32), 
    "metadata": "sample_1"
}, {
    "id": "Demo spectrum 8",
    "precursor_mz": 987.123,
    "peaks": np.array([[2.0079, 0.324], [4.12, 0.324]], dtype=np.float32),
    "metadata": "sample_2"
}]

lib_path=Path('data/dynamic_lib_3')


###################################
### Under test condition, the directory is deleted each time and then reference index is newly created under the path from scratch.
### If use this script to build a reference index for long-term use or want to update the reference index, DO NOT DELETE the directory.

if lib_path.exists():
    import shutil
    shutil.rmtree(lib_path)

###################################


# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=lib_path, 
                                    max_ms2_tolerance_in_da=0.024, 
                                    extend_fold=3,
                                    mass_per_block=0.05,
                                    num_per_group=100_000_000,
                                    cache_list_threshold=1_000_000,
                                    max_indexed_mz=1500.00005,
                                    intensity_weight="entropy", )

# Add spectra into the library using external clean function
precursor_ions_removal_da = 1.6

# spectra_1_for_library
spectra_1_for_library_clean=[]
for spec in spectra_1_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da)
    if len(spec['peaks']) >0:
        spectra_1_for_library_clean.append(spec)

entropy_search.add_new_spectra(spectra_list=spectra_1_for_library_clean, clean=False)

# spectra_2_for_library
spectra_2_for_library_clean=[]
for spec in spectra_2_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da)
    if len(spec['peaks']) >0:
        spectra_2_for_library_clean.append(spec)

entropy_search.add_new_spectra(spectra_list=spectra_2_for_library_clean, clean=False)

# spectra_3_for_library
spectra_3_for_library_clean=[]
for spec in spectra_3_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da)
    if len(spec['peaks']) >0:
        spectra_3_for_library_clean.append(spec)

entropy_search.add_new_spectra(spectra_list=spectra_3_for_library_clean, clean=False)

# Call build_index() and write() lastly
entropy_search.build_index()
entropy_search.write()

# Perform search
query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}


# Use external clean function
precursor_ions_removal_da = 1.6

query_spectrum['peaks'] = clean_spectrum(
    peaks = query_spectrum['peaks'],
    max_mz = query_spectrum['precursor_mz'] - precursor_ions_removal_da
)

result=entropy_search.search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, 
        ms2_tolerance_in_da=0.02, 
        method='all', 
        clean=False, 
)
print(result)