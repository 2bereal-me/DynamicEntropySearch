from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch
import numpy as np
from pathlib import Path

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

lib_path=Path('data/dynamic_lib_1')


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

# Add spectra into the library
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library, clean=True)
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library, clean=True)

# Call build_index() and write() lastly
entropy_search.build_index()
entropy_search.write()

# Perform search
query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}

result=entropy_search.search_topn_matches(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, 
        ms2_tolerance_in_da=0.02, 
        method='open', 
        precursor_ions_removal_da=1.6, 
        noise_threshold=0.01, 
        min_ms2_difference_in_da=0.05, 
        max_peak_num=None, 
        clean=True, 
        topn=3, 
        need_metadata=True, 
)
print(result)