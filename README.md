Dataset for benchmark can be found at the following link:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18035369.svg)](https://doi.org/10.5281/zenodo.18035369)


# Theoretical Background

`Spectral entropy` is a useful property to measure the complexity of a spectrum. It is inspired by the concept of Shannon entropy in information theory. [(ref)](https://doi.org/10.1038/s41592-021-01331-z)

`Entropy similarity`, which measured spectral similarity based on spectral entropy, has been shown to outperform dot product similarity in compound identification. [(ref)](https://doi.org/10.1038/s41592-021-01331-z)

The calculation of entropy similarity can be accelerated by using the `Flash Entropy Search` algorithm. [(ref)](https://doi.org/10.1038/s41592-023-02012-9)

`Dynamic Entropy Search` is built and optimized based on `Flash Entropy Search`. Besides the excellent search performance, it allows unlimited library spectra with high speed and low memory.

![DynamicEntropySearch Flow](Figure_1.png)

# How to use this package

This repository contains the source code to build index, update index, calculate spectral entropy and entropy similarity in python.


## Usage of library construction (combining initializing and updating process)


### Step 1: prepare the spectral libraries

Suppose you have a lot of spectra and want to build library based on them, you need to format them like this:

```python
import numpy as np
# For each spectral library, it is a list consisting of multiple dictionaries of MS2 spectra.

# For each spectrum, 'precursor_mz' and 'peaks' are necessary. 
# 'precursor_mz' should be a float, and 'peaks' should be a 2D np.ndarray like np.ndarray([[m/z, intensity], [m/z, intensity], [m/z, intensity]...], dtype=np.float32).


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

spectra_2_for_library ... # Similar to spectra_1_for_library
spectra_3_for_library ... # Similar to spectra_1_for_library
```
Note that the `precursor_mz` and `peaks` keys are **required**, the reset of the keys are optional.

The spectra in the spectra library should be **cleaned** using `clean_spectrum()` in `ms_entropy` before passed into the `add_new_spectrum()`.

```python
from ms_entropy import clean_spectrum

precursor_ions_removal_da = 1.6

for spec in spectra_1_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da, # Max m/z in peaks.
        min_mz: float = -1.0, # Min m/z in peaks.
        noise_threshold: float = 0.01, # The minimum intensity to keep. Defaults to 0.01, which will remove peaks with intensity < 0.01 * max_intensity.
        min_ms2_difference_in_da: float = 0.05, # The minimum m/z difference between two peaks in the resulting spectrum.
        min_ms2_difference_in_ppm: float = -1.0, # The minimum m/z difference between two peaks in the resulting spectrum. Defaults to -1, which will use the min_ms2_difference_in_da instead.
        max_peak_num: int = -1, # The maximum number of peaks to keep.
        normalize_intensity: bool = True, # Whether to normalize the intensity to sum to 1.

    )

for spec in spectra_2_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da
    )   # Other parameters can be set as aforementioned.

for spec in spectra_3_for_library:
    spec['peaks'] = clean_spectrum(
        peaks = spec['peaks'],
        max_mz = spec['precursor_mz'] - precursor_ions_removal_da
    ) # Other parameters can be set as aforementioned.
```
- Note that three parameters  
(1) `min_ms2_difference_in_da` in `clean_spectrum()`  
(2) `max_ms2_tolerance_in_da` in the initialization of class `DynamicEntropySearch()`  
(3) `ms2_tolerance_in_da` in search functions of `DynamicEntropySearch()`  
should follow this rule: `min_ms2_difference_in_da` > `max_ms2_tolerance_in_da` * 2 >= `ms2_tolerance_in_da` * 2.  
An error will be reported if the condition is not met.


Then you can have your spectra lists to be added into the library. 

### Step 2: perform update

#### Initial construction
Suppose that you want to construct an index with spectra_1_for_library at first:
```python
# Firstly, import DynamicEntropySearch.
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Secondly, assign the path for your library.
entropy_search=DynamicEntropySearch(
        path_data=path_of_your_library, 
        max_ms2_tolerance_in_da=0.024, # Maximum MS/MS tolerance (in Daltons) used during spectrum search.
        extend_fold=3, # Expansion factor for preallocated storage in each m/z block. Determines ``reserved_len = data_len * extend_fold``. 
        mass_per_block=0.05, # m/z step size for creating the index blocks.
        num_per_group=100_000_000, # Number of spectra assigned to each group. 
        cache_list_threshold=1_000_000, # Number of spectra to accumulate in memory before writing them to disk.
        max_indexed_mz=1500.00005, # Maximum m/z value to index. Ions above this threshold are grouped into a single block. 
        intensity_weight="entropy",  # "entropy" or None.Determines whether intensities are entropy-weighted. 
)

# Thirdly, add spectra list into the library one by one.
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library)

# Lastly, call build_index() and write() to end the adding operation.
entropy_search.build_index()
entropy_search.write()
```
There are some tips in this process:  

- It is necessary to initialize `DynamicEntropySearch` using a specified `path_data`, which is the path of your library. The reset of the parameters are optional. If it is a new library, the value of `path_data` should be new and will be created in the initialization of class.

- If you only want to build index for open search, you can set `index_for_neutral_loss` in `add_new_spectra()` and `build_index()` to `False`. However, after doing this, you couldn't perform neutral loss search or hybrid search under this library anymore. Besides, adding neutral loss mass index into this library is violated too. This means that once the `index_for_neutral_loss` in the `add_new_spectra()` function as well as `build_index()` function are set to `False`, they should remain `False` from then on. Any violation can cause errors.

- It is necessary to call `build_index()` and `write()` lastly after all `add_new_spectra()` as the end of adding operation to make sure all the spectra are loaded into the index.

Once these steps are complete, you will find a folder, which serves as the library, at the `path_data`. In this folder, several binary files and one or more subfolders can be found. These binary files record the information of subfolders and metadata. Each subfolder refers to a group — the organizational unit directly under a library. These subfolders are numerically named, starting from 0. 

Example structure — one library containing 3 groups:

```
path_of_your_library/
├── 0/
├── 1/
├── 2/
├── group_start.pkl
├── metadata_start_loc.bin
└── metadata.pkl
```
The library with index of spectra_1_for_library is created. The spectra_1_for_library is now saved as index in this library in group 0. You can fetch the library whenever you want. 

#### Subsequent construction
Next time, if you want to update the index with spectra_2_for_library and spectra_3_for_library, just select the correct path and execute the update:

```python
# Firstly, import DynamicEntropySearch.
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Secondly, choose the existing library with corresponding path. This library is built with spectra_1_for_library last time.
entropy_search=DynamicEntropySearch(
        path_data=path_of_your_library, 
        max_ms2_tolerance_in_da=0.024, # Maximum MS/MS tolerance (in Daltons) used during spectrum search.
        extend_fold=3, # Expansion factor for preallocated storage in each m/z block. Determines ``reserved_len = data_len * extend_fold``. 
        mass_per_block=0.05, # m/z step size for creating the index blocks.
        num_per_group=100_000_000, # Number of spectra assigned to each group. 
        cache_list_threshold=1_000_000, # Number of spectra to accumulate in memory before writing them to disk.
        max_indexed_mz=1500.00005, # Maximum m/z value to index. Ions above this threshold are grouped into a single block. 
        intensity_weight="entropy",  # "entropy" or None.Determines whether intensities are entropy-weighted. 
)

# Thirdly, add spectra list into the library one by one.
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_3_for_library)

# Lastly, call build_index() and write() to end the adding operation.
entropy_search.build_index()
entropy_search.write()
```

Now the library in `path_of_your_library` contains index of 3 spectra lists: spectra_1_for_library, spectra_2_for_library and spectra_3_for_library. They may be distributed in one or more groups, depending on the number of spectra and the value of `num_per_group`.


## Usage of search

You can perform identity search, open search, neutral loss search or hybrid search based on your need.

### Search with internal clean function

Suppose you have established a library locally under `path_of_your_library` using the aforementioned method.

Now you can perform search with a query spectrum in correct format like this:

```python
import numpy as np
# For each query spectrum, 'precursor_mz' and 'peaks' are necessary. 
# 'precursor_mz' should be a float, and 'peaks' should be a 2D np.ndarray like np.ndarray([[m/z, intensity], [m/z, intensity], [m/z, intensity]...], dtype=np.float32).

query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}
```

If your query spectra is a list consisting of several spectra:

```python
import numpy as np

# For each query_spectra_list, it is a list consisting of multiple dictionaries of query MS2 spectra.

# For each query spectrum, 'precursor_mz' and 'peaks' are necessary. 
# 'precursor_mz' should be a float, and 'peaks' should be a 2D np.ndarray like np.ndarray([[m/z, intensity], [m/z, intensity], [m/z, intensity]...], dtype=np.float32).

query_spectra_list = [{
                "precursor_mz": 150.0,
                "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)
                },{
                "precursor_mz": 250.0,
                "peaks": np.array([[108.0, 1.0], [113.0, 1.0], [157.0, 1.0]], dtype=np.float32)
                },{
                "precursor_mz": 299.0,
                "peaks": np.array([[119.0, 1.0], [145.0, 1.0], [157.0, 1.0]], dtype=np.float32)
                },
                ]
```

You can call the `DynamicEntropySearch` class with corresponding `path_data` to search the library like this:

```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Select the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Search the library and you can fetch the metadata from the results with the highest scores
result=entropy_search.search_topn_matches(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
        method='open', # Or 'neutral_loss' or 'hybrid' or 'identity'.
        precursor_ions_removal_da=1.6, # Peaks with m/z greater than ``precursor_mz - precursor_ions_removal_da`` are removed during cleaning. 
        noise_threshold=0.01, # Relative intensity threshold for noise filtering during cleaning. Peaks with intensity ``< noise_threshold * max(intensity)`` are removed.  
        min_ms2_difference_in_da=0.05, # Minimum spacing allowed between MS/MS peaks during cleaning.  
        max_peak_num=None, # Maximum number of peaks to keep after cleaning.  
        clean=True, # If you don't want to use the internal clean process in this function, set it to False.
        topn=3, # You can change topn as needed.
        need_metadata=True, # Set it to True if need metadata.
)

# After that, you can print the result like this:
print(result)
```
Notes in search:
- Cleaning the query spectrum is necessary. You can use the internal clean function of `search_topn_matches()` or seperate the clean and search process. This is introduced in the following part.

- `search_topn_matches()` is suitable for identification that requires metadata when `need_metadata` in it is `True`. If it is set to `False`, the location of matched spectra in library will be returned. 
 

An example result:

```
[{
'id': 'Demo spectrum 3', 
'precursor_mz': 250.0, 
'peaks': array([[101.        ,   0.33333334], [200.        ,   0.33333334], [202.        ,   0.33333334]], dtype=float32), 
'XXX': 'YYY', 
'open_search_entropy_similarity': np.float32(0.99999994)
}, {
'id': 'Demo spectrum 2', 
'precursor_mz': 200.0, 
'peaks': array([[100.        ,   0.33333334], [101.        ,   0.33333334], [102.        ,   0.33333334]], dtype=float32), 
'metadata': 'ABC', 
'open_search_entropy_similarity': np.float32(0.3333333)
}, {
'precursor_mz': 350.0, 
'peaks': array([[100.        ,   0.33333334], [101.        ,   0.33333334], [302.        ,   0.33333334]], dtype=float32), 
'open_search_entropy_similarity': np.float32(0.3333333)}]
```

In this result:

- This is generated from searching the query spectrum against an existing library. Select correct `method` to perform search.

- 3 top matched spectra are given in descending order of similarity. This is set by `topn` in `search_topn_matches()`. If number of spectra with similarity greater than 0 is less than `topn`, then output the actual matching number of spectra.

- Metadata of spectra is given if `need_metadata` in `search_topn_matches()` is set to `True`. Users can add information for spectra when constructing library. These additional information other than 'precursor_mz' and 'peaks', like 'id', benefit the compound identification.


If the query spectra is a list, iterate it to perform search.

```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# For query_spectra_list, iterate it to perform search for each elements.
for spec in query_spectra_list:
    result=entropy_search.search_topn_matches(
            precursor_mz=spec['precursor_mz'],
            peaks=spec['peaks'],
            ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
            method='open', # or 'neutral_loss' or 'hybrid' or 'identity'.
            clean=True, # If you don't want to use the internal clean process in this function, set it to False.
            topn=3, # You can change topn as needed.
            need_metadata=True, # Set it to True if need metadata.
    )
    # After that, you can print the result like this:
    print(result)
```
### Search with external clean function

If you want to seperate clean and search process, you can set `clean` in `search_topn_matches()` to `False` and use an external clean function.

You can use the `clean_spectrum()` function in `ms_entropy` to clean the query spectrum and then use individual search functions to search the library.

```python
from ms_entropy import clean_spectrum

query_spectrum = {"precursor_mz": 150.0,
                    "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}

precursor_ions_removal_da = 1.6

query_spectrum['peaks'] = clean_spectrum(
    peaks = query_spectrum['peaks'],
    max_mz = query_spectrum['precursor_mz'] - precursor_ions_removal_da
)
```
Now the query_spectrum is cleaned and ready for search. Then pass it to the `search_topn_matches()` with `clean` set to `False`.

```python

result=entropy_search.search_topn_matches(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
        method='open', # Or 'neutral_loss' or 'hybrid' or 'identity'.
        precursor_ions_removal_da=1.6, # Peaks with m/z greater than ``precursor_mz - precursor_ions_removal_da`` are removed during cleaning. 
        noise_threshold=0.01, # Relative intensity threshold for noise filtering during cleaning. Peaks with intensity ``< noise_threshold * max(intensity)`` are removed.  
        min_ms2_difference_in_da=0.05, # Minimum spacing allowed between MS/MS peaks during cleaning.  
        max_peak_num=None, # Maximum number of peaks to keep after cleaning.  
        clean=False, # If you don't want to use the internal clean process in this function, set it to False.
        topn=3, # You can change topn as needed.
        need_metadata=True, # Set it to True if need metadata.
)
```

You can also pass the query spectrum into the search functions mentioned in **Multiple search options** like this:

```python
# Identity search
entropy_similarity = entropy_search.identity_search(
    precursor_mz = query_spectrum['precursor_mz'],
    peaks = query_spectrum['peaks'],
    ms1_tolerance_in_da = 0.01,
    ms2_tolerance_in_da = 0.02
)
```

### Multiple search options

Different search functions can serve different objectives.

#### Prepare before searching
For all search functions, select path and clean the query spectrum is necessary:

##### Select library path:
```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)
```
##### Clean query spectrum:
See **Search with external clean function** for details.

#### Search with different functions

Besides `search_topn_matches()`, You can also perform search using other functions:

##### Only need entropy similarity without metadata in multiple search methods

```python
### Use `search()` and get an array with all entropy similarities ###
result=entropy_search.search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
        method='all', # or 'neutral_loss' or 'hybrid' or 'identity' or 'open'.
        clean=True, # If you don't want to use the internal clean process in this function, set it to False.
)
print(result)
```
Example result:

```
{
'identity_search': array([0.        , 0.        , 0.99999994, 0.        , 0.        , 0.        ], dtype=float32), 
'open_search': array([0.3333333 , 0.3333333 , 0.99999994, 0.3333333 , 0.        , 0.        ], dtype=float32), 
'neutral_loss_search': array([0.3333333 , 0.        , 0.99999994, 0.3333333 , 0.        , 0.        ], dtype=float32), 
'hybrid_search': array([0.6666666 , 0.3333333 , 0.99999994, 0.6666666 , 0.        , 0.        ], dtype=float32)}

```
This result:
- includes the results of all search methods because `method` is set to `all`. 
- returns only similarity array in the order of spectra in the library.

##### Identity search with similarity:

```python
### Use `identity_search()` and get an array with all entropy similarities based on identity search ###
result=entropy_search.identity_search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
)
print(result)
```

Example result:

```
[0.         0.         0.99999994 0.         0.         0.        ]
```
This result:
- includes the results of identity search. 
- returns only similarity array in the order of spectra in the library.


##### Open search with similarity:

```python
### Use `open_search()` and get an array with all entropy similarities based on open search ###
result=entropy_search.open_search(
        peaks=query_spectrum['peaks'],
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
)
print(result)
```
Example result:
```
[0.3333333  0.3333333  0.99999994 0.3333333  0.         0.        ]
```
This result:
- includes the results of open search. 
- returns only similarity array in the order of spectra in the library.


##### Neutral loss search with similarity:

```python
### Use `neutral_loss_search()` and get an array with all entropy similarities based on neutral loss search ###
result=entropy_search.neutral_loss_search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
)
print(result)
```

Example result:
```
[0.3333333  0.         0.99999994 0.3333333  0.         0.        ]
```
This result:
- includes the results of neutral loss search. 
- returns only similarity array in the order of spectra in the library.


##### Hybrid search with similarity:

```python
### Use `hybrid_search()` and get an array with all entropy similarities based on hybrid search ###
result=entropy_search.hybrid_search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
        ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
)
print(result)
```

Example result:

```
[0.6666666  0.3333333  0.99999994 0.6666666  0.         0.        ]
```
This result:
- includes the results of hybrid search. 
- returns only similarity array in the order of spectra in the library.


## Usage of RepositorySearch

RepositorySearch offers prebuilt indexes for public metabolomics repositories, comprising more than 1.4 billion spectra. As a part of DynamicEntropySearch, users can use RepositorySearch to search against these public metabolomics repositories. We have built the indexes and upload them to (https://huggingface.co/datasets/YuanyueLiZJU/dynamic_entropy_search/tree/main).

Suppose you have downloaded the prebuilt indexes from (https://huggingface.co/datasets/YuanyueLiZJU/dynamic_entropy_search/tree/main) and extracted them to `path_repository_indexes` on your local machine, you can perform search like this:

Firstly, assign the path of the prebuilt indexes as the `path_data` of `RepositorySearch` class.

Remember to prepare query spectrum in correct format and clean it (see aforementioned points to prepare the format). Key `charge` here is necessary. Set it to 1 or -1.

Then perform search, and you can get top few results.

```python
from dynamic_entropy_search.repository_search import RepositorySearch
import numpy as np
from ms_entropy import clean_spectrum

# Instantiation
entropy_search=RepositorySearch(path_data=path_repository_indexes)

# Prepare query spectra
precursor_ions_removal_da=1.6

query_spec_1={
    "peaks":np.array([[217.0, 1.5], [234.0, 0.8], [398.0, 2.0]]),
    "precursor_mz":455.0,
    "charge":-1                
}

query_spec_2={
    "peaks":np.array([[123.0, 1.0], [126.0, 0.7], [101.0, 4.0]]),
    "precursor_mz":250.0,
    "charge":1                
}

query_spec_3={
    "peaks":np.array([[200.0, 0.9], [101.0, 3.2], [202.0, 1.7]]),
    "precursor_mz":345.0,
    "charge":-1                
}
query_spectra=[query_spec_1, query_spec_2, query_spec_3]

# Clean query spectra
for query_spec in query_spectra:
    query_spec['peaks']=clean_spectrum(
            peaks=query_spec['peaks'],
            max_mz = query_spec['precursor_mz'] - precursor_ions_removal_da
        )
    
# Perform search and output results
for i, query_spec in enumerate(query_spectra):
    result=entropy_search.search_topn_matches(
        method="open", 
        charge=query_spec['charge'],
        precursor_mz=query_spec['precursor_mz'],
        peaks=query_spec['peaks'],
        topn=3 # can be changed
        )

    print(f"Query spectrum {i} matches:{result}\n")
```

An example result:

```
Query spectrum 0 matches:[{'file_name': 'gnps/MSV000080555/A9_RA9_01_8358.mzML.gz', 'scan': np.uint64(1074), 'similarity': np.float64(0.758648693561554), 'spec_idx': np.uint64(187632338)}, {'file_name': 'gnps/MSV000094528/20230131_pluskal_mce_1D2_H4_id_negative.mzML.gz', 'scan': np.uint64(319), 'similarity': np.float64(0.6575593948364258), 'spec_idx': np.uint64(231674808)}, {'file_name': 'metabolights/MTBLS700/ns154.mzML.gz', 'scan': np.uint64(1664), 'similarity': np.float64(0.6054909229278564), 'spec_idx': np.uint64(261307506)}]

Query spectrum 1 matches:[{'file_name': 'gnps/MSV000079098/Stairs_Gz140_57H3_RH3_01_651.mzML.gz', 'scan': np.uint64(780), 'similarity': np.float64(0.8724607229232788), 'spec_idx': np.uint64(168727792)}, {'file_name': 'gnps/MSV000079098/Roff_SEA12_64F2_GF2_01_721.mzML.gz', 'scan': np.uint64(2860), 'similarity': np.float64(0.8721256852149963), 'spec_idx': np.uint64(249982900)}, {'file_name': 'gnps/MSV000080141/A2154D_34E05_RE5_01_25781.mzML.gz', 'scan': np.uint64(487), 'similarity': np.float64(0.8716126680374146), 'spec_idx': np.uint64(171140687)}]

Query spectrum 2 matches:[{'file_name': 'metabolomics_workbench/ST003745/x01997_NEG.mzML.gz', 'scan': np.uint64(1139), 'similarity': np.float64(0.7167922258377075), 'spec_idx': np.uint64(259794461)}, {'file_name': 'gnps/MSV000090773/B2_EC_P_NEG_QC_Cond_MSMS_AutoMSMS_1.mzML.gz', 'scan': np.uint64(421), 'similarity': np.float64(0.6405000686645508), 'spec_idx': np.uint64(325131566)}, {'file_name': 'gnps/MSV000090773/B2_EC_P_NEG_QC_Cond_MSMS_AutoMSMS_1.mzML.gz', 'scan': np.uint64(423), 'similarity': np.float64(0.6405000686645508), 'spec_idx': np.uint64(325131568)}]

```
In this result:
- Every query spectrum gets 3 matched reference spectra because `topn` in `search_topn_matches()` is set to `3`. This value can be changed based on your need.
- Every result contains `file_name`, `similarity`, `scan` that can be used in further identification.

Another example can be found in the root directory of **DynamicEntropySearch** project and can serve as an instruction.  
By using `RepositorySearch`, you can easily search spectra against global spectra library.