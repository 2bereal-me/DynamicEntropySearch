Datasets for benchmark can be found at the following links:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17835213.svg)](https://doi.org/10.5281/zenodo.17835213)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844649.svg)](https://doi.org/10.5281/zenodo.17844649)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844588.svg)](https://doi.org/10.5281/zenodo.17844588)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844626.svg)](https://doi.org/10.5281/zenodo.17844626)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844642.svg)](https://doi.org/10.5281/zenodo.17844642)

# Theoretical Background

`Spectral entropy` is a useful property to measure the complexity of a spectrum. It is inspired by the concept of Shannon entropy in information theory. [(ref)](https://doi.org/10.1038/s41592-021-01331-z)

`Entropy similarity`, which measured spectral similarity based on spectral entropy, has been shown to outperform dot product similarity in compound identification. [(ref)](https://doi.org/10.1038/s41592-021-01331-z)

The calculation of entropy similarity can be accelerated by using the `Flash Entropy Search` algorithm. [(ref)](https://doi.org/10.1038/s41592-023-02012-9)

`Dynamic Entropy Search` is built and optimized based on `Flash Entropy Search`. Besides the excellent search performance, it allows unlimited library spectra with high speed and low memory.


# How to use this package

This repository contains the source code to build index, update index, calculate spectral entropy and entropy similarity in python.


## Usage of library construction (combining initializing and updating process)

You can establish your own library locally as follows:

### In brief
```python
# Step 1: Import DynamicEntropySearch.
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Step 2: Assign the path for your library.
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Step 3: Add spectra into the library. This adding operation can be performed multiple times.
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library)
......

# Step 4: Call build_index() and write() lastly to end adding operation.
entropy_search.build_index()
entropy_search.write()
```

### In details

#### Step 1: prepare the spectral libraries

Suppose you have a lot of spectral libraries, you need to format them like this:

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
Note that the `precursor_mz` and `peaks` keys are required, the reset of the keys are optional.

Then you can have your spectra libraries to be added into the library.

#### Step 2: perform update
```python
# Firstly, import DynamicEntropySearch.
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Secondly, assign the path for your library.
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Thirdly, add spectra into the library one by one.
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_3_for_library)

# Lastly, call build_index() and write() to end the adding operation.
entropy_search.build_index()
entropy_search.write()
```
It is necessary to initialize `DynamicEntropySearch` using a specified `path_data`, which is the path of your library. The reset of the parameters are optional.

If you only want to build index for open search, you can set `index_for_neutral_loss` in `add_new_spectra()` and `build_index()` to `False`.

It is necessary to call `build_index()` and `write()` lastly after all `add_new_spectra()` as the end of adding operation.

## Usage of search

You can perform identity search, open search, neutral loss search or hybrid search based on your need.

### In brief

Suppose you have established a library locally under `path_of_your_library` using the aforementioned method.

Now you can perform search with a query spectrum in correct format like this:

```python
import numpy as np
# For each query spectrum, 'precursor_mz' and 'peaks' are necessary. 
# 'precursor_mz' should be a float, and 'peaks' should be a 2D np.ndarray like np.ndarray([[m/z, intensity], [m/z, intensity], [m/z, intensity]...], dtype=np.float32).

query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}
```

If your query spectra is a list consisting of several spectrum:

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

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Search the library and you can fetch the metadata from the results with the highest scores
result=entropy_search.search_topn_matches(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
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

### Multiple search options

Besides `search_topn_matches()`, You can also perform search using other functions:

```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# For query_spectra_list, iterate it to perform search for each elements.
```
For example:
```python
### Use `search()` and get an array with all entropy similarities ###
for spec in query_spectra_list:
    result=entropy_search.search(
            precursor_mz=spec['precursor_mz'],
            peaks=spec['peaks'],
            ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
            method='open', # or 'neutral_loss' or 'hybrid' or 'identity' or 'all'.
            clean=True, # If you don't want to use the internal clean process in this function, set it to False.
    )
    print(result)


### Use `identity_search()` and get an array with all entropy similarities based on identity search ###
for spec in query_spectra_list:
    result=entropy_search.identity_search(
            precursor_mz=spec['precursor_mz'],
            peaks=spec['peaks'],
            ms1_tolerance_in_da=0.01, # You can change ms1_tolerance_in_da as needed.
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
    )
    print(result)


### Use `open_search()` and get an array with all entropy similarities based on open search ###
for spec in query_spectra_list:
    result=entropy_search.open_search(
            peaks=spec['peaks'],
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
    )
    print(result)


### Use `neutral_loss_search()` and get an array with all entropy similarities based on neutral loss search ###
for spec in query_spectra_list:
    result=entropy_search.neutral_loss_search(
            precursor_mz=spec['precursor_mz'],
            peaks=spec['peaks'],
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
    )
    print(result)


### Use `hybrid_search()` and get an array with all entropy similarities based on hybrid search ###
for spec in query_spectra_list:
    result=entropy_search.hybrid_search(
            precursor_mz=spec['precursor_mz'],
            peaks=spec['peaks'],
            ms2_tolerance_in_da=0.02, # You can change ms2_tolerance_in_da as needed.
    )
    print(result)
```



## Usage of RepositorySearch

RepositorySearch offers prebuilt indexes for public metabolomics repositories, comprising more than 1.4 billion spectra. As a part of DynamicEntropySearch, users can use RepositorySearch to search against these public metabolomics repositories. We have built the indexes and upload them to (https://huggingface.co/datasets/YuanyueLiZJU/dynamic_entropy_search/tree/main).

Suppose you have downloaded the prebuilt indexes from (https://huggingface.co/datasets/YuanyueLiZJU/dynamic_entropy_search/tree/main) and extracted them to `path_repository_indexes` on your local machine, you can perform search like this:

Firstly, assign the path of the prebuilt indexes as the `path_data` of `RepositorySearch` class.

```python
from dynamic_entropy_search.repository_search import RepositorySearch

search_engine = RepositorySearch(path_data=path_repository_indexes)
```

Prepare query spectrum in correct format (see aforementioned points to prepare the format).

```python
import numpy as np
query_spec={
        "charge": 1,
        "peaks": np.array([[58.0646, 1894], [86.095, 98105]], dtype=np.float32),
        "precursor_mz": 183.987125828,
    }
# Or a list:
query_spec = [{
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
Then perform search, and you can get top few results.

```python
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

# Perform search
search_result = search_spectrum(
    search_engine,
    charge=query_spec["charge"],
    precursor_mz=query_spec["precursor_mz"],
    peaks=query_spec["peaks"],
    method="hybrid", # or 'open' or 'neutral_loss' or 'identity'
)
```

If you want to extract the results:

```python
def get_spectrum_data(search_engine: RepositorySearch, charge, spec_idx):
    # You can specify the spectrum you want to extract from results by setting spec_idx
    spec = search_engine.get_spectrum(charge, spec_idx)
    spec.pop("scan", None)
    return spec
# For example, set `spec_idx` to 0.
spec_data = get_spectrum_data(search_engine, query_spec["charge"], search_result[0].pop("spec_idx"))
spec_data.update(search_result[0])
print(f"Top match spectrum data: {spec_data}")

# For example, set `spec_idx` to 1.
spec_data = get_spectrum_data(search_engine, query_spec["charge"], search_result[1].pop("spec_idx"))
spec_data.update(search_result[1])
print(f"Match spectrum data: {spec_data}")

```