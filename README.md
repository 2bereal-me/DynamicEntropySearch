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

## For python users


### Usage of library construction (combining initializing and updating process)

#### In brief
```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Add spectra into the library
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library)
......
# Call build_index() and write() lastly
entropy_search.build_index()
entropy_search.write()
```
#### In details
Suppose you have a lot of spectral libraries, you need to format it like this:

```python
import numpy as np

spectra_1_for_library = [{
    "id": "Demo spectrum 1",
    "precursor_mz": 150.0,
    "peaks": [[100.0, 1.0], [101.0, 1.0], [103.0, 1.0]]
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
    "peaks": [[100.0, 1.0], [101.0, 1.0], [302.0, 1.0]]}]

spectra_2_for_library ...
spectra_3_for_library ...
```
Note that the `precursor_mz` and `peaks` keys are required, the reset of the keys are optional.

Then you can have your spectra libraries to be added into the library.

```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Add spectra into the library
entropy_search.add_new_spectra(spectra_list=spectra_1_for_library)
entropy_search.add_new_spectra(spectra_list=spectra_2_for_library)
......

entropy_search.build_index()
entropy_search.write()
```
It is necessary to initialize `DynamicEntropySearch` using a specified `path_data`, which is the path of your library. The reset of the parameters are optional.

If you only want to build index for open search, you can set `index_for_neutral_loss` in `add_new_spectra()` to `False`.

It is necessary to call `build_index()` and `write()` lastly after all `add_new_spectra()`.

### Usage of search

#### In brief

Suppose you have established a library under `path_of_your_library` using the aforementioned method.

Now you can perform search with a query spectrum in correct format like this:

```python
import numpy as np

query_spectrum = {"precursor_mz": 150.0,
                  "peaks": np.array([[100.0, 1.0], [101.0, 1.0], [102.0, 1.0]], dtype=np.float32)}
```

You can call the `DynamicEntropySearch` class with corresponding `path_data` to search the library like this:

```python
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

# Assign the path for your library
entropy_search=DynamicEntropySearch(path_data=path_of_your_library)

# Search the library
entropy_similarity=entropy_search.search(
        precursor_mz=query_spectrum['precursor_mz'],
        peaks=query_spectrum['peaks'],
)
```

After that, you can print the results like this:

```python
print(entropy_similarity)
```

### Usage of RepositorySearch

RepositorySearch offers prebuilt indexes for public metabolomics repositories, comprising more than 1.4 billion spectra. As a part of DynamicEntropySearch, users can use RepositorySearch to search against these public metabolomics repositories.

Suppose you have downloaded the prebuilt indexes from (https://huggingface.co/datasets/YuanyueLiZJU/dynamic_entropy_search/tree/main) and extracted them to `path_repository_indexes` on your local machine.

Firstly, assign the path of the prebuilt indexes as the `path_data` of `RepositorySearch` class.

```python
from dynamic_entropy_search.repository_search import RepositorySearch

search_engine = RepositorySearch(path_data=path_repository_indexes)
```

Prepare query spectrum in correct format.

```python
import numpy as np
query_spec={
        "charge": 1,
        "peaks": np.array([[58.0646, 1894], [86.095, 98105]], dtype=np.float32),
        "precursor_mz": 183.987125828,
    }
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
    # you can specify the spectrum you want to extract from results by setting spec_idx
    spec = search_engine.get_spectrum(charge, spec_idx)
    spec.pop("scan", None)
    return spec

spec_data = get_spectrum_data(search_engine, query_spec["charge"], search_result[0].pop("spec_idx"))
spec_data.update(search_result[0])
print(f"Top match spectrum data: {spec_data}")

spec_data = get_spectrum_data(search_engine, query_spec["charge"], search_result[1].pop("spec_idx"))
spec_data.update(search_result[1])
print(f"Top match spectrum data: {spec_data}")

```