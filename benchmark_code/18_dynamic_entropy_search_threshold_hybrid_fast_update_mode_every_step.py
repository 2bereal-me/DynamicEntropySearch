from pathlib import Path
import pickle
import sys
import time
import os
import numpy as np
import msgpack
import struct
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch
from ms_entropy import clean_spectrum


def batch_load(path_spec, i, cache_list_threshold, dynamic_entropy_search):
    with open(path_spec/f"s_{cache_list_threshold}_batch_{i}.bin", "rb") as f:
        for i in range(1_000_000//cache_list_threshold):
            size_bytes=f.read(8)
            size = struct.unpack('Q', size_bytes)[0]
            data=f.read(size)
            spec=pickle.loads(data)
            dynamic_entropy_search.add_new_spectra(spec)
            
    return


start_time=time.time()
charge=sys.argv[1]
num_per_group=int(sys.argv[2])
cache_list_threshold=int(sys.argv[3])
step=sys.argv[-1]

path_dynamic_data=Path.cwd().parent.parent/f"comparison_data/dynamic/charge-{charge}"
path_dynamic_data.mkdir(parents=True, exist_ok=True)

path_spec=Path.cwd().parent.parent/f"spec_data/35_random_export_ms2/charge_{charge}"
dynamic_entropy_search=DynamicEntropySearch(path_data=path_dynamic_data, num_per_group=num_per_group, cache_list_threshold=cache_list_threshold)

if step=='build':
    if cache_list_threshold<1_000_000:
        path_spec=Path.cwd().parent.parent/f"spec_data/s_35_random_export_ms2/charge_{charge}"

        for i in range(100):
            batch_load(path_spec=path_spec, i=i, cache_list_threshold=cache_list_threshold, dynamic_entropy_search=dynamic_entropy_search)
        dynamic_entropy_search.build_index()
        dynamic_entropy_search.write()

    else:
        for i in range(100):
            with open(path_spec/f"batch_{i}.bin", "rb") as f:
                spec=msgpack.load(f)
        
            dynamic_entropy_search.add_new_spectra(spec)
            
        dynamic_entropy_search.build_index()
        dynamic_entropy_search.write()

elif step=="open_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    dynamic_entropy_search.open_search(peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elif step=="neutral_loss_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    dynamic_entropy_search.neutral_loss_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elif step=="hybrid_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    dynamic_entropy_search.hybrid_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)


elapsed_time=time.time()-start_time
print(elapsed_time)
    