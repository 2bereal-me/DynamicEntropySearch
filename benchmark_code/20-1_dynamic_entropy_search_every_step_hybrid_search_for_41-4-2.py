from pathlib import Path
import pickle
import sys
import time
import os
import msgpack
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'library'))
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

start_time=time.time()
charge=int(sys.argv[1])
step=sys.argv[-1]
num_per_group=int(sys.argv[2])
cache_list_threshold=int(sys.argv[3])
if step=="hybrid_search":
    
    query_spec_pkl=sys.argv[-2]


path_data=Path.cwd().parent.parent/f"comparison_data/dynamic/charge-{charge}"
path_data.mkdir(parents=True, exist_ok=True)

path_spec=Path.cwd().parent.parent/f"spec_data/35_random_export_ms2/charge_{charge}"
dynamic_entropy_search=DynamicEntropySearch(path_data=path_data, num_per_group=num_per_group, cache_list_threshold=cache_list_threshold)

if step=="build":

    for i in range(100):
        with open(path_spec/f"batch_{i}.bin", "rb") as f:
            spec=msgpack.load(f)
    
        dynamic_entropy_search.add_new_spectra(spectra_list=spec, convert_to_flash=False)
    
    dynamic_entropy_search.build_index(convert_to_flash=False)
    dynamic_entropy_search.write()


elif step=="hybrid_search":
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    
    dynamic_entropy_search.hybrid_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elapsed_time=time.time()-start_time
print(elapsed_time)
    