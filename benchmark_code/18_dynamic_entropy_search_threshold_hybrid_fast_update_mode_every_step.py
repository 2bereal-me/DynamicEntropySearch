from pathlib import Path
import pickle
import sys
import time
import os
import msgpack
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'library'))
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

start_time=time.time()
charge=sys.argv[1]
num_per_group=int(sys.argv[2])
cache_list_threshold=int(sys.argv[3])

path_dynamic_data=Path.cwd().parent.parent/f"comparison_data/dynamic/charge-{charge}"
path_dynamic_data.mkdir(parents=True, exist_ok=True)

path_spec=Path.cwd().parent.parent/f"spec_data/35_random_export_ms2/charge_{charge}"
dynamic_entropy_search=DynamicEntropySearch(path_data=path_dynamic_data, num_per_group=num_per_group, cache_list_threshold=cache_list_threshold)

for i in range(100):
    with open(path_spec/f"batch_{i}.bin", "rb") as f:
        spec=msgpack.load(f)
   
    dynamic_entropy_search.add_new_spectra(spec)
    
dynamic_entropy_search.build_index()
dynamic_entropy_search.write()

elapsed_time=time.time()-start_time
print(elapsed_time)
    