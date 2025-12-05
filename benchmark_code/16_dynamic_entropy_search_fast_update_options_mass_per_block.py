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
spectra_bin=sys.argv[4]
step=sys.argv[-1]
mass_per_block=float(sys.argv[-2])
     
path_data_dynamic=Path.cwd().parent.parent/f"comparison_data/dynamic/charge-{charge}"
path_data_dynamic.mkdir(parents=True, exist_ok=True)

with open(spectra_bin, "rb") as f:
    spectra=msgpack.load(f)

dynamic_entropy_search=DynamicEntropySearch(path_data=path_data_dynamic, num_per_group=num_per_group, cache_list_threshold=cache_list_threshold, mass_per_block=mass_per_block)


dynamic_entropy_search.add_new_spectra(spectra, convert_to_flash=False)
dynamic_entropy_search.build_index(convert_to_flash=False)
dynamic_entropy_search.write()

    
elapsed_time=time.time()-start_time
print(elapsed_time)
    