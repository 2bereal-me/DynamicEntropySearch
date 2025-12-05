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
charge=sys.argv[1]
num_per_group=int(sys.argv[2])
cache_list_threshold=int(sys.argv[3])
step=sys.argv[-1]
path_data=Path.cwd().parent.parent/f"comparison_data/dynamic/charge-{charge}"
path_data.mkdir(parents=True, exist_ok=True)
dynamic_entropy_search=DynamicEntropySearch(path_data=path_data, num_per_group=num_per_group, cache_list_threshold=cache_list_threshold)
if step=="update":
    spectra_bin=Path(sys.argv[-2])
    if str(spectra_bin)[-4:]==".bin":
        with open(spectra_bin, "rb") as f:
            spec=msgpack.load(f)
    elif str(spectra_bin)[-4:]==".pkl":
        spec=pickle.loads(open(spectra_bin, "rb").read())
        
    dynamic_entropy_search.add_new_spectra(spec, convert_to_flash=False)
    dynamic_entropy_search.build_index(convert_to_flash=False)
    dynamic_entropy_search.write()
elif step=="build":
    file_num=len(sys.argv)-5

    for i in range(file_num):
        file=Path(sys.argv[4+i])
        if str(file)[-4:]==".pkl":
            spec=pickle.loads(open(Path(file), "rb").read())

        elif str(file)[-4:]==".bin":
            with open(file, "rb") as f:
                spec=msgpack.load(f)

        dynamic_entropy_search.add_new_spectra(spec, convert_to_flash=False)
    
    dynamic_entropy_search.build_index(convert_to_flash=False)
    dynamic_entropy_search.write()
elif step=="hybrid_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    dynamic_entropy_search.hybrid_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)
elif step=="open_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    dynamic_entropy_search.open_search(peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)
elif step=="neutral_loss_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    dynamic_entropy_search.neutral_loss_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elapsed_time=time.time()-start_time
print(elapsed_time)
    