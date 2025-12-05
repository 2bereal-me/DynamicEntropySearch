from ms_entropy import FlashEntropySearch
import pickle
import sys
import time
import numpy as np
from pathlib import Path

start_time=time.time()
charge=sys.argv[1]
reference_pkl=Path(sys.argv[2])
step=sys.argv[-1]
if step=="hybrid_search":
    
    query_spec_pkl=Path(sys.argv[-2])

path_data=Path.cwd().parent.parent/"comparison_data"
path_data.mkdir(parents=True, exist_ok=True)

reference_spectra=pickle.loads(open(reference_pkl, "rb").read())

path_data_flash=path_data/f"flash/charge-{charge}"
path_data_flash.mkdir(parents=True, exist_ok=True)

flash_entropy_search=FlashEntropySearch(path_data=path_data_flash, low_memory=2)

if step=="build":
    flash_entropy_search.build_index(reference_spectra)
    flash_entropy_search.write()

elif step=="hybrid_search":
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    flash_entropy_search.read()
    flash_entropy_search.hybrid_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elapsed_time=time.time()-start_time
print(elapsed_time)
