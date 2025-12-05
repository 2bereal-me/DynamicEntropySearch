from ms_entropy import FlashEntropySearch
import pickle
import sys
import time
from pathlib import Path
import msgpack
import numpy as np

start_time=time.time()
charge=sys.argv[1]
path_data=Path.cwd().parent.parent/"comparison_data"
path_data_flash=path_data/f"flash/charge-{charge}"
path_data_flash.mkdir(parents=True, exist_ok=True)

flash_entropy_search=FlashEntropySearch(path_data=path_data_flash, low_memory=2)

if sys.argv[-1]=="update":
    spectra_bin=Path(sys.argv[-2])
    if str(spectra_bin)[-4:] ==".bin":
        with open(spectra_bin, "rb") as f:
            insert_spec=msgpack.load(f)
    elif str(spectra_bin)[-4:] ==".pkl":    
        insert_spec=pickle.loads(open(spectra_bin, "rb").read())
    spectra_init=sys.argv[2:-2]
    all_spec=[]
    all_spec.extend(insert_spec)
    for spectra in spectra_init:
        if str(spectra)[-4:]==".pkl":
            spec_init=pickle.loads(open(spectra, "rb").read())
        elif str(spectra)[-4:]==".bin":
            with open(spectra, "rb") as f:
                spec_init=msgpack.load(f)
        all_spec.extend(spec_init)
    flash_entropy_search.build_index(all_spec)
    flash_entropy_search.write()

elif sys.argv[-1]=="build":
    file_num=len(sys.argv)-3
    all_spec=[]
    for i in range(file_num):
        file=Path(sys.argv[2+i])
        if str(file)[-4:]==".pkl":
            spec=pickle.loads(open(file, "rb").read())

        elif str(file)[-4:]==".bin":
            with open(file, "rb") as f:
                spec=msgpack.load(f)
        all_spec.extend(spec)

    flash_entropy_search.build_index(all_spec)
    flash_entropy_search.write()

elif sys.argv[-1]=="hybrid_search":
    query_spec_pkl=Path(sys.argv[-2])
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    flash_entropy_search.read()
    flash_entropy_search.hybrid_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)
elif sys.argv[-1]=="open_search":
    query_spec_pkl=Path(sys.argv[-2])
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    flash_entropy_search.read()
    flash_entropy_search.open_search(peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)
elif sys.argv[-1]=="neutral_loss_search":
    query_spec_pkl=Path(sys.argv[-2])
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    flash_entropy_search.read()
    flash_entropy_search.neutral_loss_search(precursor_mz=spectra["precursor_mz"], peaks=np.array(spectra["peaks"]), ms2_tolerance_in_da=0.02)

elapsed_time=time.time()-start_time
print(elapsed_time)
