from pathlib import Path
import pickle
import sys
import time
import os
import msgpack
import numpy as np
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'library'))
from ms_entropy import clean_spectrum
from dynamic_entropy_search.repository_search import RepositorySearch

start_time=time.time()
charge=sys.argv[1]
step=sys.argv[-1]
path_data=Path.cwd().parent.parent/f"dynamic_repository/10_entropy_search"
path_data.mkdir(parents=True, exist_ok=True)
repository_entropy_search=RepositorySearch(path_data=path_data)

if step=="hybrid_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    repository_entropy_search.search_topn_matches(
        method='hybrid',
        charge=int(charge),
        precursor_mz=spectra["precursor_mz"], 
        peaks=np.array(spectra["peaks"]), 
        ms1_tolerance_in_da=0.01,
        ms2_tolerance_in_da=0.02,
        output_full_spectrum=False)

elif step=="open_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    repository_entropy_search.search_topn_matches(
        method='open',
        charge=int(charge),
        precursor_mz=spectra["precursor_mz"], 
        peaks=np.array(spectra["peaks"]), 
        ms1_tolerance_in_da=0.01,
        ms2_tolerance_in_da=0.02,
        output_full_spectrum=False
        )
elif step=="neutral_loss_search":
    query_spec_pkl=sys.argv[-2]
    spectra=pickle.loads(open(query_spec_pkl, "rb").read())
    spectra['peaks']=clean_spectrum(spectra['peaks'])
    repository_entropy_search.search_topn_matches(
        method='neutral_loss',
        charge=int(charge),
        precursor_mz=spectra["precursor_mz"], 
        peaks=np.array(spectra["peaks"]), 
        ms1_tolerance_in_da=0.01,
        ms2_tolerance_in_da=0.02,
        output_full_spectrum=False)

elapsed_time=time.time()-start_time
print(elapsed_time)
    