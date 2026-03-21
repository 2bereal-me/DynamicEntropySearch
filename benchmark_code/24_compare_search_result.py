from pathlib import Path
import pickle
import sys
import time
import os
import msgpack
import numpy as np
from ms_entropy import FlashEntropySearch
from ms_entropy import clean_spectrum

# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'library'))
from dynamic_entropy_search.dynamic_entropy_search import DynamicEntropySearch

def _get_max_diff(
    search_result,
    flash_result
):
    
    max_diff=np.max(np.abs(search_result-flash_result))

    return max_diff

def _get_top_idx(
    search_result,
):
    sort_idx=np.argsort(search_result)
    top_idx=[sort_idx[-1]]
    max_score=search_result[top_idx]
    for idx in sort_idx[::-1][1:]:
        if search_result[idx]==max_score:
            top_idx.append(idx)
        elif search_result[idx]<max_score:
            break

    return top_idx

def _compare_top_hits(
    search_top_idx,
    flash_top_idx
):
    if search_top_idx.sort()==flash_top_idx.sort():
        search_top_hits=True
    else:
        search_top_hits=False

    return search_top_hits

def calculate_max_diff_and_top_hits(
    flash_result,
    dynamic_all_build_result,
    dynamic_build_update_1_result,
    dynamic_build_update_2_result
):
    # max_diff
    dynamic_all_build_max_diff=_get_max_diff(search_result=dynamic_all_build_result, flash_result=flash_result)
    dynamic_build_update_1_max_diff=_get_max_diff(search_result=dynamic_build_update_1_result, flash_result=flash_result)
    dynamic_build_update_2_max_diff=_get_max_diff(search_result=dynamic_build_update_2_result, flash_result=flash_result)

    # top hits
    flash_top_idx=_get_top_idx(search_result=flash_result)

    dynamic_all_build_top_idx=_get_top_idx(search_result=dynamic_all_build_result)
    dynamic_all_build_top_hits=_compare_top_hits(search_top_idx=dynamic_all_build_top_idx, flash_top_idx=flash_top_idx)

    dynamic_build_update_1_top_idx=_get_top_idx(search_result=dynamic_build_update_1_result)
    dynamic_build_update_1_top_hits=_compare_top_hits(search_top_idx=dynamic_build_update_1_top_idx, flash_top_idx=flash_top_idx)

    dynamic_build_update_2_top_idx=_get_top_idx(search_result=dynamic_build_update_2_result)
    dynamic_build_update_2_top_hits=_compare_top_hits(search_top_idx=dynamic_build_update_2_top_idx, flash_top_idx=flash_top_idx)


    return [dynamic_all_build_max_diff, dynamic_build_update_1_max_diff, dynamic_build_update_2_max_diff, 
            dynamic_all_build_top_hits, dynamic_build_update_1_top_hits, dynamic_build_update_2_top_hits]

def _organize_result_by_id(
    spectra_list,
    all_spec,
    flash_all_build_result,
    dynamic_all_build_result_,
    dynamic_build_update_1_result_,
    dynamic_build_update_2_result_
):
        
    flash_result=np.zeros((flash_all_build_result.shape))
    for id,ref_spec in enumerate(spectra_list):
        loc=ref_spec['ID']
        flash_result[loc]=flash_all_build_result[id]

    dynamic_all_build_result=np.zeros((dynamic_all_build_result_.shape))
    dynamic_build_update_1_result=np.zeros((dynamic_build_update_1_result_.shape))
    dynamic_build_update_2_result=np.zeros((dynamic_build_update_2_result_.shape))

    for id_,ref_spec_ in enumerate(all_spec):
        loc=ref_spec_['ID']
        dynamic_all_build_result[loc]=dynamic_all_build_result_[id_]
        dynamic_build_update_1_result[loc]=dynamic_build_update_1_result_[id_]
        dynamic_build_update_2_result[loc]=dynamic_build_update_2_result_[id_]
    
    return flash_result, dynamic_all_build_result, dynamic_build_update_1_result, dynamic_build_update_2_result

def _perform_search(
        search_type,
        spec,
        flash_all_build,
        dynamic_all_build,
        dynamic_build_update_1,
        dynamic_build_update_2
):
    peaks=clean_spectrum(peaks=spec['peaks'])

    if search_type=='open':
        flash_all_build_result=flash_all_build.open_search(peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_all_build_result_=dynamic_all_build.open_search(peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_1_result_=dynamic_build_update_1.open_search(peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_2_result_=dynamic_build_update_2.open_search(peaks=peaks,ms2_tolerance_in_da=0.02)

    elif search_type=='neutral_loss':
        flash_all_build_result=flash_all_build.neutral_loss_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_all_build_result_=dynamic_all_build.neutral_loss_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_1_result_=dynamic_build_update_1.neutral_loss_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_2_result_=dynamic_build_update_2.neutral_loss_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
    
    elif search_type=='hybrid':
        flash_all_build_result=flash_all_build.hybrid_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_all_build_result_=dynamic_all_build.hybrid_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_1_result_=dynamic_build_update_1.hybrid_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)
        dynamic_build_update_2_result_=dynamic_build_update_2.hybrid_search(precursor_mz=spec['precursor_mz'], peaks=peaks,ms2_tolerance_in_da=0.02)

    return flash_all_build_result, dynamic_all_build_result_, dynamic_build_update_1_result_, dynamic_build_update_2_result_

def _output_results(
    i,
    search_type,
    dynamic_all_build_max_diff,
    dynamic_build_update_1_max_diff,
    dynamic_build_update_2_max_diff,
    dynamic_all_build_top_hits,
    dynamic_build_update_1_top_hits,
    dynamic_build_update_2_top_hits
):
    print('query_scan:', i)
    print(f'{search_type}_search_dynamic_all_build_max_diff:', dynamic_all_build_max_diff)
    print(f'{search_type}_search_dynamic_build_update_1_max_diff:', dynamic_build_update_1_max_diff)
    print(f'{search_type}_search_dynamic_build_update_2_max_diff:', dynamic_build_update_2_max_diff)
    print(f'{search_type}_search_dynamic_all_build_top_hits:',dynamic_all_build_top_hits)
    print(f'{search_type}_search_dynamic_build_update_1_top_hits:', dynamic_build_update_1_top_hits)
    print(f'{search_type}_search_dynamic_build_update_2_top_hits:', dynamic_build_update_2_top_hits)
    return



charge=sys.argv[1]
query_pkl=sys.argv[-1]
path_spec_data=Path.cwd().parent.parent/"spec_data"

query_spectra_list=pickle.loads(open(query_pkl, "rb").read())
# Select a file with 1M spectra randomly
random_i=np.random.choice(100, 1)
print('reference_file_id:', random_i[0])
spec_file=path_spec_data/f'35_random_export_ms2/charge_{charge}/batch_{random_i[0]}.bin'

with open(spec_file, "rb") as f:
    all_spec=msgpack.load(f)

for idx, spec in enumerate(all_spec):
    spec['ID']=idx

# Assign build part and update part
# For Flash
flash_spec=all_spec
path_data=Path.cwd().parent.parent/f"comparison_data/flash_all_build/charge-{charge}"
flash_all_build=FlashEntropySearch(path_data=path_data, low_memory=2)
spectra_list=flash_all_build.build_index(all_spectra_list=all_spec)
flash_all_build.write()


# For Dynamic all build
np.random.shuffle(all_spec)
dynamic_all_build_spec=all_spec
path_data=Path.cwd().parent.parent/f"comparison_data/dynamic_all_build/charge-{charge}"
dynamic_all_build=DynamicEntropySearch(path_data=path_data)
dynamic_all_build.add_new_spectra(spectra_list=all_spec)
dynamic_all_build.build_index()
dynamic_all_build.write()

# For Dynamic build0.1_update0.9*1
build_part=all_spec[:100_000]
update_part=all_spec[100_000:]
path_data=Path.cwd().parent.parent/f"comparison_data/dynamic_build_update_1/charge-{charge}"
dynamic_build_update_1=DynamicEntropySearch(path_data=path_data)
dynamic_build_update_1.add_new_spectra(spectra_list=build_part)
dynamic_build_update_1.add_new_spectra(spectra_list=update_part)
dynamic_build_update_1.build_index()
dynamic_build_update_1.write()


# For Dynamic build0.1_update0.09*10
build_part=all_spec[:100_000]
path_data=Path.cwd().parent.parent/f"comparison_data/dynamic_build_update_2/charge-{charge}"
dynamic_build_update_2=DynamicEntropySearch(path_data=path_data)
dynamic_build_update_2.add_new_spectra(spectra_list=build_part)
start=100_000
for i in range(10):
    end=start+(i+1)*90_000
    update_part=all_spec[start:end]
    dynamic_build_update_2.add_new_spectra(spectra_list=update_part)
    start=end

dynamic_build_update_2.build_index()
dynamic_build_update_2.write()

# Search 
for i, spec in enumerate(query_spectra_list):
        
    # open search
    search_type='open'
    [flash_all_build_result, 
        dynamic_all_build_result_, 
        dynamic_build_update_1_result_, 
        dynamic_build_update_2_result_]=_perform_search(search_type=search_type,
                                                    spec=spec,
                                                    flash_all_build=flash_all_build,
                                                    dynamic_all_build=dynamic_all_build,
                                                    dynamic_build_update_1=dynamic_build_update_1,
                                                    dynamic_build_update_2=dynamic_build_update_2)

    [flash_result, 
        dynamic_all_build_result, 
        dynamic_build_update_1_result, 
        dynamic_build_update_2_result]=_organize_result_by_id(
        spectra_list=spectra_list,
        all_spec=all_spec,
        flash_all_build_result=flash_all_build_result,
        dynamic_all_build_result_=dynamic_all_build_result_,
        dynamic_build_update_1_result_=dynamic_build_update_1_result_,
        dynamic_build_update_2_result_=dynamic_build_update_2_result_
        )
    

    [dynamic_all_build_max_diff, dynamic_build_update_1_max_diff, 
        dynamic_build_update_2_max_diff, dynamic_all_build_top_hits, 
        dynamic_build_update_1_top_hits, dynamic_build_update_2_top_hits]=calculate_max_diff_and_top_hits(
            flash_result=flash_result,
            dynamic_all_build_result=dynamic_all_build_result,
            dynamic_build_update_1_result=dynamic_build_update_1_result,
            dynamic_build_update_2_result=dynamic_build_update_2_result)
    
    _output_results(i=i,
                    search_type=search_type,
                    dynamic_all_build_max_diff=dynamic_all_build_max_diff,
                    dynamic_build_update_1_max_diff=dynamic_build_update_1_max_diff,
                    dynamic_build_update_2_max_diff=dynamic_build_update_2_max_diff,
                    dynamic_all_build_top_hits=dynamic_all_build_top_hits,
                    dynamic_build_update_1_top_hits=dynamic_build_update_1_top_hits,
                    dynamic_build_update_2_top_hits=dynamic_build_update_2_top_hits)



    # neutral loss search
    search_type='neutral_loss'
    [flash_all_build_result, 
        dynamic_all_build_result_, 
        dynamic_build_update_1_result_, 
        dynamic_build_update_2_result_]=_perform_search(search_type=search_type,
                                                    spec=spec,
                                                    flash_all_build=flash_all_build,
                                                    dynamic_all_build=dynamic_all_build,
                                                    dynamic_build_update_1=dynamic_build_update_1,
                                                    dynamic_build_update_2=dynamic_build_update_2)

    [flash_result, 
        dynamic_all_build_result, 
        dynamic_build_update_1_result, 
        dynamic_build_update_2_result]=_organize_result_by_id(
        spectra_list=spectra_list,
        all_spec=all_spec,
        flash_all_build_result=flash_all_build_result,
        dynamic_all_build_result_=dynamic_all_build_result_,
        dynamic_build_update_1_result_=dynamic_build_update_1_result_,
        dynamic_build_update_2_result_=dynamic_build_update_2_result_
        )
    

    [dynamic_all_build_max_diff, dynamic_build_update_1_max_diff, 
        dynamic_build_update_2_max_diff, dynamic_all_build_top_hits, 
        dynamic_build_update_1_top_hits, dynamic_build_update_2_top_hits]=calculate_max_diff_and_top_hits(
            flash_result=flash_result,
            dynamic_all_build_result=dynamic_all_build_result,
            dynamic_build_update_1_result=dynamic_build_update_1_result,
            dynamic_build_update_2_result=dynamic_build_update_2_result)
    
    _output_results(i=i,
                    search_type=search_type,
                    dynamic_all_build_max_diff=dynamic_all_build_max_diff,
                    dynamic_build_update_1_max_diff=dynamic_build_update_1_max_diff,
                    dynamic_build_update_2_max_diff=dynamic_build_update_2_max_diff,
                    dynamic_all_build_top_hits=dynamic_all_build_top_hits,
                    dynamic_build_update_1_top_hits=dynamic_build_update_1_top_hits,
                    dynamic_build_update_2_top_hits=dynamic_build_update_2_top_hits)

    # hybrid search
    search_type='hybrid'
    [flash_all_build_result, 
        dynamic_all_build_result_, 
        dynamic_build_update_1_result_, 
        dynamic_build_update_2_result_]=_perform_search(search_type=search_type,
                                                    spec=spec,
                                                    flash_all_build=flash_all_build,
                                                    dynamic_all_build=dynamic_all_build,
                                                    dynamic_build_update_1=dynamic_build_update_1,
                                                    dynamic_build_update_2=dynamic_build_update_2)

    [flash_result, 
        dynamic_all_build_result, 
        dynamic_build_update_1_result, 
        dynamic_build_update_2_result]=_organize_result_by_id(
        spectra_list=spectra_list,
        all_spec=all_spec,
        flash_all_build_result=flash_all_build_result,
        dynamic_all_build_result_=dynamic_all_build_result_,
        dynamic_build_update_1_result_=dynamic_build_update_1_result_,
        dynamic_build_update_2_result_=dynamic_build_update_2_result_
        )
    

    [dynamic_all_build_max_diff, dynamic_build_update_1_max_diff, 
        dynamic_build_update_2_max_diff, dynamic_all_build_top_hits, 
        dynamic_build_update_1_top_hits, dynamic_build_update_2_top_hits]=calculate_max_diff_and_top_hits(
            flash_result=flash_result,
            dynamic_all_build_result=dynamic_all_build_result,
            dynamic_build_update_1_result=dynamic_build_update_1_result,
            dynamic_build_update_2_result=dynamic_build_update_2_result)
    
    _output_results(i=i,
                    search_type=search_type,
                    dynamic_all_build_max_diff=dynamic_all_build_max_diff,
                    dynamic_build_update_1_max_diff=dynamic_build_update_1_max_diff,
                    dynamic_build_update_2_max_diff=dynamic_build_update_2_max_diff,
                    dynamic_all_build_top_hits=dynamic_all_build_top_hits,
                    dynamic_build_update_1_top_hits=dynamic_build_update_1_top_hits,
                    dynamic_build_update_2_top_hits=dynamic_build_update_2_top_hits)

