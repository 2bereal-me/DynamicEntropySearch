import logging
import numpy as np
def write_spec(spec_info: dict, spec_data: np.ndarray, filename_output: str):
    """
    Write spectrum to a mgf file.
    :param spec_info:
    :param spec_data:
    :param filename_output:
    :return: None
    """
    if filename_output[-3:] != "mgf":
        logging.warning("Output filename is not a mgf file!")

    with open(filename_output, 'r+') as fo:
        _write_spec(spec_info, spec_data, fo)

def write_msp_spec(spec_info: dict, spec_data: np.ndarray, filename_output: str):
    """
    Write spectrum to a msp file.
    :param spec_info:
    :param spec_data:
    :param filename_output:
    :return: None
    """
    if filename_output[-3:] != "msp":
        logging.warning("Output filename is not a msp file!")

    with open(filename_output, 'r+') as fo:
        write_one_spectrum(fo, spec_info)


def _write_spec(spec_info, spec_data, fileout):
    out_str = ['BEGIN IONS']

    def __add_to_output_str_if_exist(str_pre, str_suffix, item_name):
        if item_name in spec_info:
            out_str.append(str_pre + str(spec_info[item_name]) + str_suffix)

    __add_to_output_str_if_exist('PEPMASS=', '', 'precursor_mz')
    __add_to_output_str_if_exist('SCANS=', '', 'raw_scan_num')
    __add_to_output_str_if_exist('MSLEVEL=', '', 'ms_level')
    __add_to_output_str_if_exist('RTINSECONDS=', '', 'retention_time')

    if 'ion_mode' in spec_info:
        out_str.append('CHARGE=1' +
                       ('+' if spec_info['ion_mode'] == "P" else '-'))

    for peak in spec_data:
        out_str.append(str(peak[0]) + ' ' + str(peak[1]))

    out_str.append('END IONS\n')
    fileout.seek(0,2)
    fileout.write('\n'.join(out_str))
    fileout.write("\n\n")
    
def _write_msp_spec(spec_info, spec_data, fileout):
    out_str = []

    def __add_to_output_str_if_exist(str_pre, str_suffix, item_name):
        if item_name in spec_info:
            out_str.append(str_pre + str_suffix + str(spec_info[item_name]))

    __add_to_output_str_if_exist('precursormz', ':', 'precursormz')
    __add_to_output_str_if_exist('ion_mode', ':', 'ion_mode')
    __add_to_output_str_if_exist('num peaks', ':', 'num peaks')

    # if 'ion_mode' in spec_info:
    #     out_str.append('CHARGE=1' +
    #                    ('+' if spec_info['ion_mode'] == "P" else '-'))

    for peak in spec_data:
        out_str.append(str(peak[0]) + ' ' + str(peak[1]))

    fileout.seek(0,2)
    fileout.write('\n'.join(out_str))
    fileout.write("\n\n\n")

def write_one_spectrum(fo, spectrum: dict):
    """
    Write one spectrum to .msp file. The name starts with _ will not be written.
    """
    for name in spectrum:
        if name == "peaks":
            continue
        if name.startswith("_"):
            continue
        if name.strip().lower() == "num peaks":
            continue

        item = spectrum[name]

        if name == "comments" and (not isinstance(item, str)):
            # Deal with comments
            str_comments = []
            for comments_name in item:
                if isinstance(item[comments_name], str):
                    str_comments.append('"{}={}"'.format(comments_name, item[comments_name]))
                else:
                    for sub_value in item[comments_name]:
                        str_comments.append('"{}={}"'.format(comments_name, sub_value))
            str_out = "Comments: " + " ".join(str_comments) + "\n"
        elif isinstance(item, list):
            str_out = "".join([str(name).capitalize() + ": " + str(x) + "\n" for x in item])
        else:
            str_out = str(name).capitalize() + ": " + str(item) + "\n"

        fo.write(str_out)

    fo.write("Num peaks: {}\n".format(len(spectrum["peaks"])))
    for p in spectrum["peaks"]:
        fo.write(" ".join([str(x) for x in p]))
        fo.write("\n")
    fo.write("\n\n")
