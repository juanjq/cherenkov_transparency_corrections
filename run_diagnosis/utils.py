import numpy as np
import glob
import re
import os


def check_files_exist(global_variables):
    for file in [f"path_dl{i}" for i in [1, 2, 3]]:
        # Check that files exist
        if not (os.path.isfile(global_variables[file])):
            raise FileNotFoundError(f"Input DL{file[-1:]} file ({global_variables[file]}) not found.")
        if not (os.path.isfile(global_variables[file+"_scaled"])): #
            raise FileNotFoundError(f"Input DL{file[-1:]} scaled file ({global_variables[file+'_scaled']}) not found.") #
    file = f"path_dl2_radec"
    if not (os.path.isfile(global_variables[file])):
        raise FileNotFoundError(f"Input DL2-radec file ({global_variables[file]}) not found.")
    if not (os.path.isfile(global_variables[file+"_scaled"])): #
        raise FileNotFoundError(f"Input DL2-radec scaled file ({global_variables[file+'_scaled']}) not found.") #
    

def find_dl1_fname(run_number, dchecking=False, version_string="v*", return_version=False, print_details=True):
    str_dchecks = "" if not dchecking else "datacheck_"

    # Root location in IT cluster for DL1 data and the filename
    root_dl1 = f"/fefs/aswg/data/real/DL1/*/{version_string}/tailcut84/"
    fname_dl1_runwise = f"{str_dchecks}dl1_LST-1.Run{run_number:05}.h5"
    # Finding all DL1 files corresponding to the provided run number
    files_dl1a_runwise = np.sort(glob.glob(root_dl1 + "*/" + fname_dl1_runwise) + glob.glob(root_dl1 + fname_dl1_runwise))

    # Checking runs we have, not, or we have duplicated
    if len(files_dl1a_runwise) == 0:
        raise ValueError(f"Run {run_number:5} not found in {root_dl1}")
    
    elif len(files_dl1a_runwise) > 1:
        print(f"DL1: Run {run_number:5} presented {len(files_dl1a_runwise)} different versions:") if print_details else None
        
        str_versions, versions, lengths_versions = [], [], []
        for i, runfile in enumerate(files_dl1a_runwise):
            
            str_version = runfile.split("/")[7][1:]   # Getting the version string e.g. "0.10.1_test3"
            str_parts = re.split("\.|_", str_version) # Splitting in parts e.g. ["0", "10", "1", "test3"]
            # Then we extract as float, whenever there is only digits
            str_parts_float = [float(part) for part in str_parts if part.isdigit()]

            # Then we construct a float number associated to each version
            for ii, part in enumerate(str_parts_float):
                final_float_str = f"{part:04.0f}." if ii == 0 else final_float_str + f"{part:04.0f}"

            str_versions.append(f"v{str_version}")
            versions.append(float(final_float_str)) 
            lengths_versions.append(len(str_version))
    
        version_index = 0
        for i in range(1, len(versions)):
            condition_larger_float = versions[i] > versions[version_index]
            condition_get_shorter  = (versions[i] == versions[version_index] and lengths_versions[i] < lengths_versions[version_index])
            if condition_larger_float or condition_get_shorter:
                version_index = i
        
        for i, runfile in enumerate(files_dl1a_runwise):
            str_selected = "<-- (SELECTED)" if i == version_index else ""
            print(f"* {str_versions[i]} {str_selected}") if print_details else None
        
        final_fname   = files_dl1a_runwise[version_index]
        final_version = str_versions[version_index]
    
    else:
        final_fname   = files_dl1a_runwise[0]
        final_version = final_fname.split("/")[7]
        print(f"DL1 file version: {final_version}") if print_details else None

    if return_version:
        return final_fname, final_version
    else:
        return final_fname
