import numpy as np
import astropy.units as u
from datetime import datetime
import sys
import os
import subprocess
import tables

from astropy.coordinates import SkyCoord
from ctapipe.io import read_table

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Other auxiliar scripts
sys.path.insert(0, os.path.join(os.getcwd(), "../scripts/"))
import geometry as geom
import lstpipeline
import script_0_utils_scaling as utils


# EXPORT ALL PARAMETERS
# ///////////////////////////////////////////////
# The configuration file path is fixed for the scrips
fname_config_scaling = os.path.join("config", "config_scaling_parameters.json")

# Code to read it
with open(fname_config_scaling, "r") as json_file:
    configuration_dictionary = json.load(json_file)

# Now we wxtract all the variables with the same name as in the dictionary
source_name = configuration_dictionary["source_name"]
for superkey in ["fit_parameters", "paths"]:
    for key, value in configuration_dictionary[superkey].items():
        globals()[key] = value
# ///////////////////////////////////////////////


def main(input_str, flag_scaled_str, simulated=False):

    # Run numbers to analyze, in our case only one
    run_number = int(input_str)

    # Reading the scaled or not flag
    if flag_scaled_str == "True":
        flag_scaled = True
    elif flag_scaled_str == "False":
        flag_scaled = False
    else:
        logger.error(f"Input string for scaling: {flag_scaled_str} not valid.\nInput 'True' or 'False'")
    
    # Number of subruns to analyze per run
    subruns_num = None  # Specify the number of subruns you want to analyze, set subruns_num = None to analyze all subruns

    dir_dl3        = os.path.join(dir_dl3_base, f"{run_number:05}")
    dir_dl3_scaled = os.path.join(dir_dl3_scaled_base, f"{run_number:05}")

    # Creating the directories in case they don't exist
    for path in [os.path.dirname(config_file), dir_dl1b_scaled, dir_dl1m_scaled, dir_dl2, dir_dl2_scaled, dir_dl3_scaled, dir_dl3, dir_irfs]:
        os.makedirs(os.path.join(path), exist_ok=True)
    
    # Creating and storing a configuration file for lstchain processes
    utils.configure_lstchain(config_file)



    # Getting coordinates of source
    source_coords = SkyCoord.from_name(source_name)
    
    dict_source = {
        "name"   : source_name,
        "coords" : source_coords,
        "ra"     : source_coords.ra.deg  * u.deg, # ra in degrees
        "dec"    : source_coords.dec.deg * u.deg, # dec in degrees
    }
    
    # We create a empty dictionary to store all the information needed inside
    dict_dchecks = {}
    for run in [run_number]:
        dict_dchecks[run] = {
            "run_num" : run,
        }
    # Then we add the paths to the files and the datachecks
    dict_dchecks = lstpipeline.add_dl1_paths_to_dict(dict_dchecks, root_dl1)
    dict_dchecks = lstpipeline.add_dl1_paths_to_dict(dict_dchecks, root_dl1, dchecking=True)



    dcheck_zd, dcheck_az = [], []
    dcheck_tstart, dcheck_telapsed = [], []
    
    for srun in range(len(dict_dchecks[run_number]["dchecks"]["srunwise"])):
        tab_dcheck_srun = read_table(dict_dchecks[run_number]["dchecks"]["srunwise"][srun], "/dl1datacheck/cosmics")
        
        # reading the variables
        dcheck_zd.append(90 - np.rad2deg(tab_dcheck_srun["mean_alt_tel"]))
        dcheck_az.append(np.rad2deg(tab_dcheck_srun["mean_az_tel"]))
        
        dcheck_tstart.append(tab_dcheck_srun["dragon_time"])
        dcheck_telapsed.append(tab_dcheck_srun["elapsed_time"])
    
    dcheck_zd = np.array(dcheck_zd)
    dcheck_az = np.array(dcheck_az)
    dcheck_tstart = np.array(dcheck_tstart)
    dcheck_telapsed = np.array(dcheck_telapsed)
    
    dict_dchecks[run_number]["time"] = {
        "tstart"   : dcheck_tstart[0],         # datetime object
        "telapsed" : np.sum(dcheck_telapsed),  # s
        "srunwise" : {
            "telapsed" : dcheck_telapsed,      # s      
        },
    }
    dict_dchecks[run_number]["pointing"] = {
        "zd" : np.mean(dcheck_zd),  # deg
        "az" : np.mean(dcheck_az),  # deg
        "srunwise" : {
            "zd" : dcheck_zd, # deg
            "az" : dcheck_az, # deg
        },
    }
    # then we also select the RFs and MC files looking at the nodes available
    dict_dchecks = lstpipeline.add_rf_node(dict_dchecks, root_rfs, dict_source)




    for ir, run in enumerate(dict_dchecks.keys()):
    
        dir_run = os.path.join(dir_dl1b_scaled, f"{run:05}")
        
        sruns = [int(path.split(".")[-2]) for path in dict_dchecks[run]["dl1a"]["srunwise"]]
        
        dict_dchecks[run]["dl1b_scaled"] = {"srunwise" : []}
    
        for i, srun in enumerate(sruns[:subruns_num]):
    
            input_fname  = dict_dchecks[run]["dl1a"]["srunwise"][i]
            output_fname = os.path.join(dir_run, f"dl1_LST-1.Run{run:05}.{srun:04}.h5")
    
            dict_dchecks[run]["dl1b_scaled"]["srunwise"].append(output_fname)





    for ir, run in enumerate(dict_dchecks.keys()):
    
        dir_run = os.path.join(dir_dl1b_scaled, f"{run:05}")
        output_fname = os.path.join(dir_dl1m_scaled, f"dl1_LST-1.Run{run:05}.h5")
    
        if (not simulated) and (flag_scaled):
            if os.path.exists(output_fname):
                logger.info(f"File already exists, deleting and re-computing:\n-->{output_fname}")
                os.remove(output_fname)
    
            command_mergehdf5 = f"lstchain_merge_hdf5_files --input-dir {dir_run} --output-file {output_fname} --no-image"
            logger.info(command_mergehdf5)
            
            subprocess.run(command_mergehdf5, shell=True)
        
        dict_dchecks[run]["dl1b_scaled"]["runwise"] = output_fname



    
    for ir, run in enumerate(dict_dchecks.keys()):
    
        input_fname         = dict_dchecks[run]["dl1a"]["runwise"]
        input_fname_scaled  = dict_dchecks[run]["dl1b_scaled"]["runwise"]
        output_fname        = os.path.join(dir_dl2,        os.path.basename(input_fname       ).replace("dl1", "dl2", 1))
        output_fname_scaled = os.path.join(dir_dl2_scaled, os.path.basename(input_fname_scaled).replace("dl1", "dl2", 1))
        rf_node             = dict_dchecks[run]["simulations"]["rf"]

        # Not scaled case
        if (not simulated) and (not flag_scaled):
            _dir_dl2 = dir_dl2
            _input_fname = input_fname
            _output_fname = output_fname
        # Scaled case
        elif (not simulated) and (flag_scaled):
            _dir_dl2 = dir_dl2_scaled
            _input_fname = input_fname_scaled
            _output_fname = output_fname_scaled

        if (not simulated):
            # Check if the file exists and delete if exists (may be empty or half filled)
            if os.path.exists(_output_fname):
                logger.info(f"File already exists, deleting and re-computing:\n-->{_output_fname}")
                os.remove(_output_fname)
                
            logger.info(f"\nComputing dl2 for Run {run:5} (Scaled={str(flag_scaled)})")
            logger.info(f"--> {_output_fname}\n")
            command_dl1dl2 = f"lstchain_dl1_to_dl2 --input-files {_input_fname} --path-models {rf_node} "
            command_dl1dl2 = command_dl1dl2 + f"--output-dir {_dir_dl2} --config {config_file}"
            logger.info(command_dl1dl2)
            
            subprocess.run(command_dl1dl2, shell=True)
    
        dict_dchecks[run]["dl2"] = output_fname
        dict_dchecks[run]["dl2_scaled"] = output_fname_scaled
        
    
    ra_str  = "{}".format(dict_source["ra"]).replace(" ", "")
    dec_str = "{}".format(dict_source["dec"]).replace(" ", "")
    
    
    for ir, run in enumerate(dict_dchecks.keys()):
     
        dl2_fname         = dict_dchecks[run]["dl2"]
        dl2_fname_scaled  = dict_dchecks[run]["dl2_scaled"]
        output_dl3        = os.path.join(dir_dl3,        f"dl3_LST-1.Run{run:05}.fits")
        output_dl3_scaled = os.path.join(dir_dl3_scaled, f"dl3_LST-1.Run{run:05}.fits")
        
        
        # Not scaled case
        if (not simulated) and (not flag_scaled):
            logger.info(f"\nConverting dl2 for {run:5}")
            logger.info(f"--> {output_dl3}\n")
            
            command_dl3 = f"lstchain_create_dl3_file --input-dl2 {dl2_fname} --input-irf-path {dir_irfs} "
            command_dl3 = command_dl3 + f"--output-dl3-path {dir_dl3} --source-name {source_name} --source-ra {ra_str} "
            command_dl3 = command_dl3 + f"--source-dec {dec_str} --config {config_file} --overwrite"
            logger.info(command_dl3)
            
            subprocess.run(command_dl3, shell=True)
    
        # Scaled case
        elif (not simulated) and (flag_scaled):
            logger.info(f"\nConverting dl2 for {run:5}")
            logger.info(f"--> {output_dl3_scaled}\n")
            
            command_dl3 = f"lstchain_create_dl3_file --input-dl2 {dl2_fname_scaled} --input-irf-path {dir_irfs} "
            command_dl3 = command_dl3 + f"--output-dl3-path {dir_dl3_scaled} --source-name {source_name} --source-ra {ra_str} "
            command_dl3 = command_dl3 + f"--source-dec {dec_str} --config {config_file} --overwrite"
            logger.info(command_dl3)
            
            subprocess.run(command_dl3, shell=True)
            
        dict_dchecks[run]["dl3"] = output_dl3
        dict_dchecks[run]["dl3_scaled"] = output_dl3_scaled



if __name__ == "__main__":
    input_str = sys.argv[1]
    flag_scaled_str = sys.argv[2]
    main(input_str, flag_scaled_str)
