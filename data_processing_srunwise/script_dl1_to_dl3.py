import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import optimize
import subprocess

from astropy.coordinates import SkyCoord
from lstchain.io.config  import get_standard_config
from ctapipe.io          import read_table
import tables

# Other auxiliar scripts
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux
import geometry as geom
import lstpipeline

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


""" Source name in order to just complete the results file, and
in order to improve run organization."""
source_name = "crab"

# Standard paths for data in the IT cluster ---------
root_dl1 = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
# root_rfs = "/fefs/aswg/data/models/AllSky/20240131_allsky_v0.10.5_all_dec_base/"
root_rfs = "/fefs/aswg/data/models/AllSky/20230927_v0.10.4_crab_tuned/"
# root_mcs = "/fefs/aswg/data/mc/DL2/AllSky/20240131_allsky_v0.10.5_all_dec_base/TestingDataset/"
root_mcs = "/fefs/aswg/data/mc/DL2/AllSky/20230927_v0.10.4_crab_tuned/TestingDataset/"

# Root path of this script
root = os.getcwd() + "/"
# Path to store the configuration file we are going to use
config_file = root + "config/standard_config.json"
# Path to store objects
root_objects = root + f"objects/"
# Data main directory
root_data = root + f"../../data/cherenkov_transparency_corrections/{source_name}/"

# Directories for the data
dir_dl1b_scaled = root_data + "dl1_scaled/"
dir_dl1m_scaled = root_data + "dl1_merged_scaled/"
dir_dl2_scaled  = root_data + "dl2_scaled/"
dir_dl2         = root_data + "dl2/"
dir_dl3_scaled_base  = root_data + "dl3_scaled/"
dir_dl3_base         = root_data + "dl3/"
dir_irfs        = root_data + "irfs/"

def configure_lstchain():
    """Creates a file of standard configuration for the lstchain analysis. 
    It can be changed inside this function"""
    dict_config = get_standard_config()
    # We select the heuristic flatfield option in the standard configuration
    dict_config["source_config"]["LSTEventSource"]["use_flatfield_heuristic"] = True
    with open(config_file, "w") as json_file:
        json.dump(dict_config, json_file)

def main(input_str, flag_scaled_str):

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

    dir_dl3_scaled = dir_dl3_scaled_base + f"{run_number:05}/"
    dir_dl3        = dir_dl3_base        + f"{run_number:05}/"

    # Creating the directories in case they don't exist
    for path in [os.path.dirname(config_file), dir_dl1b_scaled, dir_dl1m_scaled, dir_dl2, dir_dl2_scaled, dir_dl3_scaled, dir_dl3, dir_irfs]:
        if not os.path.exists(path):
            os.makedirs(os.path.join(path), exist_ok=True)
    
    # Creating and storing a configuration file for lstchain processes
    configure_lstchain()



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
        "tstart"   : dcheck_tstart[0],            # datetime object
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
    dict_dchecks, dict_nodes = lstpipeline.add_mc_and_rfs_nodes(dict_dchecks, root_rfs, root_mcs, dict_source)




    for ir, run in enumerate(dict_dchecks.keys()):
    
        dir_run = dir_dl1b_scaled + f"{run:05}" + "/"
        
        sruns = [int(path.split(".")[-2]) for path in dict_dchecks[run]["dl1a"]["srunwise"]]
        
        dict_dchecks[run]["dl1b_scaled"] = {"srunwise" : []}
    
        for i, srun in enumerate(sruns[:subruns_num]):
    
            input_fname  = dict_dchecks[run]["dl1a"]["srunwise"][i]
            output_fname = dir_run + f"dl1_LST-1.Run{run:05}.{srun:04}.h5"
    
            dict_dchecks[run]["dl1b_scaled"]["srunwise"].append(output_fname)






    for ir, run in enumerate(dict_dchecks.keys()):
    
        dir_run = dir_dl1b_scaled + f"{run:05}" + "/"
        output_fname = dir_dl1m_scaled + f"dl1_LST-1.Run{run:05}.h5"

        if flag_scaled:
            command = f"lstchain_merge_hdf5_files --input-dir {dir_run} --output-file {output_fname} --run-number {run} --no-image"
            logger.info(command)
            
            subprocess.run(command, shell=True)
        
        dict_dchecks[run]["dl1b_scaled"]["runwise"] = output_fname





    for ir, run in enumerate(dict_dchecks.keys()):
    
        input_fname  = dict_dchecks[run]["dl1b_scaled"]["runwise"]
        output_fname = dir_dl2_scaled + input_fname.split("/")[-1].replace("dl1", "dl2", 1)
        rf_node      = dict_dchecks[run]["simulations"]["rf"]

        if flag_scaled:
            # Check if the file exists and delete if exists (may be empty or half filled)
            if os.path.exists(output_fname):
                logger.info(f"File already exists, deleting and re-computing:\n-->{output_fname}")
                os.remove(output_fname)
                
            logger.info(f"\nComputing dl2 for Run {run:5} (scaled data)")
            logger.info(f"--> {output_fname}\n")
            command = f"lstchain_dl1_to_dl2 --input-files {input_fname} --path-models {rf_node} --output-dir {dir_dl2_scaled} --config {config_file}"
            logger.info(command)
            
            subprocess.run(command, shell=True)
    
        dict_dchecks[run]["dl2_scaled"] = output_fname



    for ir, run in enumerate(dict_dchecks.keys()):
    
        input_fname  = dict_dchecks[run]["dl1a"]["runwise"]
        output_fname = dir_dl2 + input_fname.split("/")[-1].replace("dl1", "dl2", 1)
        rf_node      = dict_dchecks[run]["simulations"]["rf"]

        if not flag_scaled:
            # Check if the file exists and delete if exists (may be empty or half filled)
            if os.path.exists(output_fname):
                logger.info(f"File already exists, deleting and re-computing:\n-->{output_fname}")
                os.remove(output_fname)
            
            logger.info(f"\nComputing dl2 for Run {run:5} (original data)")
            logger.info(f"--> {output_fname}\n")
            command = f"lstchain_dl1_to_dl2 --input-files {input_fname} --path-models {rf_node} --output-dir {dir_dl2} --config {config_file}"
            logger.info(command)
            
            subprocess.run(command, shell=True)
        
        dict_dchecks[run]["dl2"] = output_fname





    # Already computed IRFs
    computed_irfs = glob.glob(dir_irfs + "*")
    
    for ir, run in enumerate(dict_dchecks.keys()):
        
        input_mc = dict_dchecks[run]["simulations"]["mc"]
    
        output_irf = dir_irfs + "irf_{}_{}.fits.gz".format(input_mc.split("/")[-3], input_mc.split("/")[-2])
    
        # we don't compute the IRF if it has been already done
        if output_irf not in computed_irfs:
            
            logger.info(f"\nComputing IRF for Run {run:5}")
            logger.info(f"--> {output_irf}\n")
            
            command = f"lstchain_create_irf_files --input-gamma-dl2 {input_mc} --output-irf-file {output_irf} --point-like"
            command = command + f" --energy-dependent-gh --energy-dependent-theta --overwrite"
            logger.info(command)
        
            subprocess.run(command, shell=True)
        
        else:
            logger.info("\nIRF {}_{} already computed\n".format(input_mc.split("/")[-3], input_mc.split("/")[-2]))
        dict_dchecks[run]["irf"] = output_irf





    ra_str  = "{}".format(dict_source["ra"]).replace(" ", "")
    dec_str = "{}".format(dict_source["dec"]).replace(" ", "")
    
    
    for ir, run in enumerate(dict_dchecks.keys()):
    
        # dir_run = dir_dl3 + f"{run:05}" + "/"    
        dl2_fname = dict_dchecks[run]["dl2"]
        dl2_fname_scaled = dict_dchecks[run]["dl2_scaled"]
        output_dl3 = dir_dl3 + f"dl3_LST-1.Run{run:05}.fits"
        output_dl3_scaled = dir_dl3_scaled + f"dl3_LST-1.Run{run:05}.fits"
        
        
        if not flag_scaled:
            logger.info(f"\nConverting dl2 for {run:5}")
            command = f"lstchain_create_dl3_file --input-dl2 {dl2_fname} --input-irf-path {dir_irfs} --output-dl3-path {dir_dl3} "
            command = command + f"--source-name {source_name} --source-ra {ra_str} --source-dec {dec_str} --config {config_file} --overwrite"
            logger.info(command)
            
            subprocess.run(command, shell=True)
    
        if flag_scaled:
            logger.info(f"--> {output_dl3}\n--> {output_dl3_scaled}\n")
            command = f"lstchain_create_dl3_file --input-dl2 {dl2_fname_scaled} --input-irf-path {dir_irfs} --output-dl3-path {dir_dl3_scaled} "
            command = command + f"--source-name {source_name} --source-ra {ra_str} --source-dec {dec_str} --config {config_file} --overwrite"
            logger.info(command)
            
            subprocess.run(command, shell=True)
            
        dict_dchecks[run]["dl3"] = output_dl3
        dict_dchecks[run]["dl3_scaled"] = output_dl3_scaled
            



    # logger.info(f"All dl3 files created 100%\n\n\nCreating index files...")
    
    # command = f"lstchain_create_dl3_index_files --input-dl3-dir {dir_dl3} --overwrite"
    # logger.info(command)
    # subprocess.run(command, shell=True)
    
    # command = f"lstchain_create_dl3_index_files --input-dl3-dir {dir_dl3_scaled} --overwrite"
    # logger.info(command)
    # subprocess.run(command, shell=True)
    
    # logger.info(f"\nFinished with the dl3 process")




if __name__ == "__main__":
    input_str = sys.argv[1]
    flag_scaled_str = sys.argv[2]
    main(input_str, flag_scaled_str)
