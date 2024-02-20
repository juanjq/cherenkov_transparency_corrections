import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
from scipy import optimize

from traitlets.config.loader import Config
from astropy.coordinates     import SkyCoord
from lstchain.io.config      import get_standard_config
from ctapipe.io              import read_table
import tables

# Other auxiliar scripts
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux
import geometry as geom
import lstpipeline
import plotting

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

""" Source name in order to just complete the results file, and
in order to improve run organization."""
source_name = "crab"

""" Fit parameters
Chosen limits in intensity (p.e.) for applying the fit i.e. the
power law will be fitted only with the points within this range."""
limits_intensity = [316, 562]
""" For the positive scaling cases (most of them), we need to have a lower 
limit in intensity. Thi slimit is used for the subset of events that are 
scaled just to find which is the scaling value. We use a very low limit by
default 60 p.e. compared to the lower limit of the fit 316 p.e. because in 
the worst cases we will have a very non-linear scaling that will displace 
significantly the events intensities."""
limits_intensity_extended = 60

""" Power law parameters for the reference
All these parameters are taken from a common analysis of the full dataset
Where the period of end of 2022 and start 2023 is taken as reference for good 
runs. Then we take as reference the mean power law parameters in that period.
p0 is the normalization factor and p1 is the slope."""
ref_p0 =  1.74 
ref_p1 = -2.23

""" Threshold in statistics for the last subrun
The limit in number of events after cleaning that we need to consider the last
subrun has enough statistics to perform the analysis over it. Otherwise the 
values of the scaling that will be applied to this last rubrun are the same 
that are applied to the last last subrun."""
statistics_threshold_last_srun = 15000

""" Parameters for the empyrical fits for Zenith Distance corrections
Are simply two 2 degree polynomials for each variable of the power law."""
p0a, p0b, p0c = -0.44751321, 3.62502037, -1.43611437
p1a, p1b, p1c = -2.89253919, 0.99443581, -0.34013068


def main():

    ################################
    # Defining paths and directories
    # Root path of this script
    root = os.getcwd() + "/"
    # Path to store the configuration file we are going to use
    config_file = root + "config/standard_config.json"
    # Path to store objects
    root_objects = root + f"objects/"
    # Data main directory
    root_data = root + f"../../data/cherenkov_transparency_corrections/{source_name}/"
    # Directory for the results of the fit of each run
    root_results = root_objects + "results_fits/"
    root_final_results = root_objects + "final_results_fits/"
    
    # STANDARD paths ---------
    root_dl1 = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
    root_rfs = "/fefs/aswg/data/models/AllSky/20230901_v0.10.4_allsky_base_prod/"
    root_mcs = "/fefs/aswg/data/mc/DL2/AllSky/20230901_v0.10.4_allsky_base_prod/TestingDataset/"
    
    # Create the paths that do not exist
    for path in [root_final_results]:
        if not os.path.exists(path):
            os.makedirs(os.path.join(path), exist_ok=True)

    ###########################################################
    # Storing all the dictionaries stored in the results folder    
    dict_files = np.sort(glob.glob(root_results + "*.pkl"))
    
    total_runs = []
    dictionaries = []
    for file in dict_files:
    
        fname = file.split("/")[-1]
        run   = int(fname.split("_")[2])
        sruns = [int(sr) for sr in fname.split(".")[0].split("_")[3:]]
    
        total_runs.append(run)
    
        # Also we store the dictionaries
        # Reading the object
        with open(file, 'rb') as f:
            tmp_dict = pickle.load(f)
        dictionaries.append(tmp_dict)

    ###################################
    # Merging the dictionaries run-wise
    # Keep only non-repeated runs
    total_runs = np.unique(total_runs)
    
    dict_runs = {}
    for run in total_runs:
        tmp = { 
            "run": run, "filenames": {}, "statistics": {}, "flag_error" : {},
            "scaled" :           {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "p0":                {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "delta_p0":          {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "p1":                {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "delta_p1":          {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "chi2":              {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "pvalue":            {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "light_yield":       {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "delta_light_yield": {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "scaling":           {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "delta_scaling":     {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "scaling_percent":       {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "delta_scaling_percent": {"original": {}, "upper": {}, "linear": {}, "final": {}},
            "final_scaling": {}, "final_scaling_interpolated": {}, "interpolation" : {},
        }
        dict_runs[run] = tmp
    
    def merge_dicts(dict1, dict2):
        """
        Recursively merge two dictionaries with nested structures.
    
        Args:
        - dict1: First dictionary
        - dict2: Second dictionary
    
        Returns:
        - Merged dictionary
        """
        merged = dict1.copy()
    
        for key, value in dict2.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # If both values are dictionaries, recursively merge them
                merged[key] = merge_dicts(merged[key], value)
            else:
                # Otherwise, just update or add the key-value pair
                merged[key] = value
        return merged
        
    # Now we fill this dicts with the info we have
    for d in dictionaries:
        run = d["run"]
    
        dict_runs[run] = merge_dicts(dict_runs[run], d)

    ##############################
    # Last subrun statistics check
    for run in dict_runs.keys():
        d = dict_runs[run]
        stats = dict_runs[run]["statistics"]
        last_srun = max(stats.keys())
        last_last_srun = last_srun - 1
    
        last_srun_stats = stats[last_srun]
    
        # If we have less events we modify the values
        if last_srun_stats < statistics_threshold_last_srun:
    
            dict_runs[run]["final_scaling"][last_srun] = dict_runs[run]["final_scaling"][last_last_srun]

    #######################
    # Filling the datacheck
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
    for run in total_runs:
        dict_dchecks[run] = {
            "run_num" : run,
        }
    
    dict_dchecks = lstpipeline.add_dl1_paths_to_dict(dict_dchecks, root_dl1)
    dict_dchecks = lstpipeline.add_dl1_paths_to_dict(dict_dchecks, root_dl1, dchecking=True)

    for run_number in total_runs:
        tab_dcheck_run = read_table(dict_dchecks[run_number]["dchecks"]["runwise"], "/dl1datacheck/cosmics")
        
        # reading the variables
        dcheck_zd = 90 - np.rad2deg(np.array(tab_dcheck_run["mean_alt_tel"]))
        dcheck_az = np.rad2deg(np.array(tab_dcheck_run["mean_az_tel"]))
        dcheck_tstart = tab_dcheck_run["dragon_time"][0][0]
        dcheck_telapsed = np.array(tab_dcheck_run["elapsed_time"])
    
        dict_dchecks[run_number]["time"] = {
            "tstart"   : dcheck_tstart,            # datetime object
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


    #######################################################################################
    # For each run we perform the interpolation and store the info in a run-wise dictionary
    for run_number in total_runs:
    
        dict_results = dict_runs[run_number]
    
        x_fit = np.cumsum(dict_dchecks[run_number]["time"]["srunwise"]["telapsed"])
        y_fit = np.array([dict_results["final_scaling"][srun] for srun in np.sort(list(dict_results["final_scaling"].keys()))])
    
        # Performing the fit
        params, pcov, info, _, _ = curve_fit(
            f     = geom.straight_line,
            xdata = x_fit,
            ydata = y_fit,
            p0    = [1, 0],
            full_output = True,
        )
            
        intercept       = params[0]
        slope           = params[1]
        delta_intercept = np.sqrt(pcov[0, 0])
        delta_slope     = np.sqrt(pcov[1, 1])
        _chi2           = np.sum(info['fvec'] ** 2)
        pvalue          = 1 - chi2.cdf(_chi2, len(x_fit))
        
        dict_results["interpolation"] = {
            "chi2" : _chi2,      
            "p_value" : pvalue,         
            "slope": slope,      
            "delta_slope" : delta_slope,     
            "intercept" : intercept, 
            "delta_intercept" : delta_intercept,
        }
        
        # Setting a interpolated scaling factor
        for srun in dict_results["final_scaling"].keys():
            
            scaling_interpolated = intercept + slope * x_fit[srun]
            
            dict_results["final_scaling_interpolated"][srun] = scaling_interpolated
            dict_results["scaled"]["final"][srun]            = scaling_interpolated

    dict_fname = root_final_results + f"results_job_{run_number}.pkl"
    
    # Saving the objects
    with open(dict_fname, 'wb') as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()