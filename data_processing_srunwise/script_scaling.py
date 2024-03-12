import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import chi2
import subprocess

from astropy.coordinates import SkyCoord
from lstchain.io.config  import get_standard_config
from ctapipe.io          import read_table
import tables

# Other auxiliar scripts
sys.path.insert(0, os.path.join(os.getcwd(), "../scripts/"))
import geometry as geom
import lstpipeline
import script_utils_scaling as utils

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
statistics_threshold = 10000

""" Parameters for the empyrical fits for Zenith Distance corrections
Are simply two 2 degree polynomials for each variable of the power law."""
p0a, p0b, p0c = -0.44751321, 3.62502037, -1.43611437
p1a, p1b, p1c = -2.89253919, 0.99443581, -0.34013068

# Standard paths for data in the IT cluster ---------
root_dl1 = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
root_rfs = "/fefs/aswg/data/models/AllSky/20230927_v0.10.4_crab_tuned/"
root_mcs = "/fefs/aswg/data/mc/DL2/AllSky/20230927_v0.10.4_crab_tuned/TestingDataset/"

# Root path of this script
root = os.getcwd()
# Path to store the configuration file we are going to use
config_file = os.path.join(root, "config/standard_config.json")
# Path to store objects
root_objects = os.path.join(root, f"objects/")
# Data main directory
root_data = os.path.join(root, f"../../data/cherenkov_transparency_corrections/{source_name}/")
# Sub-dl1 objects directory
root_sub_dl1 = os.path.join(root_objects, "sub_dl1/")
# Directory for the results of the fit of each run
root_results = os.path.join(root_objects, "results_fits/")
root_final_results = os.path.join(root_objects, "final_results_fits/")
# Configuration file for the job launching
file_job_config = os.path.join(root, "config/job_config_runs.txt")

# Directories for the data
dir_dl1b_scaled = os.path.join(root_data, "dl1_scaled/")
dir_dl1m_scaled = os.path.join(root_data, "dl1_merged_scaled/")
dir_dl2_scaled = os.path.join(root_data, "dl2_scaled/")
dir_dl2 = os.path.join(root_data, "dl2/")
dir_dl3_scaled_base = os.path.join(root_data, "dl3_scaled/")
dir_dl3_base = os.path.join(root_data, "dl3/")
dir_irfs = os.path.join(root_data, "irfs/")



def main_irf_creation():

    # Creating configuration file
    utils.configure_lstchain(config_file)
    
    # Computing IRF for each MC file
    all_mcs_dl2 = np.sort(glob.glob(os.path.join(root_mcs, "*/*/*.h5")))
    for file_mc in all_mcs_dl2:
        # Filename of MC
        fname_mc  = os.path.basename(file_mc)
        # Creating the derived MC filename
        fname_irf = fname_mc.replace("dl2", "irf").replace(".h5", ".fits.gz")
        path_irf  = os.path.join(dir_irfs, fname_irf)
    
        logger.info(f"\nComputing IRF for MC file: {fname_mc}")
        logger.info(f"--> {fname_irf}\n")
        
        command_irfs = f"lstchain_create_irf_files --input-gamma-dl2 {file_mc} --output-irf-file {path_irf} --point-like"
        command_irfs = command_irfs + f" --energy-dependent-gh --energy-dependent-theta --config {config_file}"
        logger.info(command_irfs)
        subprocess.run(command_irfs, shell=True)

    
def main_init(input_str, simulate_data=False):

    ########################################
    # Initial configuring and paths creation
    # Extracting the run number from the input string
    run_number   = int(input_str.split("_")[0])
    first_last_srun = [int(s) for s in input_str.split("_")[1:]]
    if len(first_last_srun) == 1:
        srun_numbers = np.array(first_last_srun)
    else:
        srun_numbers = np.arange(first_last_srun[0], first_last_srun[1] + 1)
    """ Empty dictionary to store all the results of one run."""
    dict_results_empty = { 
        "run": run_number, "filenames": {}, "statistics": {}, "flag_error" : {},
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
        "final_scaling": {}, "delta_final_scaling": {}, "final_scaling_interpolated": {}, "interpolation" : {},
    }
    # Create the paths that do not exist
    for path in [os.path.dirname(config_file), root_data, root_objects, root_results, root_final_results, root_sub_dl1]:
        os.makedirs(os.path.join(path), exist_ok=True)
    # Creating and storing a configuration file for lstchain processes
    utils.configure_lstchain(config_file)
    
    ################################################################
    # Generating a dictionary with the information of all datachecks
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
    dict_dchecks = lstpipeline.add_rf_node(dict_dchecks, root_rfs, dict_source)
    
    
    # Pivot intensity for decorrelation
    ref_intensity = (limits_intensity[0] * limits_intensity[1]) ** 0.5
    logger.info(f"The intensity in the middle of the intensity range is {ref_intensity:.1f} p.e.")
    
    ########################################################
    # Reading the binning from the datacheck ---------------
    # Opening the corresponding datacheck
    fname_dcheck = dict_dchecks[run_number]["dchecks"]["runwise"]
    tab_dcheck_run = tables.open_file(fname_dcheck)
    
    # Read the binning from the datacheck of the first subrun
    dcheck_intensity_binning = np.array(tab_dcheck_run.root.dl1datacheck.histogram_binning.col("hist_intensity")[0])
    # Calculating the logarithmic center of each bin
    dcheck_intensity_binning_centers = (dcheck_intensity_binning[:-1] * dcheck_intensity_binning[1:]) ** 0.5
    # Calculating the width of each bin
    dcheck_intensity_binning_widths = np.diff(dcheck_intensity_binning)
    tab_dcheck_run.close()
    
    # Mask for the fitting region in the fits
    mask_dcheck_bins_fit = (
        (dcheck_intensity_binning_centers >= limits_intensity[0]) &
        (dcheck_intensity_binning_centers <= limits_intensity[1])
    )
      
    ##########################################################
    # Reading the histogram from the datacheck ---------------
    # Opening the corresponding datacheck
    dcheck_hist_intensities = []
    for fname_dcheck_srun in dict_dchecks[run_number]["dchecks"]["srunwise"]:
        tab_dcheck_srun = tables.open_file(fname_dcheck_srun)
        dcheck_hist_intensities.append(np.array(tab_dcheck_srun.root.dl1datacheck.cosmics.col("hist_intensity")))
        tab_dcheck_srun.close()
    
    # Converting from counts to rate per intensity unit (non-binning dependent quantity)
    dcheck_rates       = [] # Array of histogram of rates for each subrun
    dcheck_delta_rates = [] # The statistical error
    for srun, dcheck_hist_intensity in enumerate(dcheck_hist_intensities):
    
        effective_time_srun = dict_dchecks[run_number]["time"]["srunwise"]["telapsed"][srun]
        
        dcheck_rates.append(              dcheck_hist_intensity  / effective_time_srun / dcheck_intensity_binning_widths)
    
        dcheck_delta_rates.append(np.sqrt(dcheck_hist_intensity) / effective_time_srun / dcheck_intensity_binning_widths)
    
    ####################################
    # Zenith correction factors to apply
    corr_factor_p0 = geom.pol2(1, p0a, p0b, p0c) / geom.pol2(np.cos(np.deg2rad(dict_dchecks[run_number]["pointing"]["zd"])), p0a, p0b, p0c)
    corr_factor_p1 = geom.pol2(1, p1a, p1b, p1c) - geom.pol2(np.cos(np.deg2rad(dict_dchecks[run_number]["pointing"]["zd"])), p1a, p1b, p1c)
    
    # Zenith correction of the reference (putting the reference in the zenith of the determined subrun)
    corr_ref_p0 = ref_p0 / corr_factor_p0
    corr_ref_p1 = ref_p1 - corr_factor_p1
    
    ####################################################
    # Parameters to input to the find_scaling() function
    other_parameters = {
        "srun_numbers" : srun_numbers,
        "dict_dchecks" : dict_dchecks,
        "ref_intensity" : ref_intensity,
        "dcheck_intensity_binning" : dcheck_intensity_binning,
        "dcheck_intensity_binning_widths" : dcheck_intensity_binning_widths,
        "dcheck_intensity_binning_centers" : dcheck_intensity_binning_centers,
        "mask_dcheck_bins_fit" : mask_dcheck_bins_fit,
        "corr_factor_p0" : corr_factor_p0, "corr_factor_p1" : corr_factor_p1,
        "root_sub_dl1" : root_sub_dl1,
        "dir_dl1b_scaled" : dir_dl1b_scaled,
        "lims_intensity" : lims_intensity,
        "lims_intensity_extended" : lims_intensity_extended,
        "config_file" : config_file,
        "ref_p0" : ref_p0, "ref_p1" : ref_p1,
    }

    #######################
    # Reading original data
    # The main results dictionary
    dict_results = dict_results_empty.copy()
    # First filling the dictionary with ones in the scaled values
    # and saving the number of events stored in each subrun.
    for srun in srun_numbers:
        dict_results["scaled"]["original"][srun] = 1.0
        dict_results["statistics"][srun] = int(np.sum(dcheck_hist_intensities[srun]))
    
    # Then we read these files and perform the fits
    dict_results = utils.find_scaling(
        iteration_step="original", dict_results=dict_results, other_parameters=other_parameters, simulated=simulate_data
    )
    
    # Then filling the next step "scaled" with the calculated one
    for srun in srun_numbers:
        dict_results["scaled"]["upper"][srun] = dict_results["scaling"]["original"][srun]
    
    ###################################
    # Then performing the upper scaling
    dict_results = utils.find_scaling(
        iteration_step="upper", dict_results=dict_results, other_parameters=other_parameters, simulated=simulate_data
    )
    
    #####################################################
    # Calculating the linear factor to the linear scaling
    for srun in srun_numbers:
    
        # Now putting all together, upper and half
        points_scaling     = np.array([1, dict_results["scaling"]["original"][srun]])
        points_light_yield = np.array([dict_results["light_yield"]["original"][srun], dict_results["light_yield"]["upper"][srun]])
    
        # Finding the final scaling as a line that pass trogh the two points we have
        # Then we calculate where the light yield will be 1 in linear approximation
        slope = (points_light_yield[1] - points_light_yield[0]) / (points_scaling[1] - points_scaling[0])
        intercept = points_light_yield[0] - slope * points_scaling[0]
        linear_scale_factor = 1 / slope - points_light_yield[0] / slope + points_scaling[0]
    
        dict_results["scaled"]["linear"][srun] = linear_scale_factor
    
    ##########################################
    # Then applying this linear scaling factor
    dict_results = utils.find_scaling(
        iteration_step="linear", dict_results=dict_results, other_parameters=other_parameters, simulated=simulate_data
    )
    
    ###################################################
    # And finally calculating the final scaling factors
    for srun in srun_numbers:
    
        # Only calculating for the cases with no flag errors:
        if not dict_results["flag_error"][srun]:
    
            # Now putting all together, upper and half
            points_scaling           = np.array([dict_results["scaled"][key][srun]            for key in ["original", "linear", "upper"]])
            points_light_yield       = np.array([dict_results["light_yield"][key][srun]       for key in ["original", "linear", "upper"]])
            points_delta_light_yield = np.array([dict_results["delta_light_yield"][key][srun] for key in ["original", "linear", "upper"]])
            
            if simulate_data:
                points_scaling           = np.array([1, 1.2, 1.4])       + np.random.rand(3) * 0.1
                points_light_yield       = np.array([0.7, 0.9, 1.2])     + np.random.rand(3) * 0.1
                points_delta_light_yield = np.array([0.05, 0.05, 0.05])  + np.random.rand(3) * 0.01        
                
            srun_a, srun_b, srun_c, srun_delta_a, srun_delta_b, srun_delta_c = geom.parabola_3points(
                *points_scaling, *points_light_yield, *points_delta_light_yield
            )
            
            range_avg_point = np.mean(points_scaling)
            x0, delta_x0 = geom.get_roots_pol2(
                range_avg_point, 1,*points_scaling, *points_light_yield, *points_delta_light_yield
            )
    
            final_scale_factor = x0
            delta_final_scale_factor = delta_x0
    
        else:
            final_scale_factor = np.nan
            delta_final_scale_factor = np.nan
            
        dict_results["final_scaling"][srun] = final_scale_factor
        dict_results["delta_final_scaling"][srun] = delta_final_scale_factor
    
    ##############################
    # Storing data in a pkl object
    dict_fname = os.path.join(root_results, f"results_job_{input_str}.pkl")
    
    # Saving the objects
    with open(dict_fname, "wb") as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)


def main_merge():

    #############################################
    #Reading all the information on the directory
    # All the stored dictionaries that are inside the results folder
    dict_files = np.sort(glob.glob(os.path.join(root_results, "*.pkl")))
    
    # Storing all the run numbers and all the separate dictionaries
    total_runs, dictionaries = [], []
    for file in dict_files:
    
        # Reading the dictionaries using pickle
        with open(file, "rb") as f:
            tmp_dict = pickle.load(f)
    
        total_runs.append(int(os.path.basename(file).split("_")[2]))
        dictionaries.append(tmp_dict)

    ##############################################
    # Cleaning some directories with temporal data
    # Iterate over all the entries in the directory
    for dir_to_delete in [root_sub_dl1]:
        for entry in os.listdir(dir_to_delete):
            entry_path = os.path.join(dir_to_delete, entry)
        
            # Check if it"s a file and delete it
            if os.path.isfile(entry_path):
                os.remove(entry_path)
                
    runs_config_file_str = np.sort(np.unique([int(s.split("_")[0]) for s in np.loadtxt(file_job_config, dtype=str)]))
    runs_config_file_str = np.array([f"_job_{i}_" for i in runs_config_file_str])
    tmp_files_results = glob.glob(os.path.join(root_results, "*"))
    for file in tmp_files_results:
    
        file_is_in_job = False
        for string in runs_config_file_str:
            if string in file:
                file_is_in_job = True
        
        # Then we delete runs not in job config
        if not file_is_in_job:
            subprocess.run(f"rm {file}", shell=True)
        
    ##########################
    # Merging the dictionaries
    # Keep only non-repeated runs
    total_runs = np.unique(total_runs)
    
    # We create a empty dict for each run we have information
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
            "final_scaling": {}, "delta_final_scaling": {}, "final_scaling_interpolated": {}, "interpolation" : {},
        }
        dict_runs[run] = tmp
        
    # Now we fill this dicts one by one with the empty one
    for d in dictionaries:
        run = d["run"]
        dict_runs[run] = utils.merge_dicts(dict_runs[run], d)

    #####################
    # Checking statistics
    # We don"t trust the fit for subruns with too few events or in which the fit have not suceeded
    # In tose cases we will apply as the final scaling the average with the neighbors.
    for run in dict_runs.keys():
        # The dictionary of one run
        dict_run = dict_runs[run]
        # Statistics object (dict)
        dict_stats = dict_run["statistics"]
        last_srun = max(dict_stats.keys())
    
        # We check subrun by subrun
        for srun in np.sort(list(dict_stats.keys())):
            stats = dict_stats[srun]
            flag = dict_run["flag_error"][srun]
    
            if stats < statistics_threshold or flag:
                logger.warning(f"\nFor run {run} subrun {srun}:\nN_events = {stats}\nFlag_error = {flag}\nSo interpolating neighbors.")
    
                # Search for the right neighbor until a valid one is found
                right_neighbor = srun + 1
                while right_neighbor <= last_srun and (dict_run["statistics"][right_neighbor] < statistics_threshold or dict_run["flag_error"][right_neighbor]):
                    right_neighbor += 1
    
                # Search for the left neighbor until a valid one is found
                left_neighbor = srun - 1
                while left_neighbor >= 0 and (dict_run["statistics"][left_neighbor] < statistics_threshold or dict_run["flag_error"][left_neighbor]):
                    left_neighbor -= 1
    
                # Case of two invalid subruns in a row
                if right_neighbor <= last_srun and left_neighbor >= 0:
                    dict_runs[run]["final_scaling"][srun] = (dict_runs[run]["final_scaling"][left_neighbor] + dict_runs[run]["final_scaling"][right_neighbor]) / 2
                    dict_runs[run]["delta_final_scaling"][srun] = (dict_runs[run]["delta_final_scaling"][left_neighbor] + dict_runs[run]["delta_final_scaling"][right_neighbor]) / 2
                elif right_neighbor <= last_srun:
                    dict_runs[run]["final_scaling"][srun] = dict_runs[run]["final_scaling"][right_neighbor]
                    dict_runs[run]["delta_final_scaling"][srun] = dict_runs[run]["delta_final_scaling"][right_neighbor]
                elif left_neighbor >= 0:
                    dict_runs[run]["final_scaling"][srun] = dict_runs[run]["final_scaling"][left_neighbor]
                    dict_runs[run]["delta_final_scaling"][srun] = dict_runs[run]["delta_final_scaling"][left_neighbor]
                else:
                    logger.warning(f"No valid neighbors found for run {run} subrun {srun}. Unable to interpolate.")


    #######################
    # Reading the datacheck
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
    
    ##########################################
    # Then calculating the interpolated values
    for ir, run_number in enumerate(total_runs):
    
        # logger.info(f"Interpolating... {run_number} -{ir / len(total_runs) * 100:.1f}%")
        dict_results = dict_runs[run_number]
    
        x_fit = np.cumsum(dict_dchecks[run_number]["time"]["srunwise"]["telapsed"])
        y_fit = np.array([dict_results["final_scaling"][srun] for srun in np.sort(list(dict_results["final_scaling"].keys()))])
        yerr_fit = np.array([dict_results["delta_final_scaling"][srun] for srun in np.sort(list(dict_results["final_scaling"].keys()))])
        
        nan_mask = ~(np.isnan(x_fit) | np.isnan(y_fit) | np.isnan(yerr_fit))
        x_fit_masked = x_fit[nan_mask]
        y_fit_masked = y_fit[nan_mask]
        yerr_fit_masked = yerr_fit[nan_mask]
        
        # Performing the fit
        params, pcov, info, _, _ = curve_fit(
            f     = geom.straight_line,
            xdata = x_fit_masked,
            ydata = y_fit_masked,
            sigma = yerr_fit_masked,
            p0    = [1, 0],
            full_output = True,
        )
            
        intercept       = params[0]
        slope           = params[1]
        delta_intercept = np.sqrt(pcov[0, 0])
        delta_slope     = np.sqrt(pcov[1, 1])
        _chi2           = np.sum(info["fvec"] ** 2)
        pvalue          = 1 - chi2.cdf(_chi2, len(x_fit_masked))
        
        dict_results["interpolation"] = {
            "chi2" : _chi2,      
            "pvalue" : pvalue,         
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
    
        dict_fname = os.path.join(root_final_results, f"results_job_{run_number}.pkl")

        ####################
        # Storing the object
        with open(dict_fname, "wb") as f:
            pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)

    ######################################
    # Cleaning the temporal results folder
    for file in glob.glob(os.path.join(root_results, "*")):
        command_rm = f"rm {file}"
        subprocess.run(command_rm, shell=True)

    
def main_final(input_str, simulate_data=False):

    ################
    # First calculus
    # Extracting the run number from the input string
    run_number   = int(input_str.split("_")[0])
    first_last_srun = [int(s) for s in input_str.split("_")[1:]]
    if len(first_last_srun) == 1:
        srun_numbers = np.array(first_last_srun)
    else:
        srun_numbers = np.arange(first_last_srun[0], first_last_srun[1] + 1)
    
    # Creating and storing a configuration file for lstchain processes
    utils.configure_lstchain(config_file)

    # Create the paths that do not exist
    for path in [os.path.join(root_data, "dl1_scaled", f"{run_number:05}"]:
        os.makedirs(os.path.join(path), exist_ok=True)
        
    ####################
    # Reading datachecks
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
    dict_dchecks = lstpipeline.add_rf_node(dict_dchecks, root_rfs, dict_source)

    # Pivot intensity for decorrelation
    ref_intensity = (limits_intensity[0] * limits_intensity[1]) ** 0.5
    logger.info(f"The intensity in the middle of the intensity range is {ref_intensity:.1f} p.e.")
    
    ########################################################
    # Reading the binning from the datacheck ---------------
    # Opening the corresponding datacheck
    fname_dcheck = dict_dchecks[run_number]["dchecks"]["runwise"]
    tab_dcheck_run = tables.open_file(fname_dcheck)
    
    # Read the binning from the datacheck of the first subrun
    dcheck_intensity_binning = np.array(tab_dcheck_run.root.dl1datacheck.histogram_binning.col("hist_intensity")[0])
    # Calculating the logarithmic center of each bin
    dcheck_intensity_binning_centers = (dcheck_intensity_binning[:-1] * dcheck_intensity_binning[1:]) ** 0.5
    # Calculating the width of each bin
    dcheck_intensity_binning_widths = np.diff(dcheck_intensity_binning)
    tab_dcheck_run.close()
    
    # Mask for the fitting region in the fits
    mask_dcheck_bins_fit = (
        (dcheck_intensity_binning_centers >= limits_intensity[0]) &
        (dcheck_intensity_binning_centers <= limits_intensity[1])
    )

    ##########################################################
    # Reading the histogram from the datacheck ---------------
    # Opening the corresponding datacheck
    dcheck_hist_intensities = []
    for fname_dcheck_srun in dict_dchecks[run_number]["dchecks"]["srunwise"]:
        tab_dcheck_srun = tables.open_file(fname_dcheck_srun)
        dcheck_hist_intensities.append(np.array(tab_dcheck_srun.root.dl1datacheck.cosmics.col("hist_intensity")))
        tab_dcheck_srun.close()
    
    # Converting from counts to rate per intensity unit (non-binning dependent quantity)
    dcheck_rates       = [] # Array of histogram of rates for each subrun
    dcheck_delta_rates = [] # The statistical error
    for srun, dcheck_hist_intensity in enumerate(dcheck_hist_intensities):
    
        effective_time_srun = dict_dchecks[run_number]["time"]["srunwise"]["telapsed"][srun]
        
        dcheck_rates.append(              dcheck_hist_intensity  / effective_time_srun / dcheck_intensity_binning_widths)
    
        dcheck_delta_rates.append(np.sqrt(dcheck_hist_intensity) / effective_time_srun / dcheck_intensity_binning_widths)

    ###################
    # Zenith correction
    corr_factor_p0 = geom.pol2(1, p0a, p0b, p0c) / geom.pol2(np.cos(np.deg2rad(dict_dchecks[run_number]["pointing"]["zd"])), p0a, p0b, p0c)
    corr_factor_p1 = geom.pol2(1, p1a, p1b, p1c) - geom.pol2(np.cos(np.deg2rad(dict_dchecks[run_number]["pointing"]["zd"])), p1a, p1b, p1c)
    
    # Zenith correction of the reference (putting the reference in the zenith of the determined subrun)
    corr_ref_p0 = ref_p0 / corr_factor_p0
    corr_ref_p1 = ref_p1 - corr_factor_p1

    ####################################################
    # Parameters to input to the find_scaling() function
    other_parameters = {
        "srun_numbers" : srun_numbers,
        "dict_dchecks" : dict_dchecks,
        "ref_intensity" : ref_intensity,
        "dcheck_intensity_binning" : dcheck_intensity_binning,
        "dcheck_intensity_binning_widths" : dcheck_intensity_binning_widths,
        "dcheck_intensity_binning_centers" : dcheck_intensity_binning_centers,
        "mask_dcheck_bins_fit" : mask_dcheck_bins_fit,
        "corr_factor_p0" : corr_factor_p0, "corr_factor_p1" : corr_factor_p1,
        "root_sub_dl1" : root_sub_dl1,
        "dir_dl1b_scaled" : dir_dl1b_scaled,
        "lims_intensity" : lims_intensity,
        "lims_intensity_extended" : lims_intensity_extended,
        "config_file" : config_file,
        "ref_p0" : ref_p0, "ref_p1" : ref_p1,
    }

    ##############################################################################################################
    # Scaling the final files and also saving the final results dictionary with all the information o fthe process
    dict_fname = os.path.join(root_final_results, f"results_job_{run_number}.pkl")
    
    # Reading the object
    with open(dict_fname, "rb") as f:
        dict_results = pickle.load(f)
    
    dict_results = utils.find_scaling(
        iteration_step="final", dict_results=dict_results, other_parameters=other_parameters, simulated=simulate_data
    )
    
    # Saving the object again
    with open(dict_fname, "wb") as f:
        pickle.dump(dict_results, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    function_name = sys.argv[1]
    
    # Call the appropriate function based on the provided function name
    if function_name == "init":
        input_str = sys.argv[2]
        main_init(input_str)
    elif function_name == "merge":
        main_merge()
    elif function_name == "final":
        input_str = sys.argv[2]
        main_final(input_str)
    elif function_name == "irfs":
        main_irf_creation()
    else:
        print(f"Unknown function: {function_name}\nOptions: irfs, init, merge, final")
        sys.exit(1)