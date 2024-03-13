import numpy as np
import json, sys, os
from scipy.optimize import curve_fit
from scipy.stats import chi2
import subprocess

from lstchain.io.config import get_standard_config
import tables

# Other auxiliar scripts
sys.path.insert(0, os.path.join(os.getcwd(), "../scripts/"))
import geometry as geom

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def configure_lstchain(config_file):
    """
    Creates a file of standard configuration for the lstchain analysis. 
    It can be changed inside this function
    """
    
    dict_config = get_standard_config()

    #######################################################
    # Configure here:
    #######################################################
    
    # We select the heuristic flatfield option in the standard configuration
    dict_config["source_config"]["LSTEventSource"]["use_flatfield_heuristic"] = True

    dict_config["events_filters"] = {
       "intensity": [50, np.inf],
       "width": [0, np.inf],
       "length": [0, np.inf],
       "r": [0, 1],
       "wl": [0.01, 1],
       "leakage_intensity_width_2": [0, 1],
       "event_type": [32, 32]
    }

    dict_config["DL3Cuts"] ={
         "min_event_p_en_bin": 100,
         "global_gh_cut": 0.7,
         "gh_efficiency": 0.7,
         "min_gh_cut": 0.1,
         "max_gh_cut": 0.98,
         "global_alpha_cut": 10,
         "global_theta_cut": 0.2,
         "theta_containment": 0.7,
         "alpha_containment": 0.7,
         "min_theta_cut": 0.1,
         "max_theta_cut": 0.32,
         "fill_theta_cut": 0.32,
         "min_alpha_cut": 1,
         "max_alpha_cut": 20,
         "fill_alpha_cut": 20,
         "allowed_tels": [1]
    }

    dict_config["DataBinning"] = {
      "true_energy_min": 0.005,
      "true_energy_max": 500,
      "true_energy_n_bins": 25,
      "reco_energy_min": 0.005,
      "reco_energy_max": 500,
      "reco_energy_n_bins": 25,
      "energy_migration_min": 0.2,
      "energy_migration_max": 5,
      "energy_migration_n_bins": 30,
      "fov_offset_min": 0.1,
      "fov_offset_max": 1.1,
      "fov_offset_n_edges": 9,
      "bkg_fov_offset_min": 0,
      "bkg_fov_offset_max": 10,
      "bkg_fov_offset_n_edges": 21,
      "source_offset_min": 0,
      "source_offset_max": 1,
      "source_offset_n_edges": 101
    }
    
    #######################################################

    
    with open(config_file, "w") as json_file:
        json.dump(dict_config, json_file)

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
    
def find_scaling(iteration_step, dict_results, other_parameters, simulated=False):
    """
    A function to perform scaling and evaluating the results. Returning everything in a updated dictionary

    Input:
    - iteration_step: (str) 
        The iteration step you are in, that can be "original" for the original data, "upper" for the upper
        limit on the scale factor, "linear" for the linear intepolation factor and "final" for the final scaling 
        and results.
        
    - dict_results: (dict)
        Dictionary with the results of the before step.
        
    - simulated: (bool)
        If instead of scaling the data, random data is generated just to fill the values. Short to run tests.

    - other_parameters (dict)
        A dictionary with all other needed parameters:
        * "srun_numbers"
        * "dict_dchecks"
        * "ref_intensity"
        * "dcheck_intensity_binning"
        * "dcheck_intensity_binning_widths"
        * "dcheck_intensity_binning_centers"
        * "mask_dcheck_bins_fit"
        * "corr_factor_p0"
        * "corr_factor_p1"
        * "root_sub_dl1"
        * "dir_dl1b_scaled"
        * "limits_intensity"
        * "limits_intensity_extended"
        * "config_file"
        * "ref_p0"
        * "ref_p1"
    """

    # Reading the other variables dictionary
    srun_numbers = other_parameters["srun_numbers"]
    dict_dchecks = other_parameters["dict_dchecks"]
    ref_intensity = other_parameters["ref_intensity"]
    dcheck_intensity_binning = other_parameters["dcheck_intensity_binning"]
    dcheck_intensity_binning_widths = other_parameters["dcheck_intensity_binning_widths"]
    dcheck_intensity_binning_centers = other_parameters["dcheck_intensity_binning_centers"]
    mask_dcheck_bins_fit = other_parameters["mask_dcheck_bins_fit"]
    corr_factor_p0 = other_parameters["corr_factor_p0"]
    corr_factor_p1 = other_parameters["corr_factor_p1"]
    root_sub_dl1 = other_parameters["root_sub_dl1"]
    dir_dl1b_scaled = other_parameters["dir_dl1b_scaled"]
    limits_intensity = other_parameters["limits_intensity"]
    limits_intensity_extended = other_parameters["limits_intensity_extended"]
    config_file = other_parameters["config_file"]
    ref_p0 = other_parameters["ref_p0"]
    ref_p1 = other_parameters["ref_p1"]
    
    # Creating a arrray of subruns looking at the datachecks and also extracting the run number
    run_number  = dict_results["run"]

    # Empty arrays to store the fit information
    data_p0, data_delta_p0 = [], []
    data_p1, data_delta_p1 = [], []
    data_chi2, data_pvalue = [], []
    
    # Processing subrun by subrun---------------------------------------------------------------
    for srun in srun_numbers:    

        # Reading dl1
        #################################################
        input_fname = dict_dchecks[run_number]["dl1a"]["srunwise"][srun]   # Input dl1a 
        data_scale_factor = dict_results["scaled"][iteration_step][srun]   # Reading the scaling factor

        # Here we do different things depending on the iteration step
        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the first one i.e. == "original"
        # We do not run lstchain_dl1ab because the data is already scaled
        if iteration_step == "original":
            data_output_fname = input_fname

        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the second or third: "upper" or "linear"
        # We perform lstchain_dl1ab but over a subset of the data only to keep it shorter
        elif iteration_step in ["upper", "linear"]:

            # Temporal dl1 file that will be overwritten in the next iteration / subrun
            data_output_fname = os.path.join(root_sub_dl1, f"tmp_dl1_srunwise_run{run_number}_srun{srun}_{iteration_step}_scaled.h5")

            logger.info(f"\nProcessing subrun {srun}")

            # If scale is greater than 1 we select a range lower than the upper one
            # otherwise we select a range higher than the upper one
            if data_scale_factor > 1:
                dl1_selected_range = f"{limits_intensity_extended:.2f},{limits_intensity[1]:.2f}"
            else:
                dl1_selected_range = f"{limits_intensity[0]:.2f},inf"

            if not simulated:
                logger.info(f"Running lstchain_dl1ab... scale: {data_scale_factor:.2f}")
                # If the file already exists we delete it
                if os.path.exists(data_output_fname):
                    os.remove(data_output_fname)
                
                command_dl1ab = f"lstchain_dl1ab --input-file {input_fname} --output-file {data_output_fname} --config {config_file}"
                command_dl1ab = command_dl1ab + f" --no-image --light-scaling {data_scale_factor} --intensity-range {dl1_selected_range}"
                logger.info(command_dl1ab)

                subprocess.run(command_dl1ab, shell=True)
                # # We add an exception because sometimes can fail...
                # ntries = 3
                # while ntries > 0:
                #     try:
                #         ntries = ntries - 1
                #         subprocess.run(command_dl1ab, shell=True)
                #     except Exception as e:
                #         logger.error(f"Failed to run {command_dl1ab} with error: {repr(e)}")
        
        # ////////////////////////////////////////////////////////////
        # ////////////////////////////////////////////////////////////
        # If is the last step i.e. "final"
        # The lstchain_dl1ab script is run over all thedataset to generate the final file
        elif iteration_step == "final":

            data_output_fname = os.path.join(dir_dl1b_scaled, f"{run_number:05}", os.path.basename(dict_dchecks[run_number]["dl1a"]["srunwise"][srun]))
            logger.info(f"\nProcessing subrun {srun}")

            if not simulated:
                logger.info(f"Running lstchain_dl1ab... scale: {data_scale_factor:.2f}")
                # If the file already exists we delete it
                if os.path.exists(data_output_fname):
                    os.remove(data_output_fname)

                data_scale_factor = 1.3
                
                command_dl1ab = f"lstchain_dl1ab --input-file {input_fname} --output-file {data_output_fname} --config {config_file}"
                command_dl1ab = command_dl1ab + f" --no-image --light-scaling {data_scale_factor}"
                logger.info(command_dl1ab)

                subprocess.run(command_dl1ab, shell=True)
                
                # # We add an exception because sometimes can fail...
                # ntries = 3
                # while ntries > 0:
                #     try:
                #         ntries = ntries - 1
                #         subprocess.run(command_dl1ab, shell=True)
                #     except Exception as e:
                #         logger.error(f"Failed to run {command_dl1ab} with error: {repr(e)}")
    
            # We store this info also in the dictionary in the final case
            dict_results["filenames"][srun] = data_output_fname

        #################################################################
        # Reading the dl1 file
        #################################################################
        if not simulated:
            table_data = tables.open_file(data_output_fname)
            data_counts_intensity, _ = np.histogram(
                table_data.root.dl1.event.telescope.parameters.LST_LSTCam.col("intensity"), 
                bins=dcheck_intensity_binning
            )
            table_data.close()
        else:
            # Simulated example data where we add random noise
            simdata = np.array([0,0,2,6,12,23,20,15,25,56,105,214,441,694,933,1244,1429,1582,1597,1545,1498,1479,1484,1364,1296,1290,
                                1228,1089,1004,834,732,665,613,529,411,426,307,266,201,191,186,150,114,121,93,87,62,60,38,39,27,31,33,
                                26,22,24,20,13,11,11,8,6,5,8,4,4,5,4,3,1,1,2,1,0,1,1,1,1,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
            
            if iteration_step   == "original":
                data_counts_intensity = simdata * 1.00 + np.random.rand(100) * 100
            elif iteration_step == "upper":
                data_counts_intensity = simdata * 0.30 + np.random.rand(100) * 100
            elif iteration_step == "linear":
                data_counts_intensity = simdata * 0.15 + np.random.rand(100) * 100
            elif iteration_step == "final":
                data_counts_intensity = simdata * 0.18 + np.random.rand(100) * 100
        
        # Calculating the non binning dependent transformation
        effective_time_srun = dict_dchecks[run_number]["time"]["srunwise"]["telapsed"][srun]
        data_rates       = np.array(data_counts_intensity) / effective_time_srun / dcheck_intensity_binning_widths
        data_delta_rates = np.sqrt(data_counts_intensity)  / effective_time_srun / dcheck_intensity_binning_widths


        #################################################################
        # Performing the fit
        #################################################################
        # Displacing the X-coordinates to the center of the fit, in order to decorrelate the fit
        x_fit = dcheck_intensity_binning_centers[mask_dcheck_bins_fit] / ref_intensity
        y_fit = data_rates[mask_dcheck_bins_fit]
        yerr_fit = data_delta_rates[mask_dcheck_bins_fit]
        
        # Trying for the cases where the data is bad and the fit returns an error
        try:
            params, pcov, info, _, _ = curve_fit(
                f     = geom.powerlaw,
                xdata = x_fit,
                ydata = y_fit,
                sigma = yerr_fit,
                p0    = [ref_p0, ref_p1],
                full_output = True,
            )
        
            srun_p0, srun_p1  = params
            srun_delta_p0 = np.sqrt(pcov[0, 0])
            srun_delta_p1 = np.sqrt(pcov[1, 1])
            srun_chi2     = np.sum(info["fvec"] ** 2)
            srun_pvalue   = 1 - chi2.cdf(srun_chi2, sum(mask_dcheck_bins_fit))
            dict_results["flag_error"][srun] = False

        # If the fit is not successful we return nan values
        except RuntimeError:
            logger.error(f"For run {run_number} and subrun {srun}, the fit failed due to RuntimeError.")
            srun_p0, srun_p1  = np.nan, np.nan
            srun_delta_p0 = np.nan
            srun_delta_p1 = np.nan
            srun_chi2     = np.nan
            srun_pvalue   = np.nan
            dict_results["flag_error"][srun] = True
            
        dict_results["chi2"][iteration_step][srun]   = srun_chi2
        dict_results["pvalue"][iteration_step][srun] = srun_pvalue
        dict_results["scaled"][iteration_step][srun] = data_scale_factor
    
        data_p0.append(srun_p0)
        data_p1.append(srun_p1)
        data_delta_p0.append(srun_delta_p0)
        data_delta_p1.append(srun_delta_p1)
        data_chi2.append(srun_chi2)
        data_pvalue.append(srun_pvalue)
    
    # Convert to numpy arrays
    data_p0       = np.array(data_p0)
    data_p1       = np.array(data_p1)
    data_delta_p0 = np.array(data_delta_p0)
    data_delta_p1 = np.array(data_delta_p1)
    data_chi2     = np.array(data_chi2)
    data_pvalue   = np.array(data_pvalue)
  
    # Zenith corrections to the parameters
    #########################################################
    data_corr_p0 = data_p0 * corr_factor_p0
    data_corr_p1 = data_p1 + corr_factor_p1
    
    data_corr_delta_p0 = data_delta_p0 * corr_factor_p0
    data_corr_delta_p1 = data_delta_p1
    
    # Calculating the needed light yield  
    data_light_yield, data_delta_light_yield = geom.calc_light_yield(
        p0_fit = data_corr_p0,
        p1_fit = data_corr_p1, 
        sigma_p0_fit = data_corr_delta_p0, 
        sigma_p1_fit = data_corr_delta_p1, 
        p0_ref = ref_p0,
    )
    # Scalings to apply
    data_scaling       = 1 / data_light_yield
    data_delta_scaling = 1 / data_light_yield ** 4 * data_delta_light_yield
    # The scaling in percentage
    data_scaling_percent       = (data_scaling - 1) * 100
    data_delta_scaling_percent = data_delta_scaling * 100
    
    # Adding to dictionary
    for i, srun in enumerate(srun_numbers):
        dict_results["p0"][iteration_step][srun]       = data_corr_p0[i]
        dict_results["delta_p0"][iteration_step][srun] = data_corr_delta_p0[i]
        dict_results["p1"][iteration_step][srun]       = data_corr_p1[i]
        dict_results["delta_p1"][iteration_step][srun] = data_corr_delta_p1[i]
        
        dict_results["light_yield"][iteration_step][srun]       = data_light_yield[i]
        dict_results["delta_light_yield"][iteration_step][srun] = data_delta_light_yield[i]
        dict_results["scaling"][iteration_step][srun]           = data_scaling[i]
        dict_results["delta_scaling"][iteration_step][srun]     = data_delta_scaling[i]
        dict_results["scaling_percent"][iteration_step][srun]   = data_scaling_percent[i]
        dict_results["delta_scaling_percent"][iteration_step][srun] = data_delta_scaling_percent[i]

    return dict_results
    