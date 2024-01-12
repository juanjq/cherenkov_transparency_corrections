# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob
import pandas as pd

# Importing custom utility functions
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux

# Root path of this script
root = os.getcwd() + "/"
# Objects directory
root_objects = root + "objects/"

# Directory of all the night-wise datachecks
root_dchecks = "/fefs/aswg/workspace/abelardo.moralejo/data/datachecks/night_wise/DL1_datacheck_"
# Weather station file
ws_database = root_objects + "WS2003-22_short.h5"

# Some filenames -------------------
# Filename of the datacheck dictionary
fname_datacheck_dict = root_objects + "datacheck_dict.pkl"
# Job list file
fname_job_list = root_objects + "bash_job_list.txt"
# Filename of the relation between run and night
fname_run_night_relation = root_objects + "ws_run_relation.txt"

def run(string):

    # Extracting the initial and final run
    init = int(string.split(",")[0])
    end  = int(string.split(",")[1])    

    # Reading the datacheck dictionary
    with open(fname_datacheck_dict, 'rb') as f:
        dict_dcheck = pickle.load(f) 

    # Loading the weather station database
    df_ws = pd.read_hdf(ws_database)

    # Loading the timestamp of each entry in the datacheck dictionary
    dates_dcheck = dict_dcheck["time"]

    # Getting the min and max dates
    maxdate = np.max(dates_dcheck)
    mindate = np.min(dates_dcheck)

    # Converting the weather station dates to datetime objects
    dates_ws = np.array([datetime.fromisoformat(str(d).split(".")[0]) for d in df_ws.index])

    # Getting the max date of the weather station
    maxdate_ws = np.max(dates_ws)

    # Masking the weather station data to the min and max dates of the datacheck dictionary
    mask_dates  = (dates_ws > mindate) & (dates_ws < maxdate)

    # Masking also for day data, i.e. sun_alt > 0 we are not interested in 
    mask_night = (df_ws["sun_alt"] < 0)

    total_mask = (mask_dates & mask_night)

    dates_ws = dates_ws[total_mask]
    df_ws    = df_ws[total_mask]

    print("Starting to iterate")
    
    ws_entry  = []
    run, srun = [], []
    for j, date_dcheck in enumerate(dict_dcheck["time"]):

        if j >= init and j <= end:    

            if date_dcheck > maxdate_ws:
                ws_entry.append(None)
                
                run.append(dict_dcheck["run"][j])
                srun.append(dict_dcheck["subrun"][j])

            else:
                str_id = str(df_ws.iloc[np.argmin(np.abs(dates_ws - date_dcheck))].name)

                ws_entry.append(str_id)
                run.append(dict_dcheck["run"][j])
                srun.append(dict_dcheck["subrun"][j])

    print("Writting...")
    with open(fname_run_night_relation, "a") as f:
        for r, s, e in zip(run, srun, ws_entry):
            f.write(f"{r}-{s},{e}\n")
                    
if __name__ == "__main__":
    input = sys.argv[1]
    run(input)