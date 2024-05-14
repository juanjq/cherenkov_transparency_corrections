# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from datetime import datetime
import pickle, json, sys, os, glob
import tables
import pandas as pd
import matplotlib

pd.set_option('display.max_columns', None)

# Importing custom utility functions
sys.path.insert(0, os.getcwd() + "/../scripts/")
import auxiliar as aux
import lstpipeline


def main(string):

    im, iM = string.split(",")
    im = int(im)
    iM = int(iM)
    
    # Root path of this script
    root = os.getcwd() + "/"
    # Objects directory
    root_objects = root + "objects/"
    root_objects_tmp = root_objects + "tmp/"

    fname = root_objects_tmp + f"{im}.pkl"
    
    # Directory of all the night-wise datachecks
    root_dchecks = "/fefs/aswg/workspace/abelardo.moralejo/data/datachecks/night_wise/DL1_datacheck_"
    # Weather station file
    ws_database = root_objects + "WS2003-23.h5"
    
    # Some filenames -------------------
    # Filename of the datacheck dictionary
    fname_datacheck_dict = root_objects + "tmp_datacheck_dict.pkl"
    
    # Flags for computing or not different parts
    # Compute the datacheck dictionary
    compute_datacheck_dict = True
    
    # Create needed folders
    for dir in [root_objects]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    str_dchecks = "datacheck_"
    dl1_root = "/fefs/aswg/data/real/DL1/*/v0.*/tailcut84/"
    main_name = f"{str_dchecks}dl1_LST-1.Run?????"
    total_dl1a_runwise = np.sort(glob.glob(dl1_root + "*/" + f"{main_name}.h5") + glob.glob(dl1_root + f"{main_name}.h5"))
    
    lengroups = 100


    counts = []
    intensity_points = []
    
    tab = tables.open_file(total_dl1a_runwise[0])
    _bin_edges = tab.root.dl1datacheck.histogram_binning.col("hist_intensity")[0]
    tab.close()
    binning = list((_bin_edges[1:] + _bin_edges[:-1]) / 2)
    
    for i, srun_dcheck in enumerate(total_dl1a_runwise[im:iM]):
    
        print(f"Analysing... {i:3}/{lengroups}") if i % 10 == 0 else None
        
        tab = tables.open_file(srun_dcheck)
    
        for inte, time  in zip(tab.root.dl1datacheck.cosmics.col("hist_intensity"), tab.root.dl1datacheck.cosmics.col("elapsed_time")):
            counts = counts + list(np.array(inte) / time)
            intensity_points = intensity_points + binning
    
        tab.close()

    print("Writting...")
    with open(fname, 'wb') as f:
        pickle.dump([intensity_points, counts], f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    input_str = sys.argv[1]
    main(input_str)