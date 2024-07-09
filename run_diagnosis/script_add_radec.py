import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, ICRS
from ctapipe.coordinates import CameraFrame
from lstchain.reco.utils import clip_alt
import sys, os
import pandas as pd

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def main(
    input_file, 
    output_folder, 
    location_lst = EarthLocation.from_geodetic(-17.89139 * u.deg, 28.76139 * u.deg, 2184 * u.m), 
    focal_length_lst = 28 * u.m,
    overwrite = False
):

    logger.info(f"Reading hdf table")
    table_dl2 = pd.read_hdf(input_file, "/dl2/event/telescope/parameters/LST_LSTCam")
    
    # Extracting the run number
    run_number = table_dl2["obs_id"][0]
    logger.info(f"Computing for Run {run_number}")

    # Checking if the file already exists in the output folder
    # Defining the outpuf filename:
    output_file = os.path.join(output_folder, f"dl2_LST-1.Run{run_number:05}_radec.h5")
    file_exists = os.path.isfile(output_file)
    
    if (not file_exists) or overwrite:
    
        # Observation time
        logger.info(f"\nExtracting observation times...")
        obstime_dl2 = pd.to_datetime(table_dl2["dragon_time"], unit="s")
        
        # Coordinate frame
        logger.info(f"Extracting coordinate frame...")
        horizon_frame = AltAz(location=location_lst, obstime=obstime_dl2)
        
        # Telescope pointing
        pointing_alt = u.Quantity(table_dl2["alt_tel"], u.rad, copy=False)
        pointing_az  = u.Quantity(table_dl2["az_tel"], u.rad, copy=False)

        logger.info(f"Defining coordinates...")
        pointing_direction = SkyCoord(alt=clip_alt(pointing_alt), az=pointing_az, frame=horizon_frame)
        logger.info(f"Converting to ICRS...")
        pointing_direction_icrs = pointing_direction.transform_to("icrs")
        
        # Defining the camera coordinates frame
        logger.info(f"Creating camera frame...")
        camera_frame = CameraFrame(
            focal_length=focal_length_lst,
            telescope_pointing=pointing_direction,
            obstime=obstime_dl2,
            location=location_lst,
        )

        logger.info(f"Defining camera coordinates...")
        camera_coords = SkyCoord(x=table_dl2["reco_src_x"], y=table_dl2["reco_src_y"], frame=camera_frame, unit=(u.m, u.m))
        logger.info(f"Converting to ICRS...")
        radec_coords = camera_coords.transform_to(frame=ICRS)

        logger.info(f"Appending to table...")
        # adding the ra-dec coordinates of pointing and events for lst data
        table_dl2.loc[:,"reco_ra"]      = radec_coords.ra.deg
        table_dl2.loc[:,"reco_dec"]     = radec_coords.dec.deg 
        table_dl2.loc[:,"pointing_ra"]  = pointing_direction_icrs.ra.deg
        table_dl2.loc[:,"pointing_dec"] = pointing_direction_icrs.dec.deg    
        
        # setting run-event as indexes
        table_dl2.set_index(["obs_id", "event_id"], inplace=True)
        table_dl2.sort_index(inplace=True)

        # Then we store it with the default key
        logger.info(f"\nStoring in a new file\n--> {output_file}")
        table_dl2.to_hdf(output_file, "/dl1/event/telescope/parameters/LST_LSTCam")

    else:
        print(f"Run {run_number} already found in {output_folder}, skipping...")


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_file, output_folder)