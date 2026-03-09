import os
from datetime import timedelta
import numpy as np
import pandas as pd

#################################################################################
# CODE DESCRIPTION
# 01_extract_noaa_reports_ff.py extracts the flash flood reports from the NOAA database, and 
# manipulates the raw data to make it suitable for subsequent analysis:
#     - eliminates reports with no lat/lon coordinates or reporting date/time;
#     - expands the datapoints based on the reporting times; creates a new variable "REPORT_DATE".

# Usage: python3 01_extract_noaa_reports_ff.py

# Runtime: ~ 2 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year_s (integer, in the YYYY format): start year to consider.
# year_f (integer, in the YYYY format): final year to consider.
# git_repo (string): repository's local path.
# dir_in (string): relative path of the directory containing the raw NOAA's reports.
# dir_out (string): relative path of the directory containing the extracted flash flood reports.

# INPUT PARAMETERS
year_s = 2021
year_f = 2024
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
dir_in = "data/raw/obs/noaa_reports_ff"
dir_out = "data/processed/01_extract_noaa_reports_ff"
#################################################################################


####################
# COSTUME FUNCTION #
####################

# Function to expand the data rows based on the reports' duration
def expand_row(row):
    date_range = pd.date_range(row['BEGIN_DATE_TIME'], row['END_DATE_TIME'], freq='D')
    expanded = pd.DataFrame([row.values] * len(date_range), columns=row.index)
    expanded['REPORT_DATE'] = date_range
    return expanded
###########################################################################


# Setting the main input/output directories
temp_dir_in = git_repo + "/" + dir_in
temp_dir_out = git_repo + "/" + dir_out 
if not os.path.exists(temp_dir_out):
    os.makedirs(temp_dir_out)

# Creating the variables where to store the total number of reports per year
years_rep = []
num_rep_all = []
num_rep_ff = []
num_rep_ff_withCoord = []

# Post-processing the raw flood reports
print("Creating the cleaned and merged flash flood report database. Post-processing the reports for: ")
ff_withCoord_all = pd.DataFrame()

for Year in range(year_s, year_f+1):
      
    print(" - " + str(Year))
    
    # Reading the raw flood reports
    string2find = "d" + str(Year) + "_"
    FileIN = [f for f in os.listdir(temp_dir_in) if string2find in f and os.path.isfile(os.path.join(temp_dir_in, f))]
    df = pd.read_csv(git_repo + "/" + dir_in + "/" + FileIN[0], low_memory=False)
    num_rows_all = df.shape[0]

    # Extracting reports for flash floods and removing reports with no lat/lon coordinates or no reporting date/time
    ff = df[df["EVENT_TYPE"].isin(["Flash Flood", "Heavy Rain", "Hurricane/Typhoon", "Tropical Storm"])] # "Flash Flood", "Heavy Rain", "Hurricane/Typhoon", "Tropical Storm"
    ff = ff.reset_index(drop=True)  # to reset the indexes of the new dataframe
    num_rows_ff = ff.shape[0]
    ff_withCoord = ff.dropna(subset=["BEGIN_DATE_TIME", "END_DATE_TIME", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"])
    ff_withCoord = ff_withCoord.reset_index(drop=True)  # to reset the indexes of the new dataframe
    num_rows_ff_withCoord = ff_withCoord.shape[0]
    
    # Storing the total number of reports per year
    years_rep.append(Year)
    num_rep_all.append(num_rows_all)
    num_rep_ff.append(num_rows_ff)
    num_rep_ff_withCoord.append(num_rows_ff_withCoord)

    # Creating the cleaned and merged flash flood report database
    if num_rows_ff_withCoord != 0:
 
        # Extracting the reports' lat/lon coordinates
        begin_lat = ff_withCoord["BEGIN_LAT"].to_numpy()
        begin_lon = ff_withCoord["BEGIN_LON"].to_numpy()
        end_lat = ff_withCoord["END_LAT"].to_numpy()
        end_lon = ff_withCoord["END_LON"].to_numpy()

        # Computing the lat/lon coordinates of the centre's area affected, and adding the column to the dataframe
        area_affected_centre_lat = np.round((begin_lat + end_lat) / 2, decimals = 4)
        area_affected_centre_lon = np.round( ( (begin_lon + end_lon) / 2 ) + 360, decimals = 4)  
        ff_withCoord.loc[:, "AREA_AFFECTED_CENTRE_LAT"] = area_affected_centre_lat
        ff_withCoord.loc[:, "AREA_AFFECTED_CENTRE_LON"] = area_affected_centre_lon

        # Extract all the rows from certain columns to streamline the dataset
        reduced_ff_withCoord = ff_withCoord[["EVENT_ID", "STATE", "CZ_TIMEZONE", "SOURCE", "EVENT_TYPE", "FLOOD_CAUSE", "BEGIN_DATE_TIME", "END_DATE_TIME", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON", "AREA_AFFECTED_CENTRE_LAT", "AREA_AFFECTED_CENTRE_LON"]]

        # Merging the reports from all the years
        ff_withCoord_all = pd.concat([ff_withCoord_all, reduced_ff_withCoord], ignore_index=True)

print(f"Total number of flash flood events with lat/lon coordinates between {year_s} and {year_f}: {len(ff_withCoord_all)}")

# Determine the reports' dates
ff_withCoord_all_expanded = pd.concat([expand_row(row) for _, row in ff_withCoord_all.iterrows()], ignore_index=True)
print(f"Total number of 'expanded' flash flood events with lat/lon coordinates between {year_s} and {year_f}: {len(ff_withCoord_all_expanded)}")


# # Saving the database with only flash flood reports and some metadata
# file_out = temp_dir_out + "/noaa_reports_ff.csv"
# ff_withCoord_all_expanded.to_csv(file_out, index=False)
# file_out = temp_dir_out + "/years"
# np.save(file_out, years_rep)
# file_out = temp_dir_out + "/counts_reports_all"
# np.save(file_out, num_rep_all)
# file_out = temp_dir_out + "/counts_reports_ff"
# np.save(file_out, num_rep_ff)
# file_out = temp_dir_out + "/counts_reports_ff_with_coord"
# np.save(file_out, num_rep_ff_withCoord)