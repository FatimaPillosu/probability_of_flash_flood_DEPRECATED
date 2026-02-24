import os
from datetime import datetime, timedelta
import numpy as np
import metview as mv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

###########################################################################################
# CODE DESCRIPTION
# 12_timeseries_point_grid_ff.py plots the daily timeseries per year of the counts of point and gridded 
# accumulated flash flood reports.

# Usage: python3 12_timeseries_point_grid_ff.py

# Runtime: ~ 10 minutes.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# year_s (year, in YYYY format): start year to consider.
# year_f (year, in YYYY format): final year to consider.
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in (string): relative path of the directory containing the accumulated point flash flood reports per grid-box.
# dir_out (string): relative path containing the timeseries plots. 

###########################################################################################
# INPUT PARAMETERS
year_s = 2021
year_f = 2021
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in = "data/processed/03_grid_acc_reports_ff"
dir_out = "data/plot/12_timeseries_point_grid_ff"
###########################################################################################


# Reading the domain's mask
mask = mv.read(git_repo + "/" + file_in_mask)
mask = mv.bitmap(mask,0) # bitmap the values outside the domain

# Plotting the daily timeseries per year of the counts of point and gridded accumulated flash flood reports
print()
print("Plotting the daily timeseries of the counts of point and gridded accumulated flash flood reports for:")
for year in range(year_s, year_f+1):

      print(" - " + str(year))

      # Defining the accumulation periods to consider
      the_date_start_s = datetime(year,1,1)
      the_date_start_f = datetime(year,12,31)
      
      # Initializing the variables that will contain the count of point and gridded accumulated flash flood reports
      dates_all = []
      count_point_ff_all = []
      count_grid_ff_all = []

      # Computing the counts of point and gridded accumulated flash flood reports per accumulation period
      the_date_start = the_date_start_s
      while the_date_start <= the_date_start_f:

            the_date_final = the_date_start + timedelta(hours=24)
            dates_all.append(the_date_final)
            
            # Reading the point and gridded accumulated flash flood reports
            file_in_ff = git_repo + "/" + dir_in + "/" + the_date_final.strftime("%Y") + "/grid_acc_reports_ff_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H") + ".grib"
            if os.path.exists(file_in_ff):
                  
                  point_ff_grib = mv.read(file_in_ff) * mask
                  grid_ff_grib = (point_ff_grib > 0)

                  # Defining the counts of point and gridded accumulated flash flood reports 
                  cound_point_ff_grib = np.nansum(mv.values(point_ff_grib))
                  cound_grid_ff_grib = np.nansum(mv.values(grid_ff_grib))
                  count_point_ff_all.append(cound_point_ff_grib)
                  count_grid_ff_all.append(cound_grid_ff_grib)

            else:

                  # Defining the counts of point and gridded accumulated flash flood reports 
                  count_point_ff_all.append(0)
                  count_grid_ff_all.append(0)

            the_date_start = the_date_start + timedelta(hours=24)
      
      # Plot the histogram with the counts
      fig, ax = plt.subplots(figsize=(25, 8))
      rects1 = ax.bar(dates_all, count_point_ff_all, 0.5, color="red", align='center', label="Point")
      # rects2 = ax.bar(dates_all, count_grid_ff_all, 0.5, color="black", align='center', label="Grid")
      ax.xaxis.set_major_locator(mdates.MonthLocator())
      ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
      plt.xticks(rotation=30)
      ax.set_xlabel("End of 24-h accumulation periods", fontsize=16, labelpad = 15)
      ax.set_ylabel("Counts", fontsize=16, labelpad = 10)
      ax.set_title("Count of point and gridded accumulated flash flood reports in " + str(year), fontsize=18, pad=15, weight = "bold")
      ax.legend(fontsize=14)
      ax.tick_params(axis='both', which='major', labelsize=16)

      # Save the plot
      dir_out_temp = git_repo + "/" + dir_out
      if not os.path.exists(dir_out_temp):
            os.makedirs(dir_out_temp)
      file_out = dir_out_temp + "/timeseries_point_grid_ff_" + str(year) + ".png"
      plt.savefig(file_out, format="jpeg", bbox_inches="tight", dpi=1000)