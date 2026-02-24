import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import metview as mv

########################################################################################################################################
# CODE DESCRIPTION
# 13_map_point_grid_ff.py creates interactive map plots of the point and gridded accumulated flash flood reports over a specific accumulation period.

# Usage: python3 13_map_point_grid_ff.py

# Runtime: negligible.

# Author: Fatima M. Pillosu <fatima.pillosu@ecmwf.int> | ORCID 0000-0001-8127-0990
# License: Creative Commons Attribution-NonCommercial_ShareAlike 4.0 International

# INPUT PARAMETERS DESCRIPTION
# the_date_start (date, in YYYYMMDDHH format): start of the accumulation period to consider.
# mask_domain (list of floats, in S/W/N/E coordinates): domain's coordinates.
# type_plot (string): type of plot. Valid values are "interactive" to open a Metview window or "file" to save the map as a jpeg. 
# git_repo (string): repository's local path.
# file_in_mask (string): relative path of the file containing the domain's mask.
# dir_in_point (string): relative path of the directory containing the point accumulated flash flood reports.
# dir_in_grid (string): relative path of the directory containing the gridded accumulated flash flood reports.
# dir_out (string): relative path of the directory containing the map plot with the point and gridded accumulated flash flood reports. Used only when "type_plot = file".

# INPUT PARAMETERS
the_date_start = datetime(2021,9,1,0)
mask_domain = [22,-130,52,-60]
type_plot = "file"
git_repo = "/ec/vol/ecpoint_dev/mofp/phd/probability_of_flash_flood"
file_in_mask = "data/raw/mask/usa_era5.grib"
dir_in_point = "data/processed/02_point_acc_reports_ff"
dir_in_grid = "data/processed/03_grid_acc_reports_ff"
dir_out = "data/plot/13_map_point_grid_ff"
########################################################################################################################################


print()
the_date_final = the_date_start + timedelta(hours=24)
print("Plotting the map with the point and gridded accumulated flash flood reports over the 24-hourly period ending")
print(" - on " + the_date_final.strftime("%Y-%m-%d") + " at " + the_date_final.strftime("%H") + " UTC")

# Converting the domain's mask into a geopoint to plot the actual location of the grid-points within the domain
mask = mv.read(git_repo + "/" + file_in_mask)
mask_vals = mv.values(mask)
mask_lats = mv.latitudes(mask)
mask_lons = mv.longitudes(mask)
mask_lats = mv.filter(mask_lats, mask_vals == 1)
mask_lons = mv.filter(mask_lons, mask_vals == 1)
mask_vals = mv.filter(mask_vals, mask_vals == 1) - 2 # to assign the value -1 to the domain's grid points 
mask_geo = mv.create_geo(type = "xyv",
      latitudes = mask_lats,
      longitudes = mask_lons,
      values = mask_vals
      )

# Creating the geopoints containing the point flash flood reports
file_in_point = git_repo + "/" + dir_in_point + "/" + the_date_final.strftime("%Y") + "/point_acc_reports_ff_" + the_date_final.strftime("%Y%m%d%H") + ".csv"
point = pd.read_csv(file_in_point)
lats_point = point["AREA_AFFECTED_CENTRE_LAT"].values
lons_point = point["AREA_AFFECTED_CENTRE_LON"].values
point_geo = mv.create_geo(type = "xyv",
                  latitudes = lats_point,
                  longitudes = lons_point,
                  values = np.zeros(len(lats_point))
                  )

# Creating the geopoints containing the gridded flash flood reports
file_in_grid = git_repo + "/" + dir_in_grid + "/" + the_date_final.strftime("%Y") + "/grid_acc_reports_ff_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H")  + ".grib"
grid = mv.read(file_in_grid)
ind_grid = np.where(mv.values(grid) > 0)[0]
lats_grid = mv.latitudes(grid)[ind_grid]
lons_grid = mv.longitudes(grid)[ind_grid]
grid_geo = mv.create_geo(type = "xyv",
                  latitudes = lats_grid,
                  longitudes = lons_grid,
                  values = np.zeros(len(lats_grid))
                  )

# Plotting the map with the point and gridded accumulated flash flood reports 
coastlines = mv.mcoast(
      map_coastline_colour = "charcoal",
      map_coastline_thickness = 2,
      map_coastline_resolution = "full",
      map_coastline_sea_shade = "on",
      map_coastline_sea_shade_colour = "rgb(0.665,0.9193,0.9108)",
      map_boundaries = "on",
      map_boundaries_colour = "charcoal",
      map_boundaries_thickness = 4,
      map_grid_latitude_increment = 10,
      map_grid_longitude_increment = 20,
      map_label_right = "off",
      map_label_top = "off",
      map_label_colour = "charcoal",
      map_grid_thickness = 1,
      map_grid_colour = "charcoal",
      map_label_height = 0.7
      )

geo_view = mv.geoview(
      map_projection = "epsg:3857",
      map_area_definition = "corners",
      area = mask_domain,
      coastlines = coastlines
      )

symb_point = mv.msymb(
      legend = "off",
      symbol_type = "marker",
      symbol_table_mode = "on",
      symbol_outline = "on",
      symbol_min_table = [-0.1],
      symbol_max_table = [0.1],
      symbol_colour_table = "red",
      symbol_marker_table = 15,
      symbol_height_table = 0.2
      )

symb_grid = mv.msymb(
      legend = "off",
      symbol_type = "marker",
      symbol_table_mode = "on",
      symbol_outline = "on",
      symbol_min_table = [-0.1],
      symbol_max_table = [0.1],
      symbol_colour_table = "black",
      symbol_marker_table = 15,
      symbol_height_table = 0.3
      )

symb_gridpoints = mv.msymb(
      legend = "off",
      symbol_type = "marker",
      symbol_table_mode = "on",
      symbol_outline = "on",
      symbol_min_table = [-1.1],
      symbol_max_table = [-0.9],
      symbol_colour_table = "rgb(0.8,0.8,0.8)",
      symbol_marker_table = 15,
      symbol_height_table = 0.1
      )

title = mv.mtext(
      text_line_count = 2,
      text_line_1 = "Point and gridded flash flood reports accumulated over the 24-hourly period ending on " + the_date_final.strftime("%Y-%m-%d") + " at " + the_date_final.strftime("%H") + " UTC",
      text_line_2 = " ",
      text_colour = "charcoal",
      text_font = "arial",
      text_font_size = 0.6
      )

# Saving or displaying the map plot
if type_plot == "file":
      dir_out_temp = git_repo + "/" + dir_out
      if not os.path.exists(dir_out_temp):
            os.makedirs(dir_out_temp)
      file_out = dir_out_temp + "/map_point_grid_ff_" + the_date_final.strftime("%Y%m%d") + "_" + the_date_final.strftime("%H")
      png = mv.png_output(output_width = 5000, output_name = file_out)
      mv.setoutput(png)
# mv.plot(geo_view, mask_geo, symb_gridpoints, grid_geo, symb_grid, point_geo, symb_point, title)
mv.plot(geo_view, mask_geo, point_geo, symb_point, title)