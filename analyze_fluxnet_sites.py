#!/usr/bin/env python

import sys, os
import numpy as np

#===============================================================================
# This script formats ECMWF weather data stored on capegrim 
def main():
#===============================================================================
    from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
#-------------------------------------------------------------------------------
# Define general working directories
    EUROSTATdir = '../observations/EUROSTAT_data/'
#-------------------------------------------------------------------------------
# we read the CGMS grid cells coordinates from file
    CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']
#-------------------------------------------------------------------------------
# list FluxNet sites longitude and latitude data, to retrieve their
# corresponding CGMS grid cell ID

    flux_lat = [50.5522,50.8929,43.5496,48.8442,43.4965,40.5238,51.9536]
    flux_lon = [4.7448, 13.5225, 1.1061, 1.9519, 1.2379,14.9574, 4.9029]
    flux_nam = ['BE-Lon','DE-Kli','FR-Aur','FR-Gri','FR-Lam','IT-BCi','NL-Lan']

#-------------------------------------------------------------------------------
# find closest CGMS grid cell for all sites:

    flux_gri = list()
    for i,site in enumerate(flux_nam):
        lon = flux_lon[i]
        lat = flux_lat[i]
        # compute the distance to site for all CGMS grid cells
        dist_list = list()
        for j,grid_no in enumerate(all_grids):
            distance = ((lons[j]-lon)**2. + (lats[j]-lat)**2.)**(1./2.)
            dist_list += [distance] 
        # select the closest grid cell
        indx = np.argmin(np.array(dist_list))
        flux_gri += [all_grids[indx]]

        print 'FluxNet site %s: closest grid cell is %i'%(site, all_grids[indx])

#-------------------------------------------------------------------------------
# find index of grid cell in the arable land list (useful if we want to e.g.
# retrieve CGMS input data only for those grid cells, still using the script
# 02a)

    # we select the grid cells with arable land from file
    europ_arable = get_list_CGMS_cells_in_Europe_arable(all_grids, lons, lats)
    europ_arable = sorted(europ_arable)

    for i,site in enumerate(flux_nam):
        for j,grid_no in enumerate([g for g,a in europ_arable]):
            if grid_no == flux_gri[i]:
                print 'The arable grid cell ID of %s is %i'%(site,j)

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
