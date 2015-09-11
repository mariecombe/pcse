#!/usr/bin/env python

import sys, os
import numpy as np

#===============================================================================
# This script uses WOFOST runs to simulate the carbon fluxes during the growing
# season: we use radiation data to have a diurnal cycle and we add 
# heterotrophic respiration
def main():
#===============================================================================
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
    global ecmwfdir1, ecmwfdir2, ecmwfdir3
#-------------------------------------------------------------------------------
# User-defined:
    crop_no = 3
    gridcell_no = 52054
    lat = 55. # degrees N of the CGMS grid cell
    lon = 0. # degrees E of the CGMS grid cell
    soil_no = 340001
    opti_year = 2000

#-------------------------------------------------------------------------------
# We define working directories

    # ecmwfdir1/year/month contains: cld, convec, mfuv, mfw, q, sp, sub, t, tsp
    ecmwfdir1 = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/glb100x100'

    # ecmwfdir2/year/month contains: blh, ci, cp, d2m, ewss, g10m, lsp, nsss, 
    # sd, sf, skt, slhf, src, sshf, ssr, ssrd, sst, str, strd, swvl1, t2m, u10m,
    # v10m
    ecmwfdir2 = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/sfc/glb100x100/'

    # ecmwfdir3/year/month contains: albedo, sr, srols, veg
    ecmwfdir3 = '/Storage/TM5/METEO/tm5-nc/ec/ei/an0tr6/sfc/glb100x100/'

#-------------------------------------------------------------------------------
#   WE NEED TO LOOP OVER THE GRID CELLS
#-------------------------------------------------------------------------------
# We open the incoming surface shortwave radiation [W.m-2] 

    rad = retrieve_ecmwf_ssrd(opti_year, lon, lat)

#-------------------------------------------------------------------------------
# We open the surface temperature record

#-------------------------------------------------------------------------------
#       WE NEED TO LOOP OVER THE SOIL TYPE
#-------------------------------------------------------------------------------
# We open the WOFOST results file

    filelist = 'pcse_output_c%i_g%i_s%i_y%i'\
                %(crop_no, gridcell_no, soil_no, opti_year) 
    wofost_data = open_csv(currentdir, [filelist], convert_to_float=True)

#-------------------------------------------------------------------------------
# We apply the short wave radiation diurnal cycle on the GPP and R_auto

#-------------------------------------------------------------------------------
# We calculate the heterotrophic respiration with the surface temperature

#-------------------------------------------------------------------------------
# We store the growing season's C fluxes for each grid x soil combi

# crop_no, year, lon_ecmwf, lat_ecmwf, grid_no, soil_no, GPP, R_auto_, R_hetero  

#-------------------------------------------------------------------------------
#   END OF THE TWO LOOPS
#-------------------------------------------------------------------------------

#===============================================================================
# function that will retrieve the incoming surface shortwave radiation from the
# ECMWF data (ERA-interim). It will return two arrays: one of the time in
# seconds since 1st of Jan, and one with the ssrd variable in W.m-2.
def retrieve_ecmwf_ssrd(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    ssrd = np.array([])
    time = np.array([])

    for month in range (1,13):
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssrd_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir2,'%i/%02d'%(year,month),
                                                             namefile))==False):
                continue
            pathfile = os.path.join(ecmwfdir2,'%i/%02d'%(year,month),namefile)
            f = cdf.Dataset(pathfile)

            # retrieve closest latitude and longitude index of desired location
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the shortwave downward surface radiation at that location 
            #print f.variables['ssrd'] # to get the dimensions of the variable
            ssrd = np.append(ssrd, f.variables['ssrd'][0:8, latindx, lonindx])
            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]
            # NB: the file has 8 time steps (3-hourly)
            time = np.append(time, f.variables['time'][:] - convtime)  
            f.close()

    if (len(time) < 2920):
        print '!!!WARNING!!!'
        print 'there are less than 365 days of data that we could retrieve'
        print 'check the folder %s for year %i'%(ecmwfdir2, year)
 
    return time, ssrd

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
