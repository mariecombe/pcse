#!/usr/bin/env python

import sys, os
import numpy as np

#===============================================================================
# This script uses WOFOST runs to simulate the carbon fluxes during the growing
# season: we use radiation data to have a diurnal cycle and we add 
# heterotrophic respiration
def main():
#===============================================================================
    from maries_toolbox import open_pcse_csv_output
    from cPickle import load as pickle_load
    import datetime
    import math
#-------------------------------------------------------------------------------
    global ecmwfdir1, ecmwfdir2, ecmwfdir3, folderpickle, pcseouputdir
#-------------------------------------------------------------------------------
# User-defined:

    NUTS_no = 'ES43'
    crop_no = 3
    opti_year = 2006

#-------------------------------------------------------------------------------
# constants for R_hetero
    Eact0 = 53.3e3
    R10 = 0.23
    Cw = 1.6e-3
    wsmax = 0.55
    wsmin = 0.005
#-------------------------------------------------------------------------------
# We define working directories

    # ecmwfdir1/year/month contains: cld, convec, mfuv, mfw, q, sp, sub, t, tsp
    ecmwfdir1 = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/eur100x100'


    # ecmwfdir3/year/month contains: albedo, sr, srols, veg
    ecmwfdir3 = '/Storage/TM5/METEO/tm5-nc/ec/ei/an0tr6/sfc/glb100x100/'

    # storage folder for individual PCSE runs:
    pcseoutputdir = '/Storage/CO2/mariecombe/pcse_individual_output/'
    
    # storage folder for the CGMS input data files
    folderpickle  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'

#-------------------------------------------------------------------------------
# Select the grid cells

	# we first read the list of all 'whole' grid cells contained in that
	# region NB: grid_list_tuples[0] is a list of (grid_cell_id,
	# arable_land_area) tuples, which are already sorted by decreasing
	# amount of arable land
    filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
    grid_list_tuples = pickle_load(open(filename,'rb'))

    # we select all grid cells
    selected_grid_cells = grid_list_tuples[0]
    print '\nWe have selected all', len(selected_grid_cells),'grid cells:',\
              [g for g,a in selected_grid_cells]

#-------------------------------------------------------------------------------
# Select the soil types

    # we first read the list of suitable soil types for our chosen crop 
    filename   = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no
    suit_soils = pickle_load(open(filename,'rb')) 

    selected_soil_types = {}

    for grid, arable_land in selected_grid_cells:

        # we read the entire list of soil types contained within the grid
        # cell
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soils    = pickle_load(open(filename,'rb'))

        # We select all of them
        selected_soil_types[grid] = select_soils(grid, soils, suit_soils, 
                                                 method='all', n=1)
        print 'We have selected all', len(selected_soil_types[grid]),\
              'soil types:',\
              [stu for smu, stu, w, data in selected_soil_types[grid]],\
              'for grid', grid

#-------------------------------------------------------------------------------
# we retrieve the crop cultivated fraction frac_crop
# it is read from EUROSTAT data

    # frac_crop = 

#-------------------------------------------------------------------------------
#   WE NEED TO LOOP OVER THE GRID CELLS
    for grid_no, arable_land in selected_grid_cells:
#-------------------------------------------------------------------------------
# We retrieve the longitude and latitude of the CGMS grid cell

    # TO BE CODED WITH ALLARD's GRID DICTIONNARY
    # for now we assume one fixed coordinates at the beginning of the script
        lat = 42. # degrees N of the CGMS grid cell
        lon = -4. # degrees E of the CGMS grid cell

#-------------------------------------------------------------------------------
# We open the incoming surface shortwave radiation [W.m-2] 

        rad = retrieve_ecmwf_ssrd(opti_year, lon, lat)

#-------------------------------------------------------------------------------
# We open the surface temperature record

        ts  = retrieve_ecmwf_tsurf(opti_year, lon, lat)

#-------------------------------------------------------------------------------
# we retrieve the arable_land area of the grid cell

        #frac_arable    = FINAL_YLD['arable_area(ha)'][l] / 62500.
        #frac_culti     = frac_arable * frac_crop

#-------------------------------------------------------------------------------
# we initialize the timeseries of harvest gpp and Resp for the grid cell

        time_cell_timeseries = rad[0]
        len_cell_timeseries  = len(rad[0])
        gpp_cell_timeseries  = np.array([0.]*len_cell_timeseries)
        raut_cell_timeseries = np.array([0.]*len_cell_timeseries)
        rhet_cell_timeseries = np.array([0.]*len_cell_timeseries)
        sum_stu_areas        = 0.

#-------------------------------------------------------------------------------
#       WE NEED TO LOOP OVER THE SOIL TYPE
        for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
            print grid_no, stu_no
#-------------------------------------------------------------------------------
# We open the WOFOST results file

            filelist    = 'pcse_output_c%i_g%i_s%i_y%i.csv'\
                           %(crop_no, grid_no, stu_no, opti_year) 
            wofost_data = open_pcse_csv_output(pcseoutputdir, [filelist])

#-------------------------------------------------------------------------------
# We apply the short wave radiation diurnal cycle on the GPP and R_auto

            # we create empty time series for this specific stu
            gpp_stu_timeseries  = np.array([])
            raut_stu_timeseries = np.array([])

            # we compile the sum of the stu areas to do a weighted average of
            # GPP and Rauto later on
            sum_stu_areas += stu_area 

            for DOY, timeinsec in enumerate(time_cell_timeseries[::8]):
                #print 'doy, timeinsec:', DOY, timeinsec

                # conversion of current time in seconds into date
                time = datetime.date(opti_year,1,1) + datetime.timedelta(DOY)
                #print 'date:', time

                # we test to see if we are within the growing season
                test_sow = (time - wofost_data[filelist]['day'][0]).total_seconds()
                test_rip = (time - wofost_data[filelist]['day'][-1]).total_seconds() 
                #print 'tests:', test_sow, test_rip

                # if the day of the time series is before sowing date: npp=0
                if test_sow < 0.: 
                    gpp_day  = 0.
                    raut_day = 0.
                # if the day of the time series is after the harvest date: npp=0
                elif test_rip > 0.: 
                    gpp_day  = 0.
                    raut_day = 0.
                # else we get the daily total GPP and Raut in kgCH2O/ha/day from 
                # wofost, and we weigh it with the stu area to later on calculate
                # the weighted average GPP and Raut in the grid cell
                else: 
                    # index of the sowing date in the time_cell_timeseries:
                    if (test_sow == 0.): DOY_sowing  = DOY
                    if (test_rip == 0.): DOY_harvest = DOY
                    #print 'DOY sowing:', DOY_sowing
                    # translation of cell to stu timeseries index
                    index_day_w  = DOY - DOY_sowing
                    #print 'index of day in wofost record:', index_day_w
                    gpp_day      = wofost_data[filelist]['RD'][index_day_w] * \
                                                 stu_area # need to specify GASS
                    raut_day     = wofost_data[filelist]['RD'][index_day_w] * \
                                                 stu_area # need to specify MRES

                # we select the radiation diurnal cycle for that date
                # NB: the last index is ignored in the selection, so we DO have
                # 8 time steps selected only (it's a 3-hourly dataset)
                rad_cycle   = rad[1][DOY*8:DOY*8+8] 

                # we apply the radiation cycle on the GPP and Rauto
                # NB: we check if the applied kernel is correct
                kernel      = rad_cycle / sum(rad_cycle)
                gpp_cycle   = kernel * gpp_day
                raut_cycle  = kernel * raut_day
                if (sum(gpp_cycle)-gpp_day) > 0.00001: 
                    print 'wrong kernel applied on GPP', \
                                        sum(gpp_cycle)-gpp_day, sum(kernel), DOY
                if (sum(raut_cycle)-raut_day) > 0.00001: 
                    print 'wrong kernel applied on Rauto', \
                                      sum(raut_cycle)-raut_day, sum(kernel), DOY
                gpp_stu_timeseries  = np.concatenate((gpp_stu_timeseries, 
                                                      gpp_cycle), axis=0)
                raut_stu_timeseries = np.concatenate((raut_stu_timeseries,
                                                      raut_cycle), axis=0)

#-------------------------------------------------------------------------------
# We add the gpp of all soil types in the grid cell. NB: different calendars
# are applied depending on the site!! so the sowing and maturity dates might
# differ from stu to stu

            # sum the timeseries of GPP and Rauto for all soil types
            gpp_cell_timeseries  = gpp_cell_timeseries  + gpp_stu_timeseries
            raut_cell_timeseries = raut_cell_timeseries + raut_stu_timeseries

        # for each grid cell, calculate the weighted average GPP and Rauto
        gpp_cell_timeseries  = gpp_cell_timeseries  / sum_stu_areas
        raut_cell_timeseries = raut_cell_timeseries / sum_stu_areas

#-------------------------------------------------------------------------------
# We calculate the heterotrophic respiration with the surface temperature

        # from the A-gs model:
        # pb: we need to simulate wg with that approach...
        #fw = Cw * wsmax / (wg + wsmin)
        tsurf_inter = Eact0 / (283.15 * 8.314) * (1 - 283.15 / ts[1])
        print [math.exp(t) for t in tsurf_inter], R10
        rhet_cell_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 

#-------------------------------------------------------------------------------
# We store the growing season's C fluxes for each grid x soil combi

# crop_no, year, lon_ecmwf, lat_ecmwf, grid_no, soil_no, GPP, R_auto_, R_hetero  

        from matplotlib import pyplot
        pyplot.close('all')
        fig = pyplot.figure(figsize=(14,3))
        fig.subplots_adjust(0.1,0.2,0.98,0.88,0.4,0.6)
        pyplot.plot(time_cell_timeseries/(3600.*24.),gpp_cell_timeseries, label='GPP')
        pyplot.plot(time_cell_timeseries/(3600.*24.),raut_cell_timeseries, label=r'$R_{aut}$')
        pyplot.plot(time_cell_timeseries/(3600.*24.),rhet_cell_timeseries, label=r'$R_{het}$')
        #pyplot.plot(time_cell_timeseries/(3600.*24.),gpp_stu_timeseries, label='GPP')
        pyplot.xlim([40.,170.])
        pyplot.xlabel('time (DOY)')
        pyplot.ylabel(r'carbon flux (kg$_{CH_2O}$ ha$^{-1}$)')
        pyplot.legend(loc='best')
        pyplot.show()

#-------------------------------------------------------------------------------
#   END OF THE TWO LOOPS
#-------------------------------------------------------------------------------

#===============================================================================
# function that will retrieve the surface temperature from the ECMWF data
# (ERA-interim). It will return two arrays: one of the time in seconds since
# 1st of Jan, and one with the tsurf variable in K.
def retrieve_ecmwf_tsurf(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    ecmwfdir = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/eur100x100/'
    tsurf = np.array([])
    time  = np.array([])

    for month in range (1,13):
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 't_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir,'%i/%02d'%(year,month),namefile)
            f = cdf.Dataset(pathfile)

            # retrieve closest latitude and longitude index of desired location
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the temperature at the highest pressure level, at that 
            # lon,lat location 
            #print f.variables['ssrd'] # to get the dimensions of the variable
            tsurf = np.append(tsurf, f.variables['T'][0:8, 0, latindx, lonindx])
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
 
    return time, tsurf

#===============================================================================
# function that will retrieve the incoming surface shortwave radiation from the
# ECMWF data (ERA-interim). It will return two arrays: one of the time in
# seconds since 1st of Jan, and one with the ssrd variable in W.m-2.
def retrieve_ecmwf_ssrd(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    # ecmwfdir/year/month contains: blh, ci, cp, d2m, ewss, g10m, lsp, nsss, 
    # sd, sf, skt, slhf, src, sshf, ssr, ssrd, sst, str, strd, swvl1, t2m, u10m,
    # v10m
    ecmwfdir = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/sfc/glb100x100/'

    ssrd = np.array([])
    time = np.array([])

    for month in range (1,13):
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssrd_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir,'%i/%02d'%(year,month),namefile)
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
# Function to select a subset of soil types within a grid cell
def select_soils(grid_cell_id, soil_iterator_, suitable_soils, method='topn', n=3):
#===============================================================================

    from random import sample as random_sample
    from operator import itemgetter as operator_itemgetter

    # Rank soils by decreasing area
    sorted_soils = []
    for smu_no, area_smu, stu_no, percentage_stu, soildata in soil_iterator_:
        if stu_no not in suitable_soils: continue
        weight_factor = area_smu * percentage_stu/100.
        sorted_soils = sorted_soils + [(smu_no, stu_no, weight_factor, soildata)]
    sorted_soils = sorted(sorted_soils, key=operator_itemgetter(2), reverse=True)
   
    # select a subset of soil types to loop over 
    # first option: we select the top n most present soils in the grid cell
    if   (method == 'topn'):
        subset_list   = sorted_soils[0:n]
    # second option: we select a random set of n soils within the grid cell
    elif (method == 'randomn'):
        try: # try to sample n random soil types:
            subset_list   = random_sample(sorted_soils,n)
        except: # if an error is raised ie. sample size bigger than population do
            subset_list   = sorted_soils
    # last option: we select all available soils in the grid cell
    else:
        subset_list   = sorted_soils
    
    return subset_list

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
