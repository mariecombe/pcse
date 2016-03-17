#!/usr/bin/env python

import sys, os
import numpy as np

from cPickle import load as pickle_load
from cPickle import dump as pickle_dump
import datetime as dt

#===============================================================================
def main():
#===============================================================================
    """
    This method postprocesses WOFOST output (daily carbon pools increment) into 
    3-hourly crop surface CO2 exchange: we use radiation data to create a
    diurnal cycle for photosynthesis (GPP) and autotrophic respiration (Rauto),
    and we add an heterotrophic respiration (Rhet) component.

    The result of this post-processing is saved as pandas time series object in
    pickle files, for easy ploting purposes.

    """
#-------------------------------------------------------------------------------
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
# declare as global variables: folder names, main() method arguments, and a few
# variables/constants passed between functions
    global ecmwfdir_ssrd, ecmwfdir_tsurf, wofostdir, analysisdir, CGMSdir,\
           prod_figure, R10, Eact0, selec_method, nsoils,\
           mmC, mmCO2, mmCH2O, all_grids, lons, lats, crop_no, crop, year,\
           CGMSsoil
#-------------------------------------------------------------------------------
# constant molar masses for unit conversion of carbon fluxes
    mmC    = 12.01
    mmCO2  = 44.01
    mmCH2O = 30.03 
#-------------------------------------------------------------------------------
# Temporarily add the parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # Post-processing settings
    process     = 'parallel' # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores    = 12       # number of cores used in case of a parallelization
    prod_figure = False     # if True, will produce plots of fluxes per grid cell
    Eact0       = 53.3e3   # activation energy [kJ kmol-1]
    R10         = 0.14     # respiration at 10C [mgCO2 m-2 s-1], can be between

    # forward run settings that were used:
    potential_sim = False  # potential / optimum simulations
    selec_method  = 'topn' # can be 'topn' or 'randomn' or 'all'
    nsoils        = 3      # number of selected soil types within a grid cell

    # input data directory paths
    inputdir = '/Users/mariecombe/mnt/promise/CO2/wofost/'
    ecmwfdir_tsurf = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/'+\
                     'eur100x100/'

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    cwdir       = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    try:
        crop_dict    = pickle_load(open('../tmp/selected_crops.pickle','rb'))
        years        = pickle_load(open('../tmp/selected_years.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of crops and years '+\
              'to loop over'
        print 'Run the script _01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# open the pickle files containing the CGMS input data
    CGMSsoil  = pickle_load(open(os.path.join(CGMSdir,'CGMSsoil.pickle'),'rb'))
#-------------------------------------------------------------------------------
# we read the coordinates of all possible CGMS grid cells from file
    CGMS_cells = open_csv(CGMSdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# PERFORM THE POST-PROCESSING:
#-------------------------------------------------------------------------------
    # loop over years:
    #---------------------------------------------------------------------------
    for year in years:
        print '\nYear ', year
        print '================================================'

        #-----------------------------------------------------------------------
        # loop over crops
        #-----------------------------------------------------------------------
        for crop in sorted(crop_dict.keys()):
            crop_no    = crop_dict[crop][0]
            crop_name  = crop_dict[crop][1]
            print '\nCrop no %i: %s / %s'%(crop_no, crop, crop_name)
            print '================================================'

            # build folder name from year and crop
            if potential_sim:
                subfolder = 'potential' 
            else:
                subfolder = 'optimized' 
            wofostdir = os.path.join(cwdir,"../output/%i/%s/wofost/"%(year,crop),
                        subfolder)

            # create post-processing folder if needed
            analysisdir = os.path.join(cwdir,"../analysis/%i/%s/"%(year,crop),
                          subfolder)
            if not os.path.exists(analysisdir):
                os.makedirs(analysisdir)

            # list wofost output files in that directory
            filelist = os.listdir(wofostdir)

            # if the directory is not empty, we get the list of grid cell no 
            # for which we have performed forward simulations
            if len(filelist) > 0:
                gridlist = [int(f.split('_')[1][1:]) for f in filelist]
                gridlist = list(set(gridlist))
            # otherwise we skip to the next year x crop combination
            else:
                continue

            #-------------------------------------------------------------------
            # loop over grid cells - this is the parallelized part -
            #-------------------------------------------------------------------
            # We add a timestamp at start of the forward runs
            start_timestamp = dt.datetime.utcnow()
            
            # if we do a serial iteration, we loop over the grid cells
            if (process == 'serial'):
                for grid in sorted(gridlist):
                    compute_timeseries_fluxes(grid)
            
            # if we do a parallelization, we use the multiprocessor
            # module to provide series of cells to the function
            if (process == 'parallel'):
                import multiprocessing
                p = multiprocessing.Pool(nb_cores)
                data = p.map(compute_timeseries_fluxes, sorted(gridlist))
                p.close()
            
            # We add an end timestamp to time the process
            end_timestamp = dt.datetime.utcnow()
            print '\nDuration of the post-processing:', end_timestamp - \
            start_timestamp



# END OF THE MAIN CODE

#===============================================================================
def compute_timeseries_fluxes(grid_no):
#===============================================================================
        
    import math
    import pandas as pd
    from maries_toolbox import open_pcse_csv_output, select_soils

    # We retrieve the longitude and latitude of the CGMS grid cell
    i   = np.argmin(np.absolute(all_grids - grid_no))
    lon = lons[i]
    lat = lats[i]
    print '- grid cell no %i: lon = %.2f , lat = %.2f'%(grid_no,lon,
                                                                lat)
    # we retrieve the tsurf and rad variables from ECMWF
    rad = retrieve_ecmwf_ssrd(year, lon, lat)
    ts = retrieve_ecmwf_tsurf(year, lon, lat)
    
    # we initialize the timeseries for the grid cell
    # time list for timeseries
    time_cell_persec_timeseries = rad[0]
    time_cell_perday_timeseries = rad[0][::8]/(3600.*24.)
    # length of all carbon fluxes timeseries
    len_persec = len(rad[0])
    len_perday = len(rad[0][::8])
    # GPP timeseries
    gpp_cell_persec_timeseries  = np.array([0.]*len_persec)
    gpp_cell_perday_timeseries  = np.array([0.]*len_perday)
    # autotrophic respiration timeseries
    raut_cell_persec_timeseries = np.array([0.]*len_persec)
    raut_cell_perday_timeseries = np.array([0.]*len_perday)
    # heterotrophic respiration timeseries
    rhet_cell_persec_timeseries = np.array([0.]*len_persec)
    rhet_cell_perday_timeseries = np.array([0.]*len_perday)

    # we initialize some variables
    sum_stu_areas = 0. # sum of soil types areas
    delta = 3600. * 3. # number of seconds in delta (here 3 hours)

    if (prod_figure == True):
        from matplotlib import pyplot as plt
        plt.close('all')
        fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
        fig1.subplots_adjust(0.1,0.1,0.98,0.9,0.2,0.2)

    # Select soil types to loop over
    soilist = select_soils(crop_no, [grid_no], CGMSsoil
                           method=selec_method, n=nsoils)

    #---------------------------------------------------------------
    # loop over soil types
    #---------------------------------------------------------------
    for smu, stu_no, stu_area, soildata in soilist[grid_no]:

        # We open the WOFOST results file
        filename    = 'wofost_g%i_s%i.txt'%(grid_no, stu_no) 
        results_set = open_pcse_csv_output(wofostdir, [filename])
        wofost_data = results_set[0]

        # We apply the short wave radiation diurnal cycle on the GPP 
        # and R_auto

        # we create empty time series for this specific stu
        gpp_cycle_timeseries   = np.array([])
        raut_cycle_timeseries  = np.array([])
        gpp_perday_timeseries  = np.array([])
        raut_perday_timeseries = np.array([])
     
        # we compile the sum of the stu areas to do a weighted average of
        # GPP and Rauto later on
        sum_stu_areas += stu_area 
     
        #-----------------------------------------------------------
        # loop over days of the year
        #-----------------------------------------------------------
        for DOY, timeinsec in enumerate(time_cell_persec_timeseries[::8]):
            # conversion of current time in seconds into date
            time = dt.date(year,1,1) + dt.timedelta(DOY)
            #print 'date:', time
     
            # we test to see if we are within the growing season
            test_sow = (time - wofost_data[filename]['day'][0]).total_seconds()
            test_rip = (time - wofost_data[filename]['day'][-1]).total_seconds() 
            #print 'tests:', test_sow, test_rip
     
            # if the day of the time series is before sowing date: plant 
            # fluxes are set to zero
            if test_sow < 0.: 
                gpp_day  = 0.
                raut_day = 0.
            # or if the day of the time series is after the harvest date: 
            # plant fluxes are set to zero
            elif test_rip > 0.: 
                gpp_day  = 0.
                raut_day = 0.
            # else we get the daily total GPP and Raut in kgCH2O/ha/day
            # from wofost, and we weigh it with the stu area to later on 
            # calculate the weighted average GPP and Raut in the grid cell
            else: 
                # index of the sowing date in the time_cell_timeseries:
                if (test_sow == 0.): DOY_sowing  = DOY
                if (test_rip == 0.): DOY_harvest = DOY
                #print 'DOY sowing:', DOY_sowing
                # translation of cell to stu timeseries index
                index_day_w  = DOY - DOY_sowing
                #print 'index of day in wofost record:', index_day_w
     
                # unit conversion: from kgCH2O/ha/day to gC/m2/day
                gpp_day    = - wofost_data[filename]['GASS'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                maint_resp = wofost_data[filename]['MRES'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                try: # if there are any available assimilates for growth
                    growth_fac = (wofost_data[filename]['DMI'][index_day_w]) / \
                             (wofost_data[filename]['GASS'][index_day_w] - 
                              wofost_data[filename]['MRES'][index_day_w])
                    growth_resp = (1.-growth_fac)*(-gpp_day-maint_resp) 
                except ZeroDivisionError: # otherwise there is no crop growth
                    growth_resp = 0.
                raut_day   = growth_resp + maint_resp
     
            # we select the radiation diurnal cycle for that date
            # NB: the last index is ignored in the selection, so we DO have
            # 8 time steps selected only (it's a 3-hourly dataset)
            rad_cycle      = rad[1][DOY*8:DOY*8+8] 
     
            # we apply the radiation cycle on the GPP and Rauto
            # and we transform the daily integral into a rate
            weights        = rad_cycle / sum(rad_cycle)
            # the sum of the 8 rates is equal to total/delta:
            sum_gpp_rates  = gpp_day   / delta
            sum_raut_rates = raut_day  / delta
            # the day's 8 values of actual gpp and raut rates per second:
            gpp_cycle      = weights * sum_gpp_rates
            raut_cycle     = weights * sum_raut_rates
            # NB: we check if the applied diurnal cycle is correct
            assert (sum(weights)-1. < 0.000001), "wrong radiation kernel"
            assert (len(gpp_cycle)*int(delta) == 86400), "wrong delta in diurnal cycle"
            assert ((sum(gpp_cycle)*delta-gpp_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on GPP: residual=%.2f "%(sum(gpp_cycle)*delta-gpp_day) +\
                "on DOY %i"%DOY
            assert ((sum(raut_cycle)*delta-raut_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on Rauto: residual=%.2f "%(sum(raut_cycle)*delta-raut_day) +\
                "on DOY %i"%DOY
     
            # if the applied diurnal cycle is ok, we append that day's cycle
            # to the yearly record of the stu
            gpp_cycle_timeseries  = np.concatenate((gpp_cycle_timeseries, 
                                                   gpp_cycle), axis=0)
            raut_cycle_timeseries = np.concatenate((raut_cycle_timeseries,
                                                   raut_cycle), axis=0)
            # we also store the carbon fluxes per day, for comparison with fluxnet
            gpp_perday_timeseries = np.concatenate((gpp_perday_timeseries,
                                                   [gpp_day]), axis=0) 
            raut_perday_timeseries = np.concatenate((raut_perday_timeseries,
                                                   [raut_day]), axis=0)

        #-----------------------------------------------------------
        # end of day nb loop
        #-----------------------------------------------------------

        # plot the soil type timeseries if requested by the user
        if (prod_figure == True):
            for ax, var, name, lims in zip(axes.flatten(), 
            [gpp_perday_timeseries, raut_perday_timeseries, 
            gpp_perday_timeseries + raut_perday_timeseries],
            ['GPP', 'Rauto', 'NPP'], [[-20.,0.],[0.,10.],[-15.,0.]]):
                ax.plot(time_cell_perday_timeseries, var, 
                                              label='stu %i'%stu_no)
                #ax.set_xlim([40.,170.])
                #ax.set_ylim(lims)
                ax.set_ylabel(name + r' (g$_{C}$ m$^{-2}$ d$^{-1}$)', 
                                                        fontsize=14)

        # We compile time series of carbon fluxes in units per day and per second
        # a- sum the PER SECOND timeseries
        gpp_cell_persec_timeseries  = gpp_cell_persec_timeseries + \
                                      gpp_cycle_timeseries*stu_area
        raut_cell_persec_timeseries = raut_cell_persec_timeseries + \
                                      raut_cycle_timeseries*stu_area

        # b- sum the PER DAY timeseries
        gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries + \
                                      gpp_perday_timeseries*stu_area
        raut_cell_perday_timeseries = raut_cell_perday_timeseries + \
                                      raut_perday_timeseries*stu_area
    #---------------------------------------------------------------
    # end of soil type loop
    #---------------------------------------------------------------

    # finish ploting the soil type timeseries if requested by the user
    if (prod_figure == True):
        plt.xlabel('time (DOY)', fontsize=14)
        plt.legend(loc='upper left', ncol=2, fontsize=10)
        fig1.suptitle('Daily carbon fluxes of %s for all '%crop+\
                     'soil types of grid cell %i in %i'%(grid_no,
                                              year), fontsize=18)
        figname = 'GPP_allsoils_%i_c%i_g%i.png'%(year,crop_no,\
                                                            grid_no)
        #plt.show()
        fig1.savefig(os.path.join(analysisdir,figname))

    # compute the weighted average of GPP, Rauto over the grid cell
    # a- PER SECOND
    gpp_cell_persec_timeseries  = gpp_cell_persec_timeseries  / sum_stu_areas
    raut_cell_persec_timeseries = raut_cell_persec_timeseries / sum_stu_areas
   
    # b- PER DAY
    gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries  / sum_stu_areas
    raut_cell_perday_timeseries = raut_cell_perday_timeseries / sum_stu_areas

    # compute the heterotrophic respiration with the A-gs equation
    # NB: we assume here Rhet only dependant on tsurf, not soil moisture
    #fw = Cw * wsmax / (wg + wsmin)
    tsurf_inter = Eact0 / (283.15 * 8.314) * (1 - 283.15 / ts[1])
    # a- PER SEC:
    rhet_cell_persec_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 
    # b- PER DAY:
    for i in range(len(rhet_cell_perday_timeseries)):
        rhet_cell_perday_timeseries[i] = rhet_cell_persec_timeseries[i*8] * delta +\
                                       rhet_cell_persec_timeseries[i*8+1] * delta +\
                                       rhet_cell_persec_timeseries[i*8+2] * delta +\
                                       rhet_cell_persec_timeseries[i*8+3] * delta +\
                                       rhet_cell_persec_timeseries[i*8+4] * delta +\
                                       rhet_cell_persec_timeseries[i*8+5] * delta +\
                                       rhet_cell_persec_timeseries[i*8+6] * delta +\
                                       rhet_cell_persec_timeseries[i*8+7] * delta 
   
    # conversion from mgCO2 to gC
    conversion_fac = (mmC / mmCO2) * 0.001
    rhet_cell_persec_timeseries = rhet_cell_persec_timeseries * conversion_fac
    rhet_cell_perday_timeseries = rhet_cell_perday_timeseries * conversion_fac

    # we format the time series using the pandas python library, for easy plotting
    startdate = '%i-01-01 00:00:00'%year
    enddate   = '%i-12-31 23:59:59'%year
    htimes = pd.date_range(startdate, enddate, freq='3H')
    dtimes = pd.date_range(startdate, enddate, freq='1d')

    # pandas series, daily frequency:
    series = dict()
    series['daily'] = dict()
    series['daily']['GPP']     = pd.Series(gpp_cell_perday_timeseries, index=dtimes)
    series['daily']['Rauto']   = pd.Series(raut_cell_perday_timeseries, index=dtimes)
    series['daily']['Rhetero'] = pd.Series(rhet_cell_perday_timeseries, index=dtimes)
    series['daily']['TER']     = pd.Series(rhet_cell_perday_timeseries +
                                           raut_cell_perday_timeseries, index=dtimes)
    series['daily']['NEE']     = pd.Series(gpp_cell_perday_timeseries +\
                                           rhet_cell_perday_timeseries +\
                                           raut_cell_perday_timeseries, index=dtimes)

    # pandas series, 3-hourly frequency:
    series['3-hourly'] = dict()
    series['3-hourly']['GPP']     = pd.Series(gpp_cell_persec_timeseries, index=htimes)
    series['3-hourly']['Rauto']   = pd.Series(raut_cell_persec_timeseries, index=htimes)
    series['3-hourly']['Rhetero'] = pd.Series(rhet_cell_persec_timeseries, index=htimes)
    series['3-hourly']['TER']     = pd.Series(rhet_cell_persec_timeseries +\
                                              raut_cell_persec_timeseries, index=htimes)
    series['3-hourly']['NEE']     = pd.Series(gpp_cell_persec_timeseries +\
                                              rhet_cell_persec_timeseries +\
                                              raut_cell_persec_timeseries, index=htimes)

    # we store the two pandas series in one pickle file
    filepath = os.path.join(analysisdir,'carbon_fluxes_g%i.pickle'%(grid_no))
    pickle_dump(series, open(filepath,'wb'))

    return None


#===============================================================================
# function that will retrieve the surface temperature from the ECMWF data
# (ERA-interim). It will return two arrays: one of the time in seconds since
# 1st of Jan, and one with the tsurf variable in K.
def retrieve_ecmwf_tsurf(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    tsurf = np.array([])
    time  = np.array([])

    for month in range (1,13):
        #print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 't_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_tsurf,'%i/%02d'%(year,month),
                                                             namefile))==False):
                #print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir_tsurf,'%i/%02d'%(year,month),namefile)
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
        print 'check the folder %s for year %i'%(ecmwfdir_tsurf, year)
 
    return time, tsurf

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
        #print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssrd_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_ssrd,'%i/%02d'%(year,month),
                                                             namefile))==False):
                #print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir_ssrd,'%i/%02d'%(year,month),namefile)
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
        print 'check the folder %s for year %i'%(ecmwfdir_ssrd, year)
 
    return time, ssrd

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
