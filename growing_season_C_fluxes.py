#!/usr/bin/env python

import sys, os
import numpy as np

#===============================================================================
# This script uses WOFOST runs to simulate the carbon fluxes during the growing
# season: we use radiation data to have a diurnal cycle and we add 
# heterotrophic respiration
def main():
#===============================================================================
    from maries_toolbox import open_pcse_csv_output, get_crop_name, \
                               select_cells, select_soils
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    import datetime
    import math
#-------------------------------------------------------------------------------
    global ecmwfdir_ssrd, ecmwfdir_tsurf, pickled_inputdir, pcse_ouputdir
#-------------------------------------------------------------------------------
# User-defined:

    NUTS_no = 'ES43'
    crop_no = 3
    opti_year = 2006
    prod_figure = True

#-------------------------------------------------------------------------------
# variables calculated from user input
    crop_name = get_crop_name([crop_no])
    crop_name = crop_name[3][0]
#-------------------------------------------------------------------------------
# constants for R_hetero
    Eact0   = 53.3e3    # activation energy [kJ kmol-1]
    R10     = 0.23      # respiration at 10C [mgCO2 m-2 s-1], can be between
    Cw      = 1.6e-3    # constant water stress correction (Jacobs et al. 2007)
    wsmax   = 0.55      # upper reference value soil water [-]
    wsmin   = 0.005     # lower reference value soil water [-]
# molar masses for unit conversion of carbon fluxes
    mmC    = 12.01
    mmCO2  = 44.01
    mmCH2O = 30.03 
#-------------------------------------------------------------------------------
# We define working directories

    ecmwfdir_tsurf = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/'+\
                     'eur100x100/'

    ecmwfdir_ssrd = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/sfc/glb100x100/'

    # storage folder for individual PCSE runs:
    #pcse_outputdir = '/Storage/CO2/mariecombe/pcse_individual_output/'
    pcse_outputdir  = '/Users/mariecombe/Documents/Work/Research_project_3/'+\
                      'pcse/pcse_individual_output/'
    
    # storage folder for the CGMS input data files
    #pickled_inputdir  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    pickled_inputdir  = '/Users/mariecombe/Documents/Work/Research_project_3/'+\
                        'pcse/pickled_CGMS_input_data/'

#-------------------------------------------------------------------------------
# Select all grid cells of the NUTS region

    # we select all whole grid cells contained in the NUTS region
    selected_grid_cells = select_cells(NUTS_no, pickled_inputdir, method='all')

#-------------------------------------------------------------------------------
# Select all soil types of the NUTS region

    # We select all suitable soil types for each selected grid cell
    selected_soil_types = select_soils(crop_no, selected_grid_cells,
                          pickled_inputdir, method='all')

#-------------------------------------------------------------------------------
#   WE NEED TO LOOP OVER THE GRID CELLS
    for grid_no, arable_land in selected_grid_cells:
#-------------------------------------------------------------------------------
# We retrieve the longitude and latitude of the CGMS grid cell

        # TO BE CODED WITH ALLARD's GRID DICTIONNARY
        # for now we assume one fixed coordinates to retrieve ssrd and tsurf
        lat = 42. # degrees N of the CGMS grid cell
        lon = -4. # degrees E of the CGMS grid cell

#-------------------------------------------------------------------------------
# We open the incoming surface shortwave radiation [W.m-2] 

        filename_rad = 'rad_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(opti_year,lon,lat)
        if os.path.exists(filename_rad):
            print
            rad = pickle_load(open(filename_rad, 'rb'))
        else:
            rad = retrieve_ecmwf_ssrd(opti_year, lon, lat)
            pickle_dump(rad,open(filename_rad, 'wb'))

#-------------------------------------------------------------------------------
# We open the surface temperature record

        filename_ts = 'ts_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(opti_year,lon,lat)
        if os.path.exists(filename_ts):
            ts = pickle_load(open(filename_ts, 'rb'))
        else:
            ts = retrieve_ecmwf_tsurf(opti_year, lon, lat)
            pickle_dump(ts,open(filename_ts, 'wb'))

#-------------------------------------------------------------------------------
# we initialize the timeseries of gpp and Resp for the grid cell

        time_cell_timeseries = rad[0]
        len_cell_timeseries  = len(rad[0])
        gpp_cell_timeseries  = np.array([0.]*len_cell_timeseries)
        raut_cell_timeseries = np.array([0.]*len_cell_timeseries)
        rhet_cell_timeseries = np.array([0.]*len_cell_timeseries)
        sum_stu_areas        = 0.


        if (prod_figure == True):
            from matplotlib import pyplot as plt
            plt.close('all')
            fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
            fig1.subplots_adjust(0.1,0.1,0.98,0.9,0.2,0.2)

#-------------------------------------------------------------------------------
#       WE NEED TO LOOP OVER THE SOIL TYPE
        for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
            print grid_no, stu_no
#-------------------------------------------------------------------------------
# We open the WOFOST results file

            filelist    = 'pcse_output_c%i_g%i_s%i_y%i.csv'\
                           %(crop_no, grid_no, stu_no, opti_year) 
            wofost_data = open_pcse_csv_output(pcse_outputdir, [filelist])

#-------------------------------------------------------------------------------
# We apply the short wave radiation diurnal cycle on the GPP and R_auto

            # we create empty time series for this specific stu
            gpp_cycle_timeseries   = np.array([])
            raut_cycle_timeseries  = np.array([])
            gpp_perday_timeseries  = np.array([])
            raut_perday_timeseries = np.array([])

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
                    gpp_day    = - wofost_data[filelist]['GASS'][index_day_w] * \
                                                            (mmC / mmCH2O) * 0.1
                    maint_resp = wofost_data[filelist]['MRES'][index_day_w] * \
                                                            (mmC / mmCH2O) * 0.1
                    try: # if there are any available assimilates for growth
                        growth_fac = (wofost_data[filelist]['DMI'][index_day_w]) / \
                                 (wofost_data[filelist]['GASS'][index_day_w] - 
                                  wofost_data[filelist]['MRES'][index_day_w])
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
                # the sum of the 8 rates is equal to total/dt:
                dt             = 3600. * 3.
                sum_gpp_rates  = gpp_day   / dt
                sum_raut_rates = raut_day  / dt 
                # the day's 8 values of actual gpp and raut rates per second:
                gpp_cycle      = weights * sum_gpp_rates
                raut_cycle     = weights * sum_raut_rates
                # NB: we check if the applied diurnal cycle is correct
                assert (sum(weights)-1. < 0.000001), "wrong radiation kernel"
                assert (len(gpp_cycle)*int(dt) == 86400), "wrong dt in diurnal cycle"
                assert ((sum(gpp_cycle)*dt-gpp_day) < 0.00001), "wrong diurnal cycle "+\
                    "applied on GPP: residual=%.2f "%(sum(gpp_cycle)*dt-gpp_day) +\
                    "on DOY %i"%DOY
                assert ((sum(raut_cycle)*dt-raut_day) < 0.00001), "wrong diurnal cycle "+\
                    "applied on Rauto: residual=%.2f "%(sum(raut_cycle)*dt-raut_day) +\
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


            if (prod_figure == True):
                for ax, var, name, lims in zip(axes.flatten(), 
                [gpp_perday_timeseries, raut_perday_timeseries, 
                gpp_perday_timeseries + raut_perday_timeseries],
                ['GPP', 'Rauto', 'NPP'], [[-20.,0.],[0.,10.],[-15.,0.]]):
                    ax.plot(time_cell_timeseries[::8]/(3600.*24.), var, 
                                                          label='stu %i'%stu_no)
                    ax.set_xlim([40.,170.])
                    ax.set_ylim(lims)
                    ax.set_ylabel(name + r' (g$_{C}$ m$^{-2}$ d$^{-1}$)', fontsize=14)

#-------------------------------------------------------------------------------
# We add the gpp of all soil types in the grid cell. NB: different calendars
# are applied depending on the site!! so the sowing and maturity dates might
# differ from stu to stu

            # sum the timeseries of GPP and Rauto for all soil types
            gpp_cell_timeseries  = gpp_cell_timeseries  + gpp_cycle_timeseries*stu_area
            raut_cell_timeseries = raut_cell_timeseries + raut_cycle_timeseries*stu_area

        # finish the figure of multiple stu carbon fluxes
        if (prod_figure == True):
            plt.xlabel('time (DOY)', fontsize=14)
            plt.legend(loc='upper left', ncol=2, fontsize=10)
            fig1.suptitle('Daily carbon fluxes of %s for all '%crop_name+\
                         'soil types of grid cell %i (%s) in %i'%(grid_no,
                                                 NUTS_no, opti_year), fontsize=18)
            figname = 'GPP_perday_c%s_%s_y%i_g%i.png'%(crop_no,NUTS_no,opti_year,\
                                                                        grid_no)
            fig1.savefig(figname)

        # for each grid cell, calculate the weighted average GPP and Rauto
        gpp_cell_timeseries  = gpp_cell_timeseries  / sum_stu_areas
        raut_cell_timeseries = raut_cell_timeseries / sum_stu_areas

#-------------------------------------------------------------------------------
# We calculate the heterotrophic respiration with the surface temperature

        # from the A-gs model:
        # pb: we need to simulate wg with that approach...
        #fw = Cw * wsmax / (wg + wsmin)
        tsurf_inter = Eact0 / (283.15 * 8.314) * (1 - 283.15 / ts[1])
        rhet_cell_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 
        # conversion from mgCO2/m2/s to gC/m2/s
        rhet_cell_timeseries = rhet_cell_timeseries * (mmC / mmCO2) * 0.001

#-------------------------------------------------------------------------------
# We calculate NEE as the net flux

        nee_cell_timeseries = gpp_cell_timeseries + raut_cell_timeseries +\
                              rhet_cell_timeseries

#-------------------------------------------------------------------------------
# We store the growing season's C fluxes for each CGMS grid cell

# crop_no, year, lon_ecmwf, lat_ecmwf, grid_no, soil_no, GPP, R_auto_, R_hetero  

#-------------------------------------------------------------------------------
# if requested by the user, we produce one NEE figure per grid cell
 
        if (prod_figure == True):
            from matplotlib import pyplot as plt
            fig2 = plt.figure(figsize=(14,6))
            fig2.subplots_adjust(0.1,0.2,0.98,0.85,0.4,0.6)
            plt.plot(time_cell_timeseries/(3600.*24.),gpp_cell_timeseries*1000., 
                                                             label=r'$\mathrm{GPP}$', c='g')
            plt.plot(time_cell_timeseries/(3600.*24.),raut_cell_timeseries*1000.,
                                                      label=r'$R_{aut}$', c='b')
            plt.plot(time_cell_timeseries/(3600.*24.),rhet_cell_timeseries*1000.,
                                                      label=r'$R_{het}$', c='r')
            plt.plot(time_cell_timeseries/(3600.*24.),nee_cell_timeseries*1000., 
                                                             label=r'$\mathrm{NEE}$', c='k')
            plt.xlim([0.,365.])
            plt.ylim([-1.,1.])
            plt.xlabel('time (DOY)')
            plt.ylabel(r'carbon flux (mg$_{C}$ m$^{-2}$ s$^{-1}$)')
            plt.legend(loc='best', ncol=2, fontsize=10)
            plt.title('Average carbon fluxes of %s over the  '%crop_name+\
                         'cultivated area of grid cell %i (%s) in %i'%(grid_no,
                                                            NUTS_no, opti_year))
            figname = 'Cbalance_c%s_%s_y%i_g%i.png'%(crop_no,NUTS_no,opti_year,\
                                                                        grid_no)
            fig2.savefig(figname)
            #plt.show()

#-------------------------------------------------------------------------------
        # until the whole code has been validated, we stop the script after
        # testing it on one grid cell only:
        sys.exit(2)

#-------------------------------------------------------------------------------
#   END OF THE TWO LOOPS
#-------------------------------------------------------------------------------
# END OF THE MAIN CODE

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
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 't_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_tsurf,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
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
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssrd_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_ssrd,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
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
