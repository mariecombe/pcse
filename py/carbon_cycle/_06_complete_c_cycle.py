#!/usr/bin/env python

import sys
from os import path
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import os
import numpy as np

import logging as mylogger
from py.tools.initexit import start_logger, parse_options
from py.carbon_cycle._01_select_crops_n_regions import select_crops_regions
import py.tools.rc as rc
import tarfile

from cPickle import load as pickle_load
from cPickle import dump as pickle_dump
import datetime as dt

sec_per_day = 86400.

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
    global ecmwfdir, wofostdir, analysisdir, CGMSdir,EUROSTATdir,\
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
    _ = start_logger(level=mylogger.INFO)

    opts, args = parse_options()


    # First message from logger
    mylogger.info('Python Code has started')
    mylogger.info('Passed arguments:')
    for arg,val in args.iteritems():
        mylogger.info('         %s : %s'%(arg,val) )

    rcfilename = args['rc']
    rcF = rc.read(rcfilename)
    crops = [ rcF['crop'] ]
    #crops = [i.strip().replace(' ','_') for i in crops]
    years = [int(rcF['year'])]
    outputdir = rcF['dir.output']
    inputdir = rcF['dir.wofost.input']
    par_process = (rcF['fwd.wofost.parallel'] in ['True','TRUE','true','T'])
    nsoils = int(rcF['fwd.wofost.nsoils'])
    selec_method = rcF['fwd.wofost.method']
    potential_sim = (rcF['fwd.wofost.potential'] in ['True','TRUE','true','T'])
    ecmwfdir= rcF['dir.ecmwf.nc-meteo']

    # Post-processing settings
    prod_figure = False     # if True, will produce plots of fluxes per grid cell
    Eact0       = 53.3e3   # activation energy [kJ kmol-1]
    R10         = 0.08     # respiration at 10C [mgCO2 m-2 s-1], can be between

    # input data directory paths

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    cwdir       = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
    EUROSTATdir = os.path.join(inputdir, 'EUROSTATobs')
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    NUTS_regions,crop_dict = select_crops_regions(crops, EUROSTATdir)

    if rcF.has_key('nuts.limit'):
        nutslimit = rcF['nuts.limit'].split(',')
        NUTS_selected = []
        for nuts in nutslimit:
            NUTS_selected.extend([ n for n in sorted(NUTS_regions) if nuts.strip() in n] ) 
        mylogger.info('NUTS list limited to length %d based on nuts.limit (%s)'%(len(NUTS_selected),sorted(NUTS_selected)))
        NUTS_regions = NUTS_selected
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
        mylogger.info( '================================================')

        #-----------------------------------------------------------------------
        # loop over crops
        #-----------------------------------------------------------------------
        for crop in sorted(crop_dict.keys()):
            crop_name  = crop_dict[crop][1]
            mylogger.info( 'Crop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name) )
            if crop_dict[crop][0]==5:
                crop_no = 1
                mylogger.info( 'Modified the internal crop_no from 5 to 1')
            elif crop_dict[crop][0]==13:
                crop_no = 3
                mylogger.info( 'Modified the internal crop_no from 13 to 3')
            elif crop_dict[crop][0]==12:
                crop_no = 2
                mylogger.info( 'Modified the internal crop_no from 12 to 2')
            else:
                crop_no = crop_dict[crop][0]

            # build folder name from year and crop
            if potential_sim:
                subfolder = 'potential' 
            else:
                subfolder = 'optimized' 
            wofostdir = os.path.join(outputdir,'wofost',subfolder)

            # create post-processing folder if needed
            analysisdir = os.path.join(outputdir,'analysis', subfolder)
            if not os.path.exists(analysisdir):
                os.makedirs(analysisdir)
                mylogger.info('Created new folder: %s' %analysisdir )

            # list wofost tar output files in that directory
            tarfilelist = [f for f in os.listdir(wofostdir) if f.endswith('.tgz')]
            mylogger.info('Checking tar output files from wofost, now available %d files for c-cycle'%len(tarfilelist))

            for filename in tarfilelist:
                tarf=tarfile.open(os.path.join(wofostdir,filename),mode='r')
                gridfiles = tarf.getnames()
                tarf.extractall(path=analysisdir)
                tarf.close()

                # Get unique grid ids
                ugridfiles=[]
                ugrid_nos=[]
                for g in gridfiles:
                    grid_no = int(os.path.basename(g).split('_')[2][1:])
                    if grid_no not in ugrid_nos:
                        ugrid_nos.append(grid_no)
                        ugridfiles.append(g)

                mylogger.info('Done extracting tar file (%s), now available: %d files for c-cycle'% (os.path.basename(filename),len(ugridfiles)))

                #-------------------------------------------------------------------
                # loop over grid cells - this is the parallelized part -
                #-------------------------------------------------------------------
                # We add a timestamp at start of the forward runs
                start_timestamp = dt.datetime.utcnow()
                
                # if we do a serial iteration, we loop over the grid cells
                if (not par_process):
                    for gridf in sorted(ugridfiles):
                        compute_timeseries_fluxes(gridf)
                # if we do a parallelization, we use the multiprocessor
                # module to provide series of cells to the function
                if (par_process):
                    import multiprocessing
                    # get number of cpus available to job
                    try:
                        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
                        print "Success reading parallel env %d" % ncpus
                    except KeyError:
                        ncpus = multiprocessing.cpu_count()
                        print "Success obtaining processor count %d" % ncpus
                    p = multiprocessing.Pool(ncpus)
                    data = p.map(compute_timeseries_fluxes, sorted(ugridfiles))
                    p.close()

                for txtfile in gridfiles:  # clean up text files in analysisdir
                    os.remove(os.path.join(analysisdir,txtfile))
                
                # We add an end timestamp to time the process
                end_timestamp = dt.datetime.utcnow()
                mylogger.info( 'Duration of the post-processing: %s'%(end_timestamp - start_timestamp))

    mylogger.info('Successfully finished the script, returning...')
    sys.exit(0)


# END OF THE MAIN CODE

#===============================================================================
def compute_timeseries_fluxes(gridfilename):
#===============================================================================
        
    import math
    import pandas as pd
    from maries_toolbox import open_pcse_csv_output, select_soils

    _ , NUTS_no, grid_no, soil_no, opt_type = os.path.basename(gridfilename).split('_')

    grid_no=int(grid_no[1:])

    filepath = os.path.join(analysisdir,'carbonfluxes_%s_g%i.pickle'%(NUTS_no,grid_no))
    if os.path.exists(filepath):
        mylogger.info("Skipping existing analysis file: %s"%filepath)
        return None


    # We retrieve the longitude and latitude of the CGMS grid cell
    i   = np.argmin(np.absolute(all_grids - grid_no))
    lon = lons[i]
    lat = lats[i]
    mylogger.info('in NUTS region %s,  grid cell no %i: lon = %.2f , lat = %.2f'%(NUTS_no,grid_no,lon, lat) )
    # we retrieve the tsurf and rad variables from ECMWF
    rad = retrieve_ecmwf_ssrd(year, lon, lat)
    ts = retrieve_ecmwf_tsurf(year, lon, lat)

    startdate=dt.datetime(year,1,1,0,0,0)
    enddate=dt.datetime(year+1,1,1,0,0,0)
    len_perday = (enddate-startdate).days
    
    # we initialize the timeseries for the grid cell
    # time list for timeseries
    time_cell_perday_timeseries = np.zeros(len_perday)
    # length of all carbon fluxes timeseries
    # GPP timeseries
    gpp_cell_perday_timeseries  = np.zeros(len_perday)
    # autotrophic respiration timeseries
    raut_cell_perday_timeseries = np.zeros(len_perday)
    # heterotrophic respiration timeseries
    rhet_cell_perday_timeseries = np.zeros(len_perday)
    # soil moisture stress 
    tra_cell_perday_timeseries = np.zeros(len_perday)
    # soil moisture stress max
    tramx_cell_perday_timeseries = np.zeros(len_perday)
    # t2m sum per day
    t2m_cell_perday_timeseries = np.zeros(len_perday)
    # tsum sum per day
    tsum_cell_perday_timeseries = np.zeros(len_perday)
    # ssr sum per day
    ssr_cell_perday_timeseries = np.zeros(len_perday)
    # extra variables to calculate the harvest:
    tagp_cell_perday_timeseries = np.array([0.]*len_persec)
    twrt_cell_perday_timeseries = np.array([0.]*len_persec)
    twso_cell_perday_timeseries = np.array([0.]*len_persec)

    # we initialize some variables
    sum_stu_areas = 0. # sum of soil types areas
    delta = 3600. * 24. # WP changed to daily

    if (prod_figure == True):
        from matplotlib import pyplot as plt
        plt.close('all')
        fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
        fig1.subplots_adjust(0.1,0.1,0.98,0.9,0.2,0.2)

    # Select soil types to loop over
    soilist = select_soils(crop_no, [grid_no], CGMSsoil,
                           method=selec_method, n=nsoils)

    if not soilist:
        return None

    #---------------------------------------------------------------
    # loop over soil types
    #---------------------------------------------------------------
    soil_codes=[]
    soil_areas=[]
    for smu, stu_no, stu_area, soildata in soilist[grid_no]:

        soil_codes.append(stu_no)
        soil_areas.append(stu_area)
        # We open the WOFOST results file
        filename    = 'wofost_%s_g%i_s%i_%s'%(NUTS_no,grid_no, stu_no,opt_type) 
        if not os.path.exists(os.path.join(analysisdir,filename)):
            mylogger.info('Skipping missing file %s'%filename)
            continue
        results_set = open_pcse_csv_output(analysisdir, [filename])
        wofost_data = results_set[0]
        wofost_mass, wofost_yield ,wofost_hindex = results_set[1]

        # We apply the short wave radiation diurnal cycle on the GPP 
        # and R_auto

        # we create empty time series for this specific stu
        gpp_cycle_timeseries   = np.array([])
        raut_cycle_timeseries  = np.array([])
        gpp_perday_timeseries  = np.array([])
        raut_perday_timeseries = np.array([])
        tra_perday_timeseries = np.array([])
        tramx_perday_timeseries = np.array([])
        tagp_perday_timeseries = np.array([])
        twrt_perday_timeseries = np.array([])
        twso_perday_timeseries = np.array([])
     
        # we compile the sum of the stu areas to do a weighted average of
        # GPP and Rauto later on
        sum_stu_areas += stu_area 
     
        #-----------------------------------------------------------
        # loop over days of the year
        #-----------------------------------------------------------
        for DOY in np.arange(len_perday): # loop over days
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
                tra_day  = 0.
                tramx_day = 0.
                twso_day = 0.
                twrt_day = 0.
                tagp_day = 0.
            # or if the day of the time series is after the harvest date: 
            # plant fluxes are set to zero
            elif test_rip > 0.: 
                gpp_day  = 0.
                raut_day = 0.
                tra_day  = 0.
                tramx_day = 0.
                twso_day = 0.
                twrt_day = 0.
                tagp_day = 0.
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

                raut_day   = growth_resp  #WP#  + maint_resp

                try:
                    tra_day    = wofost_data[filename]['TRA'][index_day_w] 
                    tramx_day    = wofost_data[filename]['TRAMX'][index_day_w] 
                except:
                    tra_day    = -999.
                    tramx_day  = -999.

                # extra variables to calculate the harvest:
                tagp_day = wofost_data[filename]['TAGP'][index_day_w]
                twrt_day = wofost_data[filename]['TWRT'][index_day_w]
                twso_day = wofost_data[filename]['TWSO'][index_day_w]

     
            # we also store the carbon fluxes per day, for comparison with fluxnet
            gpp_perday_timeseries = np.concatenate((gpp_perday_timeseries,
                                                   [gpp_day]), axis=0) 
            raut_perday_timeseries = np.concatenate((raut_perday_timeseries,
                                                   [raut_day]), axis=0)
            tra_perday_timeseries = np.concatenate((tra_perday_timeseries,
                                                   [tra_day]), axis=0)
            tramx_perday_timeseries = np.concatenate((tramx_perday_timeseries,
                                                   [tramx_day]), axis=0)
            # extra variables to calculate the harvest:
            tagp_perday_timeseries = np.concatenate((tagp_perday_timeseries,
                                                   [tagp_day]), axis=0)
            twrt_perday_timeseries = np.concatenate((twrt_perday_timeseries,
                                                   [twrt_day]), axis=0)
            twso_perday_timeseries = np.concatenate((twso_perday_timeseries,
                                                   [twso_day]), axis=0)

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

        # b- sum the PER DAY timeseries
        gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries + \
                                      gpp_perday_timeseries*stu_area
        raut_cell_perday_timeseries = raut_cell_perday_timeseries + \
                                      raut_perday_timeseries*stu_area
        tra_cell_perday_timeseries  = tra_cell_perday_timeseries + \
                                      tra_perday_timeseries*stu_area
        tramx_cell_perday_timeseries = tramx_cell_perday_timeseries + \
                                      tramx_perday_timeseries*stu_area
        # extra variables to calculate the harvest:
        tagp_cell_perday_timeseries = tagp_cell_perday_timeseries + \
                                      tagp_perday_timeseries*stu_area
        twrt_cell_perday_timeseries = twrt_cell_perday_timeseries + \
                                      twrt_perday_timeseries*stu_area
        twso_cell_perday_timeseries = twso_cell_perday_timeseries + \
                                      twso_perday_timeseries*stu_area
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
   
    # b- PER DAY
    gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries  / sum_stu_areas
    raut_cell_perday_timeseries = raut_cell_perday_timeseries / sum_stu_areas
    tra_cell_perday_timeseries  = tra_cell_perday_timeseries  / sum_stu_areas
    tramx_cell_perday_timeseries  = tramx_cell_perday_timeseries  / sum_stu_areas
    # extra variables to calculate the harvest:
    tagp_cell_perday_timeseries = tagp_cell_perday_timeseries / sum_stu_areas
    twrt_cell_perday_timeseries = twrt_cell_perday_timeseries / sum_stu_areas
    twso_cell_perday_timeseries = twso_cell_perday_timeseries / sum_stu_areas

    # compute the heterotrophic respiration with the A-gs equation
    # NB: we assume here Rhet only dependant on tsurf, not soil moisture
    #fw = Cw * wsmax / (wg + wsmin)
    tsurf_inter = Eact0 / (283.15 * 8.314) * (1.0 - 283.15 / ts[1])
    # a- PER DAY:
    rhet_cell_persec_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 
    rhet_cell_perday_timeseries = rhet_cell_persec_timeseries.reshape((-1,8)).mean(axis=1)

    t2m_cell_perday_timeseries =  np.array(ts[1]).reshape((-1,8)).sum(axis=1) # sum of t2m
    tsum_cell_perday_timeseries =  np.array([ math.exp(t) for t in tsurf_inter ]).reshape((-1,8)).sum(axis=1) # sum of interp t2m
    ssr_cell_perday_timeseries =  np.array(rad[1]).reshape((-1,8)).sum(axis=1) # sum of ssr

    # conversion from mgCO2 to gC and from sec to day
    conversion_fac = (mmC / mmCO2) * 0.001 * sec_per_day
    rhet_cell_perday_timeseries = rhet_cell_perday_timeseries * conversion_fac

    # we format the time series using the pandas python library, for easy plotting
    startdate = '%i-01-01 00:00:00'%year
    enddate   = '%i-12-31 23:59:59'%year
    #htimes = pd.date_range(startdate, enddate, freq='3H')
    dtimes = pd.date_range(startdate, enddate, freq='1d')

    # pandas series, daily frequency:
    series = dict()
    # Add some metadata useful to put into the later NetCDF files

    series['NUTS_no'] = NUTS_no
    series['grid_no'] = grid_no
    series['coords'] = (lon,lat,)
    series['soil_codes'] = soil_codes
    series['soil_areas'] = soil_areas
    series['crop_mass'] = wofost_mass
    series['crop_yield'] = wofost_yield
    series['crop_hi'] = wofost_hindex

    series['daily'] = dict()
    series['daily']['T2M']     = pd.Series(t2m_cell_perday_timeseries, index=dtimes)
    series['daily']['TSUM']     = pd.Series(tsum_cell_perday_timeseries, index=dtimes)
    series['daily']['SSR']     = pd.Series(ssr_cell_perday_timeseries, index=dtimes)
    series['daily']['TRA']     = pd.Series(tra_cell_perday_timeseries, index=dtimes)
    series['daily']['TRAMX']   = pd.Series(tramx_cell_perday_timeseries, index=dtimes)
    series['daily']['GPP']     = pd.Series(gpp_cell_perday_timeseries, index=dtimes)
    series['daily']['Rauto']   = pd.Series(raut_cell_perday_timeseries, index=dtimes)
    series['daily']['Rhetero'] = pd.Series(rhet_cell_perday_timeseries, index=dtimes)
    series['daily']['TER']     = pd.Series(rhet_cell_perday_timeseries +
                                           raut_cell_perday_timeseries, index=dtimes)
    series['daily']['NEE']     = pd.Series(gpp_cell_perday_timeseries +\
                                           rhet_cell_perday_timeseries +\
                                           raut_cell_perday_timeseries, index=dtimes)
    series['daily']['TAGP']    = pd.Series(tagp_cell_perday_timeseries, index=dtimes)
    series['daily']['TWRT']    = pd.Series(twrt_cell_perday_timeseries, index=dtimes)
    series['daily']['TWSO']    = pd.Series(twso_cell_perday_timeseries, index=dtimes)

    # pandas series, 3-hourly frequency:
    #series['3-hourly'] = dict()
    #series['3-hourly']['GPP']     = pd.Series(gpp_cell_persec_timeseries, index=htimes)
    #series['3-hourly']['Rauto']   = pd.Series(raut_cell_persec_timeseries, index=htimes)
    #series['3-hourly']['Rhetero'] = pd.Series(rhet_cell_persec_timeseries, index=htimes)
    #series['3-hourly']['TER']     = pd.Series(rhet_cell_persec_timeseries +\
                                              #raut_cell_persec_timeseries, index=htimes)
    #series['3-hourly']['NEE']     = pd.Series(gpp_cell_persec_timeseries +\
                                              #rhet_cell_persec_timeseries +\
                                              #raut_cell_persec_timeseries, index=htimes)

    # we store the two pandas series in one pickle file
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
            namefile = 't2m_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir,'%i/%02d'%(year,month),
                                                             namefile))==False):
                #print 'cannot find %s'%namefile
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
            tsurf = np.append(tsurf, f.variables['t2m'][0:8, latindx, lonindx])
            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]
            # NB: the file has 8 time steps (3-hourly)
            time = np.append(time, f.variables['time'][:] - convtime)  
            f.close()

    if (len(time) < 2920):
        print '!!!WARNING!!!'
        print 'there are less than 365 days of data that we could retrieve'
        print 'check the folder %s for year %i'%(ecmwfdir, year)
 
    return time, tsurf

#===============================================================================
# function that will retrieve the incoming surface shortwave radiation from the
# ECMWF data (ERA-interim). It will return two arrays: one of the time in
# seconds since 1st of Jan, and one with the ssrd variable in W.m-2.
def retrieve_ecmwf_ssrd(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    ssr = np.array([])
    time = np.array([])

    for month in range (1,13):
        #print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssr_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir,'%i/%02d'%(year,month),
                                                             namefile))==False):
                #print 'cannot find %s'%namefile
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
            ssr = np.append(ssr, f.variables['ssr'][0:8, latindx, lonindx])
            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]
            # NB: the file has 8 time steps (3-hourly)
            time = np.append(time, f.variables['time'][:] - convtime)  
            f.close()

    if (len(time) < 2920):
        print '!!!WARNING!!!'
        print 'there are less than 365 days of data that we could retrieve'
        print 'check the folder %s for year %i'%(ecmwfdir, year)
 
    return time, ssr

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
