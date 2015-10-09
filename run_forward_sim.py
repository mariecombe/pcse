#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load
from datetime import datetime
from pcse.models import Wofost71_WLP_FD
from pcse.base_classes import WeatherDataProvider
from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

#===============================================================================
# This script executes forward simulations of WOFOST for all cultivated CGMS 
# grid cells (which ones are depends on crop species and year)
def main():
#===============================================================================
    from maries_toolbox import open_csv, select_soils
#-------------------------------------------------------------------------------
    global crop_no, year, selected_soil_types, pickledir, caboecmwfdir,\
           pcseoutputdir
#-------------------------------------------------------------------------------
# Define the settings of the run:

    crop_no       = 7        # CGMS crop number
    years         = [2006]   # list of years we want to do forward sims for

    selec_method  = 'topn'   # can be 'topn' or 'randomn' or 'all'
    nsoils        = 10       # number of selected soil types within a grid cell

    weather       = 'ECMWF'   # weather data used for the forward simulations
                             # can be 'CGMS' or 'ECMWF'

    process  = 'serial'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 10          # number of cores used in case of a parallelization

#-------------------------------------------------------------------------------
# Calculate key variables from the user input:

    nb_years       = len(years)
    campaign_years = np.linspace(int(years[0]),int(years[-1]),len(years))

#-------------------------------------------------------------------------------
# Define working directories

    # directories on capegrim:
    EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"
    pickledir     = '/Users/mariecombe/mnt/promise/CO2/marie/pickled_CGMS_input_data/'
    caboecmwfdir  = '/Users/mariecombe/mnt/promise/CO2/marie/CABO_weather_ECMWF/'
    pcseoutputdir = '/Users/mariecombe/mnt/promise/CO2/marie/pcse_output/'

#-------------------------------------------------------------------------------
# we read the CGMS grid cells coordinates from file

    CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# From this list, we select the subset of grid cells located in Europe that
# contain arable land (no need to create weather data where there are no crops!)

    filename            = pickledir + 'europe_arable_CGMS_cellids.pickle'
    europ_arable        = pickle_load(open(filename,'rb'))    
    selected_grid_cells = sorted(europ_arable)

#-------------------------------------------------------------------------------
# Select all the available soil types to loop over for the forward runs

    # We select a subset of soil types per selected grid cell
    selected_soil_types = select_soils(crop_no, selected_grid_cells,
                          pickledir, method=selec_method)

#-------------------------------------------------------------------------------
# Perform the forward simulations:

    # We add a timestamp at start of the forward runs
    start_timestamp = datetime.utcnow()

    # WE LOOP OVER ALL YEARS:
    for y, year in enumerate(campaign_years): 
        print '######################## Year %i ########################\n'%year

        # if we do a serial iteration, we loop over the grid cells that 
        # contain arable land
        if (process == 'serial'):
            for grid in europ_arable:
                perform_yield_sim(grid)

        # if we do a parallelization, we use the multiprocessor module to 
        # provide series of cells to the function
        if (process == 'parallel'):
            import multiprocessing
            p = multiprocessing.Pool(nb_cores)
            parallel = p.map(perform_yield_sim, europ_arable)
            p.close()

    # We add a timestamp at end of the forward runs, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the pcse runs:', end_timestamp-start_timestamp


#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sim(grid_no):
#===============================================================================

    # we open a results file to write only summary output (for harvest maps)
    res_filename  = 'pcse_output_crop%i_year%i_g%i.csv'%(crop_no, year, grid_no)
    if (os.path.isfile(os.path.join(pcseoutputdir, res_filename))):
        os.remove(os.path.join(pcseoutputdir, res_filename))
        print '\nDeleted old file %s in folder pcse_output/'%res_filename
    Results = open(os.path.join(pcseoutputdir, res_filename), 'w')

    # we write the header line:
    Results.write('YLDGAPF(-),  grid_no,  year,  stu_no, arable_area(ha), '\
                 +'stu_area(ha), TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)\n')
    
    # Retrieve the weather data of one grid cell
    if (weather == 'CGMS'):
        filename = pickledir+'weatherobject_g%d.pickle'%grid_no
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
    if (weather == 'ECMWF'):
        weatherdata = CABOWeatherDataProvider('%i.%s'%(grid_no,str(year)[1:4]), 
                                                         fpath=cabo_path)
    # Retrieve the soil data of one grid cell 
    filename = pickledir+'soilobject_g%d.pickle'%grid_no
    soil_iterator = pickle_load(open(filename,'rb'))

    # Retrieve calendar data of one year for one grid cell
    filename = pickledir+'timerobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    timerdata = pickle_load(open(filename,'rb'))
                    
    # Retrieve crop data of one year for one grid cell
    filename = pickledir+'cropobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    cropdata = pickle_load(open(filename,'rb'))

    # retrieve the fgap data of one year and one grid cell
    cropdata['YLDGAPF'] = 1.

    for smu, stu_no, stu_area, soildata in selected_soil_types_[grid_no]:
        
        # Retrieve the site data of one year, one grid cell, one soil type
        filename = pickledir+'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid_no,crop_no,
                                                                      year,stu_no)
        sitedata = pickle_load(open(filename,'rb'))

        # run WOFOST
        wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, cropdata, 
                                                                    weatherdata)
        wofost_object.run_till_terminate() #will stop the run when DVS=2

        # get time series of the output and take the selected variables
        wofost_object.store_to_file( pcseoutputdir +\
                                    "pcse_timeseries_c%i_y%i_g%i_s%i.csv"\
                                    %(crop_no,year,grid_no,stu_no))

        # get major summary output variables for each run
        # total dry weight of - dead and alive - storage organs (kg/ha)
        TSO       = wofost_object.get_variable('TWSO')
        # total dry weight of - dead and alive - leaves (kg/ha) 
        TLV       = wofost_object.get_variable('TWLV')
        # total dry weight of - dead and alive - stems (kg/ha) 
        TST       = wofost_object.get_variable('TWST')
        # total dry weight of - dead and alive - roots (kg/ha) 
        TRT       = wofost_object.get_variable('TWRT')
        # maximum LAI
        MLAI      = wofost_object.get_variable('LAIMAX')
        # rooting depth (cm)
        RD        = wofost_object.get_variable('RD')
        # Total above ground dry matter (kg/ha)
        TAGP      = wofost_object.get_variable('TAGP')

        Results.write('%10.3f, %8i, %5i, %7i, %15.2f, %12.5f, %14.2f, '\
                      '%14.2f, %14.2f, %14.2f, %14.2f, %13.2f, %15.2f\n'
                   %(yldgapf, grid, year, soil_type, arable_area/10000.,
                   stu_area/10000., TSO, TLV, TST, TRT, MLAI, RD, TAGP))
    Results.close()	


    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
