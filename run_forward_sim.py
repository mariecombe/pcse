#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load
from cPickle import dump as pickle_dump
from datetime import datetime
from pcse.models import Wofost71_WLP_FD
from maries_toolbox import select_soils
from pcse.base_classes import WeatherDataProvider
from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

#===============================================================================
# This script executes forward simulations of WOFOST for all cultivated CGMS 
# grid cells (which ones are depends on crop species and year)
def main():
#===============================================================================
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
    global crop_no, year, selec_method, nsoils, weather, pickledir, caboecmwfdir,\
           pcseoutputdir, currentdir
#-------------------------------------------------------------------------------
# Define the settings of the run:

    crop_no       = 7        # CGMS crop number
    years         = [2006]   # list of years we want to do forward sims for

    selec_method  = 'topn'   # can be 'topn' or 'randomn' or 'all'
    nsoils        = 10       # number of selected soil types within a grid cell

    weather       = 'ECMWF'   # weather data used for the forward simulations
                             # can be 'CGMS' or 'ECMWF'

    process  = 'parallel'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 10          # number of cores used in case of a parallelization

#-------------------------------------------------------------------------------
# Calculate key variables from the user input:

    nb_years       = len(years)
    campaign_years = np.linspace(int(years[0]),int(years[-1]),len(years))

#-------------------------------------------------------------------------------
# Define working directories

    # directories on capegrim:
    currentdir    = os.getcwd()
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
            data = p.map(perform_yield_sim, europ_arable)
            p.close()

        # we open a results file to write only summary output (for harvest maps)
        regroup_summary_output()

    # We add a timestamp at end of the forward runs, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the pcse runs:', end_timestamp-start_timestamp


#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sim(grid_no):
#===============================================================================
    
    try:

        # Retrieve the weather data of one grid cell
        if (weather == 'CGMS'):
            filename = pickledir+'weatherobject_g%d.pickle'%grid_no
            weatherdata = WeatherDataProvider()
            weatherdata._load(filename)
        if (weather == 'ECMWF'):
            weatherdata = CABOWeatherDataProvider('%i'%(grid_no), 
                                                             fpath=caboecmwfdir)
        #print weatherdata(datetime.date(datetime(2006,4,1)))
 
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
        yldgapf = 1.
        cropdata['YLDGAPF'] = yldgapf
 
        # Select soil types to loop over for the forward runs
        selected_soil_types = select_soils(crop_no, [grid_no], pickledir, 
                                           method=selec_method, n=nsoils)
 
        for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
            
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
 
            #output_string = '%10.3f, %8i, %5i, %7i, %15.2f, %12.5f, %14.2f, '
                            #%(yldgapf, grid_no, year, stu_no, arable_area/10000.,stu_area/10000.,TSO) 
            output_string = '%10.3f, %8i, %5i, %7i, %12.5f, %14.2f, '%(yldgapf,
                             grid_no, year, stu_no, stu_area/10000., TSO) + \
                            '%14.2f, %14.2f, %14.2f, %14.2f, %13.2f, %15.2f\n'%(TLV,
                             TST, TRT, MLAI, RD, TAGP)

            # we pickle the one-liner summary output
            filename = 'pcse_oneline_c%i_y%i_g%i_s%i.pickle'%(crop_no,year,grid_no,
                                                                         stu_no)
            if os.path.exists(os.path.join(pcseoutputdir,filename)):
                os.remove(os.path.join(pcseoutputdir, filename))
            pickle_dump(output_string,open(os.path.join(pcseoutputdir,filename),'wb'))

    # if an error is raised, there are missing input files, ie. the crop was not
    # grown that year
    except IOError:
        print 'The crop was not grown that year in grid cell no %i'%grid_no

    return None

#===============================================================================
def regroup_summary_output():
#===============================================================================

    # we open a results file to write only summary output (for harvest maps)
    res_filename  = 'pcse_output_crop%i_year%i.csv'%(crop_no, year)
    if (os.path.isfile(os.path.join(pcseoutputdir, res_filename))):
        os.remove(os.path.join(pcseoutputdir, res_filename))
        print '\nDeleted old file %s in folder pcse_output/'%res_filename
    Results = open(os.path.join(pcseoutputdir, res_filename), 'w')

    # we write the header line:
    #Results.write('YLDGAPF(-),  grid_no,  year,  stu_no, arable_area(ha), '\
    Results.write('YLDGAPF(-),  grid_no,  year,  stu_no,  '\
                 +'stu_area(ha), TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)\n')

    # get list of files
    list_of_files = [f for f in os.listdir(currentdir) 
                     if ( f.startswith('pcse_oneline') 
                     and ('.pickle' in f) ) ]

    for namefile in list_of_files:
        line = pickle_load(open(namefile,'rb'))
        Results.write(line)
        os.remove(os.path.join(pcseoutputdir,namefile))

    # close the summary results file
    Results.close()	

    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
