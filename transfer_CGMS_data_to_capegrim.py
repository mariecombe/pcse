#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

import subprocess
import cx_Oracle
import sqlalchemy as sa
from cPickle import dump as pickle_dump
from cPickle import load as pickle_load
from datetime import datetime
from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
from pcse.db.cgms11 import TimerDataProvider, SoilDataIterator, \
                           CropDataProvider, STU_Suitability, \
                           SiteDataProvider, WeatherObsGridDataProvider
from pcse.exceptions import PCSEError 

#===============================================================================
# This script retrieves input data (soil, crop, timer and site) from the CGMS 
# database and transfers it to capegrim
def main():
#===============================================================================
    global currentdir, EUROSTATdir, folder_local, folder_cape,\
           crop_no, suitable_stu, year, engine, retrieve_weather
#-------------------------------------------------------------------------------
    """
    This script retrieves CGMS input data for all CGMS European grid cells that
    contain arable land, then syncs it on capegrim in folder:
    /Users/mariecombe/mnt/promise/CO2/pickled_CGMS_input_data/

    BEWARE!! To be granted access the Oracle database, you need to be connected 
    to the internet within the WU network with an ethernet cable...

    When the script works, it retrieves the following input data:
    - soil data  (1 pickle file per grid cell): soil characteristics like
                  wilting point, field capacity, maximum rooting depth, etc...
    - crop data  (1 pickle file per grid cell, crop, year)
    - timer data (1 pickle file per grid cell, crop, year)
    - site data  (1 pickle file per grid cell, crop, year, stu)
    """
#-------------------------------------------------------------------------------
# USER INPUT - only this part should be modified by the user!!
 
    crop_no          = 7      # CGMS crop species ID number
    years            = [2006] # list of years we want to retrieve data for
    retrieve_weather = False  # if True, we retrive the CGMS weather input files
                              # Beware!! these files are huge!!!
    process  = 'parallel'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 8           # number of cores used in case of a parallelization

    # flags to execute parts of the script only:
    CGMS_input_retrieval = False
    crop_mask_creation   = True
    sync_to_capegrim     = False

#-------------------------------------------------------------------------------
# Calculate key variables from the user input:

    # we create an array of integers for the years
    campaign_years = np.linspace(int(years[0]),int(years[-1]),len(years))

    # we remind the user that if all 3 flags are False, nothing will happen...
    if ( (CGMS_input_retrieval == False) 
        and (crop_mask_creation == False) 
        and (sync_to_capegrim == False) ):
        print 'You have set the script to do no calculations...'
        print 'Please set a flag to True in the user settings!'
        sys.exit(2)

#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# folder to store the input data, on my local Macbook:
    folder_local  = '/Users/mariecombe/Documents/Work/Research_project_3/'\
                   +'pcse/pickled_CGMS_input_data/'

    # folder to sync the local folder with:
    folder_cape   = '/Users/mariecombe/mnt/promise/CO2/pickled_CGMS_input_data/'

    # folder on my local macbook where the CGMS_grid_list.csv file is located:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'

#-------------------------------------------------------------------------------
# we test the connection to the remote Oracle database
    
    # define the settings of the Oracle database connection
    user = "cgms12eu_select"
    password = "OnlySelect"
    tns = "EURDAS.WORLD"
    dsn = "oracle+cx_oracle://{user}:{pw}@{tns}".format(user=user, pw=password, 
                                                                        tns=tns)
    engine = sa.create_engine(dsn)
    print engine

    # test the connection:
    try:
        connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")
    except cx_Oracle.DatabaseError:
        print '\nBEWARE!! The Oracle database is not responding. Probably, you are'
        print 'not using a computer wired within the Wageningen University network.'
        print '--> Get connected with ethernet cable before trying again!'
        sys.exit(2)

#-------------------------------------------------------------------------------
# we read the list of CGMS grid cells from file

    CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# From this list, we select the subset of grid cells located in Europe that
# contain arable land (no need to create weather data where there are no crops!)

    europ_arable = get_list_CGMS_cells_in_Europe_arable(all_grids, lons, lats)
    europ_arable = sorted(europ_arable)

#-------------------------------------------------------------------------------
    if CGMS_input_retrieval == True:
#-------------------------------------------------------------------------------
# We are gonna retrieve input data from the CGMS database

        # We add a timestamp at start of the retrieval
        start_timestamp = datetime.utcnow()

        # We retrieve the list of suitable soil types for the selected crop species
        filename = folder_local + 'suitablesoilsobject_c%d.pickle'%crop_no
        if os.path.exists(filename):
            suitable_stu = pickle_load(open(filename,'rb'))
        else:
            suitable_stu = STU_Suitability(engine, crop_no)
            suitable_stu_list = []
            for item in suitable_stu:
                suitable_stu_list = suitable_stu_list + [item]
            suitable_stu = suitable_stu_list
            pickle_dump(suitable_stu,open(filename,'wb'))       

        # WE LOOP OVER ALL YEARS:
        for y, year in enumerate(campaign_years): 
            print '######################## Year %i ########################\n'%year

            # if we do a serial iteration, we loop over the grid cells that 
            # contain arable land
            if (process == 'serial'):
                for grid in europ_arable:
                    retrieve_CGMS_input(grid)

            # if we do a parallelization, we use the multiprocessor module to 
            # provide series of cells to the function
            if (process == 'parallel'):
                import multiprocessing
                p = multiprocessing.Pool(nb_cores)
                parallel = p.map(retrieve_CGMS_input, europ_arable)
                p.close()

        # We add a timestamp at end of the retrieval, to time the process
        end_timestamp = datetime.utcnow()
        print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#-------------------------------------------------------------------------------
    if crop_mask_creation == True:
#-------------------------------------------------------------------------------
# We are gonna create a crop MASK dictionary: collect the grid cell ids where
# the crop was grown on that year

        crop_mask = {}

        # We add a timestamp at start of the crop mask creation
        start_timestamp = datetime.utcnow()

        # WE LOOP OVER ALL YEARS:
        for y, year in enumerate(campaign_years): 
            print '######################## Year %i ########################\n'%year

            # We retrieve the grid cell ids of those where the crop has been
            # sown that year

            europ_culti = list()
            if (process == 'serial'):
                for grid in europ_arable:
                    europ_culti.append(get_id_if_cultivated(grid))
 
            if (process == 'parallel'):
                import multiprocessing
                p = multiprocessing.Pool(nb_cores)
                europ_culti = p.map(get_id_if_cultivated, europ_arable)
                p.close()
            europ_culti = np.array(europ_culti)
                
            # at the end of each year's retrieval, we store the array of
            # cultivated grid cells:
            crop_mask[year] = europ_culti
 
        # now we are out of the year loop, we pickle the crop mask dictionary
        filename = 'cropmask_c%d_y%d.pickle'%(crop_no,year)
        pickle_dump(crop_mask,open(filename,'wb'))

        # We add a timestamp at end of the retrieval, to time the process
        end_timestamp = datetime.utcnow()
        print '\nDuration of the crop mask creation:', end_timestamp - \
                                                       start_timestamp

#-------------------------------------------------------------------------------
    if sync_to_capegrim == True:
#-------------------------------------------------------------------------------
# We sync the local folder containing the pickle files with the capegrim folder

        subprocess.call(["rsync","-auEv","-e",
                     "'ssh -l mariecombe -i /Users/mariecombe/.shh/id_dsa'",
                     "--delete",
                     "/Users/mariecombe/Documents/Work/Research_project_3/pcse/pickled_CGMS_input_data/",
                     "mariecombe@capegrim.wur.nl:~/mnt/promise/CO2/marie/pickled_CGMS_input_data/"])


### END OF THE MAIN CODE ###

#===============================================================================
# function that will retrieve CGMS input data from Oracle database
def retrieve_CGMS_input(grid):
#===============================================================================
# if the retrieval does not raise an error, the crop was thus cultivated that year
    print '    - grid cell no %i'%grid
    try:

# We retrieve the crop calendar (timerdata)

        filename = folder_local + \
                   'timerobject_g%d_c%d_y%d.pickle'%(grid, crop_no, year)
        if os.path.exists(filename):
            pass
        else:
            timerdata = TimerDataProvider(engine, grid, crop_no, year)
            pickle_dump(timerdata,open(filename,'wb'))    

#-------------------------------------------------------------------------------
# If required by the user, we retrieve the weather data

        if retrieve_weather == True: 
            filename = folder_local + 'weatherobject_g%d.pickle'%grid
            if os.path.exists(filename):
                pass
            else:
                weatherdata = WeatherObsGridDataProvider(engine, grid)
                weatherdata._dump(filename)

#-------------------------------------------------------------------------------
# We retrieve the soil data (soil_iterator)
 
        filename = folder_local + 'soilobject_g%d.pickle'%grid
        if os.path.exists(filename):
            soil_iterator = pickle_load(open(filename,'rb'))
        else:
            soil_iterator = SoilDataIterator(engine, grid)
            pickle_dump(soil_iterator,open(filename,'wb'))       

#-------------------------------------------------------------------------------
# We retrieve the crop variety info (crop_data)

        filename = folder_local + \
                   'cropobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)
        if os.path.exists(filename):
            pass
        else:
            cropdata = CropDataProvider(engine, grid, crop_no, year)
            pickle_dump(cropdata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
#       WE LOOP OVER ALL SOIL TYPES LOCATED IN THE GRID CELL:
        for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:
#-------------------------------------------------------------------------------

            # NB: we remove all unsuitable soils from the iteration
            if (stu_no not in suitable_stu):
                pass
            else:
                print '        soil type no %i'%stu_no

#-------------------------------------------------------------------------------
# We retrieve the site data (site management)

                filename = folder_local + \
                           'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid, crop_no,
                                                                   year, stu_no)
                if os.path.exists(filename):
                    pass
                else:
                    sitedata = SiteDataProvider(engine, grid, crop_no, year,
                                                                         stu_no)
                    pickle_dump(sitedata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
    # if an error is raised, the crop was not grown that year
    except PCSEError:
        print '        the crop was not grown that year in that grid cell'

    return None

#===============================================================================
def get_id_if_cultivated(grid_no):
#===============================================================================
    print '    - grid cell no %i'%grid_no
    try:
        timerdata    = TimerDataProvider(engine, grid_no, crop_no, year)
        id_to_return = [grid_no]
    except PCSEError: # if the crop has not been grown on that year in this cell
        id_to_return = []

    return id_to_return

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
