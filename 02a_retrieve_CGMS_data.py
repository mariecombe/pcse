#!/usr/bin/env python

import sys, os
import numpy as np
from pcse.exceptions import PCSEError 
from pcse.db.cgms11 import TimerDataProvider, SoilDataIterator, \
                           CropDataProvider, STU_Suitability, \
                           SiteDataProvider, WeatherObsGridDataProvider

# This script retrieves input data (soil, crop, timer and site) from the CGMS 
# database

#===============================================================================
def main():
#===============================================================================
    """
    This script tries to retrieve CGMS input data for all CGMS European grid 
    cells that contain arable land, then syncs it on capegrim in folder:
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
    import subprocess
    import cx_Oracle
    import sqlalchemy as sa
    from cPickle import dump as pickle_dump
    from cPickle import load as pickle_load
    from datetime import datetime
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, folder_local,\
           crop_no, suitable_stu, year, engine, retrieve_weather
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================
 
    # multiprocessing options:
    process  = 'serial'  # can be 'serial' or 'parallel'
    nb_cores = 2         # number of cores used in case of a parallelization

    # flags to execute parts of the script only:
    CGMS_input_retrieval = False
    retrieve_weather     = False  # if True, we retrive the CGMS weather input files
                              # Beware!! these files are huge!!!
    crop_mask_creation   = True
    sync_to_capegrim     = False

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    currentdir    = os.getcwd()
    folder_local  = '../model_input_data/CGMS/'
    folder_cape   = '/Users/mariecombe/mnt/promise/CO2/marie/pickled_CGMS_input_data/'
    EUROSTATdir   = '../observations/EUROSTAT_data/'
#-------------------------------------------------------------------------------
# we remind the user that if all 3 flags are False, nothing will happen...
    if ( (CGMS_input_retrieval == False) 
        and (crop_mask_creation == False) 
        and (sync_to_capegrim == False) ):
        print 'You have set the script to do no calculations...'
        print 'Please set a flag to True in the user settings!'
        sys.exit(2)
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    try:
        crop_dict = pickle_load(open('selected_crops.pickle','rb'))
        years     = pickle_load(open('selected_years.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of crops and years to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# we test the connection to the remote Oracle database
    
    # settings of the connection
    user = "cgms12eu_select"
    password = "OnlySelect"
    tns = "EURDAS.WORLD"
    dsn = "oracle+cx_oracle://{user}:{pw}@{tns}".format(user=user,pw=password,tns=tns)
    engine = sa.create_engine(dsn)
    print engine

    # test the connection:
    try:
        connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")
    except cx_Oracle.DatabaseError:
        print '\nBEWARE!! The Oracle database is not responding. Probably, you are'
        print 'not using a computer wired within the Wageningen University network.'
        print '--> Get connected with ethernet cable before trying again!'
        sys.exit()

#-------------------------------------------------------------------------------
# We only select grid cells located in Europe that contain arable land (no need 
# to retrieve data where there are no crops!)

    pathname = os.path.join('../model_input_data/europe_arable_CGMS_cellids.pickle')
    try:
        europ_arable = pickle_load(open(pathname,'rb'))
    except IOError:
        from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
        # we read the list of CGMS grid cells from file
        CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
        all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
        lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
        lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']
        # we select the grid cells with arable land from file
        europ_arable = get_list_CGMS_cells_in_Europe_arable(all_grids, lons, lats)
        europ_arable = sorted(europ_arable)
        # we pickle it for future use
        pickle_dump(europ_arable, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# LOOP OVER SELECTED CROPS:
#-------------------------------------------------------------------------------
    for crop_key in sorted(crop_dict.keys()):
        crop_no = int(crop_dict[crop_key][0])
#-------------------------------------------------------------------------------
# 1- WE RETRIEVE CGMS INPUT DATA:
#-------------------------------------------------------------------------------
        if CGMS_input_retrieval == True:
#-------------------------------------------------------------------------------
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
            for y, year in enumerate(years): 
                print '######################## Year %i ########################\n'%year
         
                # if we do a serial iteration, we loop over the grid cells that 
                # contain arable land
                if (process == 'serial'):
                    for grid in [g for g,a in europ_arable]:
                        retrieve_CGMS_input(grid)
         
                # if we do a parallelization, we use the multiprocessor module to 
                # provide series of cells to the function
                if (process == 'parallel'):
                    import multiprocessing
                    p = multiprocessing.Pool(nb_cores)
                    parallel = p.map(retrieve_CGMS_input, [g for g,a in europ_arable])
                    p.close()
         
            # We add a timestamp at end of the retrieval, to time the process
            end_timestamp = datetime.utcnow()
            print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#-------------------------------------------------------------------------------
# 2- WE RETRIEVE THE CULTIVATED GRID CELL IDS
#-------------------------------------------------------------------------------
        if crop_mask_creation == True:
#-------------------------------------------------------------------------------

            crop_mask = dict()
         
            # We add a timestamp at start of the crop mask creation
            start_timestamp = datetime.utcnow()
         
            # WE LOOP OVER ALL YEARS:
            for y, year in enumerate(years): 
                print '######################## Year %i ########################\n'%year
         
                culti_list = list()
         
                # We retrieve the grid cell ids of those where the crop has been
                # sown that year
         
                europ_culti = list()
                if (process == 'serial'):
                    for grid in [g for g,a in europ_arable]:
                        europ_culti += get_id_if_cultivated(grid)
         
                if (process == 'parallel'):
                    import multiprocessing
                    p = multiprocessing.Pool(nb_cores)
                    europ_culti = p.map(get_id_if_cultivated, europ_arable)
                    p.close()
                europ_culti = np.array(europ_culti)
                print 'europ_culti', europ_culti
               
	 	   	# we have created a list of grid cell ids, but we need their arable
	 	   	# land area as well. We do one more loop to retrieve that
	 	   	# information from europ_arable
                for cell in europ_arable:
                    if cell[0] in europ_culti:
                        culti_list += [cell]
                print 'culti list:', culti_list
                # at the end of each year's retrieval, we store the array of
                # cultivated grid cells:
                crop_mask[int(year)] = culti_list
         
            # now we are out of the year loop, we pickle the crop mask dictionary
            filename = 'cropmask_c%d_y%d.pickle'%(crop_no,year)
            pickle_dump(crop_mask,open('../model_input_data/'+filename,'wb'))
         
            # We add a timestamp at end of the retrieval, to time the process
            end_timestamp = datetime.utcnow()
            print '\nDuration of the crop mask creation:', end_timestamp - \
                                                           start_timestamp

#-------------------------------------------------------------------------------
# 3- WE SYNC THE LOCAL FOLDER WITH THE REMOTE CAPEGRIM FOLDER
#-------------------------------------------------------------------------------
        if sync_to_capegrim == True:
#-------------------------------------------------------------------------------

            subprocess.call(["rsync","-auEv","-e",
                     "'ssh -l mariecombe -i /Users/mariecombe/.shh/id_dsa'",
             "--delete",folder_local,"mariecombe@capegrim.wur.nl:"+folder_cape])


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
            print 'it exists!'
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
