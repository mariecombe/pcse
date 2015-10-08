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
    global currentdir, EUROSTATdir, folder_local, folder_cape
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

#-------------------------------------------------------------------------------
# Calculate key variables from the user input:

    # we create an array of integers for the years
    campaign_years = np.linspace(int(years[0]),int(years[-1]),len(years))

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
# We add a timestamp at start of the retrieval

    start_timestamp = datetime.utcnow()

#-------------------------------------------------------------------------------
# we read the list of CGMS grid cells from file

    all_CGMS_grid_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids           = all_CGMS_grid_cells['CGMS_grid_list.csv']['GRID_NO']
    lon                 = all_CGMS_grid_cells['CGMS_grid_list.csv']['LONGITUDE']
    lat                 = all_CGMS_grid_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# From this list, we select the subset of grid cells located in Europe that
# contain arable land (no need to create weather data where there are no crops!)

    europ_arable = get_list_CGMS_cells_in_Europe_arable(all_grids, lons, lats)

#-------------------------------------------------------------------------------
# We retrieve the list of suitable soil types for the selected crop species
    
    filename = folder_local + 'suitablesoilsobject_c%d.pickle'%crop_no

    if os.path.exists(filename):
        #print '%s exists!'%filename
        suitable_stu = pickle_load(open(filename,'rb'))
        #print 'Reading data from pickle file %s'%filename
    else:
        suitable_stu = STU_Suitability(engine, crop_no)
        suitable_stu_list = []
        for item in suitable_stu:
            suitable_stu_list = suitable_stu_list + [item]
        suitable_stu = suitable_stu_list
        #print 'Writing data to pickle file %s'%filename
        pickle_dump(suitable_stu,open(filename,'wb'))       

#-------------------------------------------------------------------------------
# We are gonna create a crop MASK dictionary: collect the grid cell ids where
# the crop was grown on that year

    crop_mask = {}

#-------------------------------------------------------------------------------
#   WE LOOP OVER ALL YEARS:
    for y, year in enumerate(campaign_years): 
        print '######################## Year %i ########################\n'%year
        europ_cultivated = np.array([])
#-------------------------------------------------------------------------------
#       WE LOOP OVER ALL EUROPEAN GRID CELLS THAT CONTAIN ARABLE LAND
        for grid in europ_arable:
            print '    - grid cell no %i'%grid
#-------------------------------------------------------------------------------
# If required by the user, we retrieve the weather data (1 file per grid cell):

            if retrieve_weather == True: 
                filename = folder_local + 'weatherobject_g%d.pickle'%grid
         
                if os.path.exists(filename):
                    #print '%s exists!'%filename
                    pass
                else:
                    weatherdata = WeatherObsGridDataProvider(engine, grid)
                    #print 'Writing data to pickle file %s'%filename
                    weatherdata._dump(filename)

#-------------------------------------------------------------------------------
# We retrieve the soil data (1 file per grid cell):
 
            filename = folder_local + 'soilobject_g%d.pickle'%grid
         
            if os.path.exists(filename):
                #print '%s exists!'%filename
                soil_iterator = pickle_load(open(filename,'rb'))
            else:
                soil_iterator = SoilDataIterator(engine, grid)
                #print 'Writing data to pickle file %s'%filename
                pickle_dump(soil_iterator,open(filename,'wb'))       

#-------------------------------------------------------------------------------
# We retrieve the timer data (crop calendar)

            filename = folder_local + \
                       'timerobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

            # if the retrieval does not raise an error, the crop was thus
            # cultivated that year
            try:
                if os.path.exists(filename):
                    #print '%s exists!!'%filename
                    pass
                else:
                    timerdata = TimerDataProvider(engine, grid, crop_no, year)
                    #print 'Writing data to pickle file %s'%filename
                    pickle_dump(timerdata,open(filename,'wb'))    
   
                europ_cultivated = np.append(europ_cultivated,grid)

            # if an error is raised, the crop was not grown that year
            except PCSEError:
                print '        the crop was not grown that year in that grid cell'
                continue # we go to the next grid cell
    
#-------------------------------------------------------------------------------
# We retrieve the crop data (crop varieties)

            filename = folder_local + \
                       'cropobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

            if os.path.exists(filename):
                #print '%s exists!!'%filename
                pass
            else:
                cropdata = CropDataProvider(engine, grid, crop_no, year)
                #print 'Writing data to pickle file %s'%filename
                pickle_dump(cropdata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
#           WE LOOP OVER ALL SOIL TYPES LOCATED IN THE GRID CELL:
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
                               'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid,crop_no,
                                                                    year,stu_no)

                    if os.path.exists(filename):
                        #print '%s exists!!'%filename
                        pass
                    else:
                        sitedata = SiteDataProvider(engine, grid, crop_no, 
                                                               year, stu_no)
                        #print 'Writing data to pickle file %s'%filename
                        pickle_dump(sitedata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
# We finalize the crop MASK dictionary

        # at the end of each year's retrieval, we store the array of cultivated
        # grid cells:
        crop_mask[year] = europ_cultivated

    # now we are out of the year loop, we pickle the crop mask dictionary
    filename = 'cropmask_c%d_y%d.pickle'%(crop_no,year)
    pickle_dump(crop_mask,open(filename,'wb'))

#-------------------------------------------------------------------------------
# We add a timestamp at end of the retrieval, to time the process

    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#-------------------------------------------------------------------------------
# We sync the local folder containing the pickle files with the capegrim folder

    subprocess.call(["rsync","-auEv","-e",
                     "'ssh -l mariecombe -i /Users/mariecombe/.shh/id_dsa'",
                     "--delete",
                     "/Users/mariecombe/Documents/Work/Research_project_3/pcse/pickled_CGMS_input_data/",
                     "mariecombe@capegrim.wur.nl:~/mnt/promise/CO2/marie/pickled_CGMS_input_data/"])

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
