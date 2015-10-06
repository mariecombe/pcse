#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

#===============================================================================
# This script retrieves input data (soil, crop, timer and site) from the CGMS 
# database and transfers it to capegrim
def main():
#===============================================================================
    import cx_Oracle
    from cPickle import dump as pickle_dump
    from datetime import datetime
    from maries_toolbox import open_csv
    from pcse.db.cgms11 import TimerDataProvider, SoilDataIterator, \
                               CropDataProvider, STU_Suitability, \
                               SiteDataProvider
#-------------------------------------------------------------------------------
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
    
    user = "cgms12eu_select"
    password = "OnlySelect"
    tns = "EURDAS.WORLD"
    dsn = "oracle+cx_oracle://{user}:{pw}@{tns}".format(user=user, pw=password, 
                                                                        tns=tns)
    try:
        engine = sa.create_engine(dsn)
        print engine
    except RuntimeError:
        print 'BEWARE!! You are not using a computer within the WU network,'
        print 'this is why you cannot access the Oracle CGMS database.'
        print 'Fix this before trying again!'
        sys.exit(2)

#-------------------------------------------------------------------------------
# We add a timestamp at start of the retrieval

    start_timestamp = datetime.utcnow()

#-------------------------------------------------------------------------------
# we read the list of CGMS grid cells from file

    all_CGMS_grid_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])

#-------------------------------------------------------------------------------
# we get the list of European grid cells that contain arable land

    connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")
    europ = np.array([]) # empty array

    # for each grid cell of the CGMS database:
    for i,grid_no in enumerate(all_CGMS_grid_cells):

        # if the grid cell is located in Europe:
        if ((-13.<= lon[i] <= 70.) and (34 <= lat[i] <= 71)):

            # we append the grid cell no to the list of European grid cells:
            europ.append(grid_no)

    # we get the list of European grid cells that contains arable land:
    europ_arable = find_grids_with_arable_land(connection, europ)

#-------------------------------------------------------------------------------
# We retrieve the list of suitable soil types for the selected crop species
    
    filename = folder_local + 'suitablesoilsobject_c%d.pickle'%crop_no

    if os.path.exists(filename):
        print '%s exists!'%filename
    else:
        suitable_stu = STU_Suitability(engine, crop_no)
        suitable_stu_list = []
        for item in suitable_stu:
            suitable_stu_list = suitable_stu_list + [item]
        suitable_stu = suitable_stu_list
        print 'Writing data to pickle file %s'%filename
        pickle.dump(suitable_stu,open(filename,'wb'))       

#-------------------------------------------------------------------------------
#   WE LOOP OVER ALL EUROPEAN GRID CELLS
    for grid in [grid_no for grid_no,area in europ_arable]:
	    print 'grid cell no %i'%grid
#-------------------------------------------------------------------------------
# If required by the user, we retrieve the weather data (1 file per grid cell):

        if retrieve_weather == True: 
            filename = folderpickle+'weatherobject_g%d.pickle'%grid

            if os.path.exists(filename):
                print '%s exists!'%filename
            else:
                weatherdata = WeatherObsGridDataProvider(engine, grid)
                print 'Writing data to pickle file %s'%filename
                weatherdata._dump(filename)

#-------------------------------------------------------------------------------
# We retrieve the soil data (1 file per grid cell):
 
        filename = folderpickle+'soilobject_g%d.pickle'%grid

        if os.path.exists(filename):
            print '%s exists!'%filename
        else:
            soil_iterator = SoilDataIterator(engine, grid)
            print 'Writing data to pickle file %s'%filename
            pickle.dump(soil_iterator,open(filename,'wb'))       

#-------------------------------------------------------------------------------
#       WE LOOP OVER ALL SOIL TYPES LOCATED IN THE GRID CELL:
        for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:
#-------------------------------------------------------------------------------

            # NB: we remove all unsuitable soils from the iteration
            if (stu_no not in suitable_stu):
                continue
            else:
                print 'soil type no %i'%stu_no

#-------------------------------------------------------------------------------
#               WE LOOP OVER ALL YEARS:
                for y, year in enumerate(campaign_years): 
#-------------------------------------------------------------------------------
# We retrieve the timer data

                    filename = folder_local + \
                            'timerobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

                    if os.path.exists(filename):
                        print '%s exists!!'%filename
                    else:
                        timerdata = TimerDataProvider(engine, grid, crop_no, year)
                        print 'Writing data to pickle file %s'%filename
                        pickle.dump(timerdata,open(filename,'wb'))       
    
#-------------------------------------------------------------------------------
# We retrieve the crop data

                    filename = folder_local + \
                             'cropobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

                    if os.path.exists(filename):
                        print '%s exists!!'%filename
                    else:
                        cropdata = CropDataProvider(engine, grid, crop_no, year)
                        print 'Writing data to pickle file %s'%filename
                        pickle.dump(cropdata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
# We retrieve the site data

                    filename = folder_local + \
                             'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid,crop_no,
                                                                  year,stu_no)

                    if os.path.exists(filename):
                        print '%s exists!!'%filename
                    else:
                        sitedata = SiteDataProvider(engine, grid, crop_no, year, 
                                                                         stu_no)
                        print 'Writing data to pickle file %s'%filename
                        pickle.dump(sitedata,open(filename,'wb'))     

#-------------------------------------------------------------------------------
# We add a timestamp at end of the retrieval, to time the process

    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#-------------------------------------------------------------------------------
# We sync the local folder containing the pickle files with the capegrim folder



#===============================================================================
def find_grids_with_arable_land(connection, grids, threshold=None, largest_n=None):
#===============================================================================
    """
    Find the grids with either
    1) an amount of arable land defined by threshold in m2
       (max for a 25km grid cell is 625000000 m2)
    2) the largest_n number of cells with largest share of arable land
    3) just all grids with the amount of arable land

    returns a list of [(grid_no1, area), (grid_no2, area), ...]
    """

    landcover = 101 # see below for other options from CROP_LANDCOVER table
    # 101	Arable Land	0	0	0
    # 102	Non-irrigated arable land	0	0	0
    # 103	Agricultural areas	0	0	0
    # 104	Pasture	0	0	0
    # 105	Temporary forage	0	0	0
    # 106	Rice	0	0	0
    # 100	Any land cover class	0	0	0

    cursor = connection.cursor()
    gridlist = str(tuple(grids))
    if threshold is not None:
        thr = float(threshold)
        sql = """
            select
                grid_no, area
            from
                link_region_grid_landcover
            where
                area > {threshold}f and
                landcover_id = {lc} and
                grid_no in {gridl}
            order by area desc
        """.format(gridl=gridlist, threshold=thr, lc=landcover)
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows
    elif largest_n is not None:
        ln = int(largest_n)
        sql = """
            select
                grid_no, area
            from
                link_region_grid_landcover
            where
                landcover_id = {lc} and
                grid_no in {grids}
            order by area desc
        """.format(grids=gridlist, nrows=ln, lc=landcover)
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows[0:ln]
    else:
        sql = """
            select
                grid_no, area
            from
                link_region_grid_landcover
            where
                landcover_id = {lc} and
                grid_no in {grids}
            order by area desc
        """.format(grids=gridlist, lc=landcover)
        cursor.execute(sql)
        rows = cursor.fetchall()
        return rows
    
#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
