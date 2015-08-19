#!/usr/bin/env python

import pcse
from pcse.db.cgms11 import WeatherObsGridDataProvider, TimerDataProvider, SoilDataIterator, CropDataProvider, STU_Suitability, SiteDataProvider
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider
import matplotlib.pyplot as plt
import sqlalchemy as sa
from sqlalchemy import MetaData, select, Table
import pandas as pd
print "Imported successfully: PCSE, numpy and SQLAlchemy, the cx_Oracle database"
print "connector needed by SQLAlchemy, and the python modules:  tabulate, matplotlib,"
print "and pandas\n"

from datetime import date
import math
import sys, os, getopt
import csv
import string
import numpy as np
from matplotlib import pyplot

import cPickle as pickle

import cx_Oracle


#----------------------------------------------------------------------
def open_eurostat_csv(inpath,filelist):

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        print "Opening %s"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv.reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[0]
        print headerow

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]
        #del lines[-1]

        # transforming data from string to float type, storing it in array 'data'
        converted_data=[]
        for line in lines:
            if (line[4] != ':'): 
                a = line[0:4] + [float(string.replace(line[4], ' ', ''))] + [line[5]]
            else:
                a = line[0:4] + [float('NaN')] + [line[5]]
            converted_data.append(a)
        data=np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        
        Dict[namefile] = dictnamelist
    
        print "Dictionary created!\n"

    return Dict

#----------------------------------------------------------------------
def open_csv_as_strings(inpath,namefilelist):

    import csv

    Dict = {}

    for i,namefile in enumerate(namefilelist):
        
        print "Opening %s"%(namefile)
        
        inputpath=os.path.join(inpath,namefile)
        f=open(inputpath,'rU')
        reader=csv.reader(f, delimiter=',', skipinitialspace=True)
        all=[]
        for row in reader:
            all.append(row)
        headerow=all[0]
        del all[0]
        print headerow
    
        datafloat=[]
        for row in all:
            a = row[0:2]
            datafloat.append(a)
        data=np.array(datafloat)
    
        dictnamelist = {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
        
        print "Dictionary created!\n"

    return Dict   

#----------------------------------------------------------------------
def retrieve_most_present_and_suitable_soil_types_in_grid(e, g, c):

    soil_input_data = SoilDataIterator(e, g)

    # Check for suitable soil types
    suitable_stu = STU_Suitability(e, c)

    # make a list of soil weights in the grid cell
    list_w = []
    for smu_no, area, stu_no, percentage, soildata in soil_input_data:
        if stu_no not in suitable_stu: continue
        weight_factor = area * percentage
        list_w = list_w + [weight_factor]
    list_w_sorted = sorted(list_w)

    # retrieve the soil type number
    for smu_no, area, stu_no, percentage, soildata in soil_input_data:

        # remove all unsuitable soils from the iteration
        if stu_no not in suitable_stu: continue
    
        # run the model for the most present soil type in the grid cell
        weight_factor = area * percentage
        if (weight_factor == np.max(list_w)):
            most_present = stu_no
            soildata1 = soildata
        elif (weight_factor == list_w_sorted[-2]):
            second_present = stu_no
            soildata2 = soildata
    
    return most_present, second_present, soildata1, soildata2

#----------------------------------------------------------------------
def retrieve_most_present_and_suitable_soil_types_in_grid_NO_DB(g, c, s, suit):

    # make a list of soil weights in the grid cell
    list_w = []
    for smu_no, area, stu_no, percentage, soildata in s:
        if stu_no not in suit: continue
        weight_factor = area * percentage
        list_w = list_w + [weight_factor]
    list_w_sorted = sorted(list_w)

    # retrieve the soil type number
    for smu_no, area, stu_no, percentage, soildata in s:

        # remove all unsuitable soils from the iteration
        if stu_no not in suit: 
	    continue
        else:
            weight_factor = area * percentage
            if (weight_factor == np.max(list_w)):
                m1 = stu_no
                soildata1 = soildata
            elif (weight_factor == list_w_sorted[-2]):
                m2 = stu_no
                soildata2 = soildata
    
    return  m1, m2, soildata1, soildata2

#----------------------------------------------------------------------
def find_level3_regions(connection, reg_code):
    """Returns the level3 regions for given region code."""

    cursor = connection.cursor()

    sql = """SELECT reg_map_id, reg_level FROM region where reg_code = '%s'""" % reg_code
    cursor.execute(sql)
    row = cursor.fetchone()
    if not row:
        msg = "Failed to retrieved ID of region '%s'" % reg_code
        raise RuntimeError(msg)
    reg_map_id, reg_level = row

    sql = """
        select
          reg.level_3_reg_map_id
        from
            (select level_0.reg_map_id as level_0_reg_map_id,
                    level_1.reg_map_id as level_1_reg_map_id,
                    level_2.reg_map_id as level_2_reg_map_id,
                    level_3.reg_map_id as level_3_reg_map_id
             from
                    (select reg_map_id, reg_map_id_bt, reg_name from region where reg_level = 0) level_0
                inner join
                    (select reg_map_id, reg_map_id_bt, reg_name from region where reg_level = 1) level_1
                  on level_0.reg_map_id = level_1.reg_map_id_bt
                inner join
                    (select reg_map_id, reg_map_id_bt, reg_name from region where reg_level = 2) level_2
                  on level_1.reg_map_id = level_2.reg_map_id_bt
                inner join
                    (select reg_map_id, reg_map_id_bt, reg_name from region where reg_level = 3) level_3
                  on level_2.reg_map_id = level_3.reg_map_id_bt) reg
        where
          reg.level_%1i_reg_map_id = %i
        """ % (reg_level, reg_map_id)

    cursor.execute(sql)
    rows = cursor.fetchall()
    l3_regions = [row[0] for row in rows]

    return l3_regions

#----------------------------------------------------------------------
def find_complete_grid_cells_in_regions(connection, regions):
    """Return the list of grid that are fully contained with the list of regions
    """
    cursor = connection.cursor()

    sql_regions = str(tuple(regions))
    sql = """
        select
          s.grid_no
        from
          (select
                 t1.grid_no, sum(t1.area) as sum_area
               from
                 link_emu_region t1
          where
            t1.reg_map_id in %s
          group by
            t1.grid_no) s
        where
          s.sum_area > 624999990
    """ % sql_regions
    cursor.execute(sql)
    rows = cursor.fetchall()
    grids = [row[0] for row in rows]
    return grids

#----------------------------------------------------------------------
def find_grids_with_arable_land(connection, grids, threshold=None, largest_n=None):
    """Find the grids with either
    1 ) an amount of arable land defined by threshold in m2
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
    
#----------------------------------------------------------------------
def SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_reg_code,_threshold=None,_largest_n=None):
    
    connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")

    #NUTS_reg_code = 'ES41'
    regions = find_level3_regions(connection, NUTS_reg_code)
    #print regions
    complete_grids = find_complete_grid_cells_in_regions(connection, regions)
    if _threshold is not None:
        r = find_grids_with_arable_land(connection, complete_grids, threshold=_threshold) #=156250000)
	crit_grid_selec = 'above_%i'%_threshold
	print 'we select cells with arable land > %i m2!'%_threshold
    elif _largest_n is not None:
        r = find_grids_with_arable_land(connection, complete_grids, largest_n=_largest_n)
	crit_grid_selec = 'top_%i'%_largest_n
	print 'we select %i top cells!'%_largest_n
    else:
        r = find_grids_with_arable_land(connection, complete_grids)
	crit_grid_selec = 'all'
	
    #for grid_no, area in r:
    #    print("grid_no: %08i, area: %012i" % (grid_no, area))
    
    return r,crit_grid_selec

#----------------------------------------------------------------------
def fetch_crop_name(engine, crop_no):
    """Retrieves the name of the crop from the CROP table for
    given crop_no.

    :param engine: SqlAlchemy engine object providing DB access
    :param crop_no: Integer crop ID, maps to the CROP_NO column in the table
    """
    metadata = MetaData(engine)
    table_crop = Table("crop", metadata, autoload=True)
    r = select([table_crop],
               table_crop.c.crop_no == crop_no).execute()
    row = r.fetchone()
    r.close()
    if row is None:
        msg = "Failed deriving crop name from CROP table for crop_no %s" % crop_no
        raise exc.PCSEError(msg)
    return row.crop_name

#----------------------------------------------------------------------

def detrend_obs_yields(full_region_name, crop_name, uncorrected_yields_dict, DM_content, reference_year):

    campaign_years = np.linspace(1975,2014,40)
    nb_years = 40
    OBS = {}
    TREND = {}
    for i,val in enumerate(campaign_years): # searching for the index of the reference year
        if val == reference_year:
            indref = i
    
    # retrieve the observed yields over that period of time
    TARGET = np.array([0.]*nb_years)
    for j,year in enumerate(campaign_years):
        for i,region in enumerate(uncorrected_yields_dict['GEO']):
            if region.startswith(full_region_name[0:12]):
                if uncorrected_yields_dict['CROP_PRO'][i]==crop_name:
                    if (uncorrected_yields_dict['TIME'][i]==str(int(year))):
                        if (uncorrected_yields_dict['STRUCPRO'][i]=='Yields (100 kg/ha)'):
                            TARGET[j] = float(uncorrected_yields_dict['Value'][i])*10.*DM_content
    print 'observed dry matter yields:', TARGET

    # fit a linear trend line
    mask = ~np.isnan(TARGET)
    z = np.polyfit(campaign_years[mask], TARGET[mask], 1)
    p = np.poly1d(z)
    OBS['ORIGINAL'] = TARGET[mask]
    TREND['ORIGINAL'] = p(campaign_years)
    
    # calculate the anomalies to the trend line
    ANOM = TARGET - (z[0]*campaign_years + z[1])
    
    # Detrend the data
    OBS['DETRENDED'] = ANOM[mask] + p(reference_year)
    z2 = np.polyfit(campaign_years[mask], OBS['DETRENDED'], 1)
    p2 = np.poly1d(z2)
    TREND['DETRENDED'] = p2(campaign_years)
    
    # plot before and after de-trending
    pyplot.close('all')
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
    fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
    for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
        ax.scatter(campaign_years[mask], OBS[var], c='b')
        ax.plot(campaign_years,TREND[var],'r-')
        ax.set_ylabel('%s yield (gDM m-2)'%var, fontsize=14)
        ax.set_xlabel('time (year)', fontsize=14)
    fig.savefig('observed_yields.png')
#    pyplot.show()
    #print 'y=%.6fx+(%.6f)'%(z[0],z[1])
    
    print 'detrended dry matter yields:', OBS['DETRENDED']
    
    return OBS['DETRENDED'], campaign_years[mask]


#######################################################################
########################### START OF SCRIPT ###########################

if __name__ == "__main__":
    
    try:                                
        opts, args = getopt.getopt(sys.argv[1:], "-h")
    except getopt.GetoptError:           
        print "Error"
        sys.exit(2)      
    
    for options in opts:
        options=options[0].lower()
        if options == '-h':
            helptext = """
    This script execute the WOFOST runs for one location

                """
            
            print helptext
            
            sys.exit(2)      

    # Define directories

    currentdir = os.getcwd()
    EUROSTATdir = '/Users/mariecombe/Documents/Work/Research project 3/EUROSTAT_data'
    folderpickle = 'pickled_CGMS_input_data/'
    #folderpickle = '/Storage/CO2/mariecombe/pickled_CGMS_input_data'

    # we establish a connection to the remote Oracle database
    
    user = "cgms12eu_select"
    password = "OnlySelect"
    tns = "EURDAS.WORLD"
    # tns = IP_address:PORT/SID
    dsn = "oracle+cx_oracle://{user}:{pw}@{tns}".format(user=user, pw=password, tns=tns)
    engine = sa.create_engine(dsn)
    print engine
    
    # we retrieve observed datasets
#    
#    NUTS_data = open_eurostat_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014/agri_yields_NUTS1-2-3_1975-2014.csv'])
#    NUTS_ids = open_csv_as_strings(EUROSTATdir,['NUTS_codes_2013.csv'])
#    # simplifying the dictionaries keys:
#    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014/agri_yields_NUTS1-2-3_1975-2014.csv']
#    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
#    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014/agri_yields_NUTS1-2-3_1975-2014.csv']
#    del NUTS_ids['NUTS_codes_2013.csv']
    
    # we define the settings of our runs:
    
####################################### USER DEFINED: #######################################

    # define the case:
        
    NUTS_no = 'NL1'
    crop_no = 7
    campaign_years = np.linspace(1975,2014,40)

    crit_selec = 'all' # 'all', 'top_n' or 'threshold'  : the selection criteria of the grid cells to retrieve input data for

    if crit_selec == 'top_n':
        __largest_n = 10
    if crit_selec == 'threshold':
        __threshold = 468750000

    pickle_input = True # if True: will retrieve the weather, soil and crop calendar data
    

    # useless settings:

#    YLDGAPF_sims = False
#    YLDGAPF_iter = False
#    
#    yldgapf_range = np.linspace(0.1,1.,5.)
#    nb_yldgapf = 5
#    NUTS_name = 'Castilla y Leon'
#    crop_name = 'Barley' #fetch_crop_name(engine, 3)
#    DM_content = 0.9
    
#############################################################################################

    # We retrieve all the grid cells that have at least 25% of arable land on their total area:

    if crit_selec == 'all': #retrieves all
        grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no) 
    elif crit_selec == 'top_n':
        grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no,_largest_n=__largest_n) #retrieves top 10 in arable land
    elif crit_selec == 'threshold':
        grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no,_threshold=__threshold) #retrieves cells with more than threshold arable land

    
    filename = folderpickle+'gridlistobject_%s_r%s.pickle'%(grid_list_tuples[1],NUTS_no)

    if os.path.exists(filename):
	grid_list_tuples = pickle.load(open(filename,'rb'))
        print 'Reading data from pickle file %s'%filename

    else:
        if crit_selec == 'all': #retrieves all
            grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no) 
        elif crit_selec == 'top_n':
            grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no,_largest_n=__largest_n) #retrieves top 10 in arable land
        elif crit_selec == 'threshold':
            grid_list_tuples = SQL_querie_select_arable_land_grid_cells_in_NUTS_region(NUTS_no,_threshold=__threshold) #retrieves cells with more than threshold arable land
        pickle.dump(grid_list_tuples,open(filename,'wb'))       
        print 'Writing data to pickle file %s'%filename

    grid_list = [g for g,a in grid_list_tuples[0]]
    print 'list of selected grids:', grid_list
  
    
    # We select the soil types for which we run the model
    
    # Check for suitable soil types for the chosen crop
    filename = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no

    if os.path.exists(filename):
	suitable_stu = pickle.load(open(filename,'rb'))
        print 'Reading data from pickle file %s'%filename

    else:
        suitable_stu = STU_Suitability(engine, crop_no)
	suitable_stu_list = []
	for item in suitable_stu:
	    suitable_stu_list = suitable_stu_list + [item]
	suitable_stu = suitable_stu_list
        pickle.dump(suitable_stu,open(filename,'wb'))       
        print 'Writing data to pickle file %s'%filename
    

#    # Retrieve all the major soil types to loop over
#    selected_soils = {}
#
#    for grid,area_arable in grid_list_tuples:
#    
#        list_soils = []
#    
#        # Retrieve all soil types 
#        soil_iterator = SoilDataIterator(engine, grid)
#
#        # loop over the soil types
#        for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:
#
#            # NB: we remove all unsuitable soils from the iteration
#            if stu_no not in suitable_stu: 
#                continue
#            else:
#                # calculate the cultivated area of the crop in this grid
#                area_soil      = area_smu*percentage/100.
#                perc_area_soil = area_soil/625000000
#                
#                if (perc_area_soil <= 0.05):
#                    continue
#                else:
#                    list_soils = list_soils + [(smu_no,stu_no)]
#            
#        selected_soils[grid] = list_soils
#
#    print "Completed the selection of soils"
    
    

    # Detrend the observed yields
#    detrend = detrend_obs_yields(NUTS_name, crop_name, NUTS_data['yields'], DM_content, 2000)


    if (pickle_input == True):

	for grid,area_arable in grid_list_tuples[0]:

	    print 'grid %i'%grid
	    filename = folderpickle+'weatherobject_g%d.pickle'%grid

	    if os.path.exists(filename):
	        print '%s exists!'%filename

	    else:
	    	weatherdata = WeatherObsGridDataProvider(engine, grid)
		print 'Writing data to pickle file %s'%filename
		weatherdata._dump(filename)

	    filename = folderpickle+'soilobject_g%d.pickle'%grid

	    if os.path.exists(filename):
	        print '%s exists!'%filename
	    	soil_iterator = pickle.load(open(filename,'rb'))

	    else:
	    	soil_iterator = SoilDataIterator(engine, grid)
		print 'Writing data to pickle file %s'%filename
		pickle.dump(soil_iterator,open(filename,'wb'))       

	    
            # loop over the soil types
            for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:

                # NB: we remove all unsuitable soils from the iteration
                if (stu_no not in suitable_stu):
                    continue
                else:
                    
		    print 'soil type no %i'%stu_no
		    
	            for y, year in enumerate(campaign_years): 
    
                        # Retrieve yearly data 
                        filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

                        if os.path.exists(filename):
                            print '%s exists!!'%filename

                        else:
                            timerdata = TimerDataProvider(engine, grid, crop_no, year)
                            print 'Writing data to pickle file %s'%filename
                            pickle.dump(timerdata,open(filename,'wb'))       
                            
                        filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)

                        if os.path.exists(filename):
                            print '%s exists!!'%filename

                        else:
                            cropdata = CropDataProvider(engine, grid, crop_no, year)
                            print 'Writing data to pickle file %s'%filename
                            pickle.dump(cropdata,open(filename,'wb'))     
				  
                        filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid,crop_no,year,stu_no)

                        if os.path.exists(filename):
                            print '%s exists!!'%filename

                        else:
                            sitedata = SiteDataProvider(engine, grid, crop_no, year, stu_no)
                            print 'Writing data to pickle file %s'%filename
                            pickle.dump(sitedata,open(filename,'wb'))     

                   



#    if (YLDGAPF_sims == True):
#    
#        # open an empty file
#	
#        resfile = 'Optimized_yields_'+NUTS_no+'_'+crop_name+'.dat'
#        if (os.path.isfile(os.path.join(currentdir,resfile))): os.remove(os.path.join(currentdir,resfile))
#        Results = open(os.path.join(currentdir,resfile), 'w')
#        Results.write("   NUTS_no,   YLDGAPF,   grid_id, area_arable,      year,    smu_no,   area_smu,    stu_no, perc_soil, TSO(gDM.m-2), TARGET(gDM.m-2),       GAP\n")
#
#        for f,factor in enumerate(yldgapf_range):
#        
#	    print factor
#
#            for grid,area_arable in grid_list_tuples:
#
#	        weatherdata = WeatherObsGridDataProvider(engine, grid)
#                soil_iterator = SoilDataIterator(engine, grid)
#
#                for y, year in enumerate(campaign_years): 
#    
#                    print year
#		
#                    # Retrieve all soil types and data 
#                    timerdata = TimerDataProvider(engine, grid, crop_no, year)
#                    cropdata = CropDataProvider(engine, grid, crop_no, year)
#	            cropdata['YLDGAPF']=factor 		    
#		    
#
#                    # loop over the soil types
#                    for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:
#
#                        # NB: we remove all unsuitable soils from the iteration
#                        if stu_no not in suitable_stu: 
#                            continue
#                        else:
#            
#                            # calculate the cultivated area of the crop in this grid
#                            area_soil      = area_smu*percentage/100.
#                            perc_area_soil = area_soil/625000000
#                
#                            if (perc_area_soil <= 0.5):
#                                continue
#                            else:
#
#                                # define the model run settings that are dependent on the stu_no
#                                sitedata = SiteDataProvider(engine, grid, crop_no, year, stu_no)
#                   
#                                # run WOFOST
#                                wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, cropdata, weatherdata)
#                                wofost_object.run_till_terminate()
#    
#                                # get the yield (in gDM.m-2) and the yield gap
#                                TSO = wofost_object.get_variable('TWSO')
#		                DIFF = TSO - detrend[0][y]
#            
#                                # write the output in a file
#                                Results.write("%10s,%10.3f,%10i,%12i,%10i,%10i,%11i,%10i,%10.2f,%13.2f,%16.2f,%10.2f\n"%(NUTS_no, factor, grid, area_arable, year, smu_no, area_smu, stu_no, percentage/100., TSO,detrend[0][y],DIFF))
#
#        Results.close()
#        print "Results file closed!"
#
#    elif (YLDGAPF_iter == True):
#    
#        FINAL_YLDGAPF = []
#	for grid,area_arable in grid_list_tuples:
#
#	    print 'grid %i'%grid
#	    filename = folderpickle+'weatherobject_g%d.pickle'%grid
#
#	    if os.path.exists(filename):
#	        weatherdata = WeatherDataProvider()
#		print 'Reading data from pickle file %s'%filename
#		weatherdata._load(filename)
#
#	    else:
#	    	weatherdata = WeatherObsGridDataProvider(engine, grid)
#		print 'Writing data to pickle file %s'%filename
#		weatherdata._dump(filename)
#
#		#pickle.dump(weatherdata,open(filename,'wb'))
# 	    
#	    filename = folderpickle+'soilobject_g%d.pickle'%grid
#
#	    if os.path.exists(filename):
#	    	soil_iterator = pickle.load(open(filename,'rb'))
#	   	print 'Reading data from pickle file %s'%filename
#
#	    else:
#	    	soil_iterator = SoilDataIterator(engine, grid)
#		print 'Writing data to pickle file %s'%filename
#		pickle.dump(soil_iterator,open(filename,'wb'))       
#
#	    
#
#            # we retrieve the most present soil type
#            top1_stu_no = retrieve_most_present_and_suitable_soil_types_in_grid_NO_DB(grid, crop_no, soil_iterator, suitable_stu)
#
#            # loop over the soil types
#            for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator:
#
#                # NB: we remove all unsuitable soils from the iteration
#                if (stu_no != top1_stu_no[0]):
#                    continue
#                else:
#                    
#		    print 'top soil no %i'%stu_no
#		    
#                    lowestf  = yldgapf_range[0]
#                    highestf = yldgapf_range[-1]
#		    f_step   = highestf - lowestf
#	
#	            while (f_step >= 0.1):
#	    
#                        print '...'
#	                # we build a range of YLDGAPF to explore until we obtain a good enough precision on the YLDGAPF
#	    
#	                f_step = (highestf - lowestf)/2.
#	                middlef = lowestf + f_step
#	                f_range = [lowestf, middlef, highestf]
#
#                        RES = {}
#	                for y, year in enumerate(campaign_years): 
#    
#                            # Retrieve yearly data 
#                            filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)
#
#                            if os.path.exists(filename):
#                                timerdata = pickle.load(open(filename,'rb'))
#                                print 'Reading data from pickle file %s'%filename
#
#                            else:
#                                timerdata = TimerDataProvider(engine, grid, crop_no, year)
#                                print 'Writing data to pickle file %s'%filename
#                                pickle.dump(timerdata,open(filename,'wb'))       
#                            
#                            filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'%(grid,crop_no,year)
#
#                            if os.path.exists(filename):
#                                cropdata = pickle.load(open(filename,'rb'))
#                                print 'Reading data from pickle file %s'%filename
#
#                            else:
#                                cropdata = CropDataProvider(engine, grid, crop_no, year)
#                                print 'Writing data to pickle file %s'%filename
#                                pickle.dump(cropdata,open(filename,'wb'))     
#				  
#                            filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid,crop_no,year,stu_no)
#
#                            if os.path.exists(filename):
#                                sitedata = pickle.load(open(filename,'rb'))
#                                print 'Reading data from pickle file %s'%filename
#
#                            else:
#                                sitedata = SiteDataProvider(engine, grid, crop_no, year, stu_no)
#                                print 'Writing data to pickle file %s'%filename
#                                pickle.dump(sitedata,open(filename,'wb'))     
#
#	                    DIFF = np.array([0.]*len(f_range))
#			    for f,factor in enumerate(f_range):
#        
#			        cropdata['YLDGAPF']=factor
#                   
#                                # run WOFOST
#                                wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, cropdata, weatherdata)
#                                wofost_object.run_till_terminate()
#    
#                                # get the yield (in gDM.m-2) and the yield gap
#                                TSO = wofost_object.get_variable('TWSO')
#		                DIFF[f] = TSO - detrend[0][y]
#				
#		            RES[year] = DIFF
#				
#	                # Calculate RMSE (root mean squared error):
#			    
#	                list_of_DIFF = []
#	                RMSE = np.array([0.]*len(f_range))
#	                for f,factor in enumerate(f_range):
#    	                    for y, year in enumerate(campaign_years):
#				list_of_DIFF = list_of_DIFF + [RES[year][f]]
#			    RMSE[f] = np.sqrt(np.mean( [ math.pow(j,2) for j in list_of_DIFF ] ))
#				
#			# linear interpolation of RMSE between the 3 data points
#			    
#			RMSE_midleft  = (RMSE[0] + RMSE[1]) / 2.
#			RMSE_midright = (RMSE[1] + RMSE[2]) / 2.
#			    
#			# we update the YLDGAPF range to explore for the next round
#			    	
#			if (RMSE_midleft <= RMSE_midright):
#	        
#			    lowestf = lowestf
#	        	    highestf = middlef
#	    
#	    		elif (RMSE_midleft > RMSE_midright):
#            
#			    lowestf = middlef
#	       		    highestf = highestf
#                        
#		    print "optimized factor!\n"
#		    FINAL_YLDGAPF = FINAL_YLDGAPF + [(grid, middlef)]
#	    
#        pickle.dump(FINAL_YLDGAPF, open( "list_of_yldgapf.pickle", "wb"))



#    # Calculate RMSE (root mean squared error):
#    list_of_DIFF = []
#    RMSE = np.array([0.]*nb_yldgapf)
#    for i,val in enumerate(yldgapf_range):
#        for y, year in enumerate(campaign_years):
#            list_of_DIFF = list_of_DIFF + [RES[year][i]]
#        RMSE[i] = np.sqrt(np.mean( [ math.pow(j,2) for j in list_of_DIFF ] ))
#
#    # Calculate MAE (mean absolute error):
#    list_of_DIFF = []
#    MAE = np.array([0.]*nb_yldgapf)
#    for i,val in enumerate(yldgapf_range):
#        for y, year in enumerate(campaign_years):
#            list_of_DIFF = list_of_DIFF + [RES[year][i]]
#        MAE[i] = np.mean( [ math.fabs(j) for j in list_of_DIFF ] )

