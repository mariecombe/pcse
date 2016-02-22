#!/usr/bin/env python

import sys, os
import numpy as np

# This script gathers a few useful general functions used by several scripts

#===============================================================================
def get_list_CGMS_cells_in_Europe_arable(gridcells, lons, lats):
#===============================================================================
    """
    This function returns the list of CGMS European grid cells that contain 
    an amount of arable land > 0. It checks the arable land data in the CGMS
    Oracle database.

    Function arguments:
    ------------------
    all_grids    list of integers, contains CGMS grid cell ID numbers for which
                 we want to check if they contain any arable land
    lons         list of floats, contains the longitudes of the grid cells
    lats         list of floats, contains the latitudes of the grid cells

    NB: this function needs to connect to the CGMS Oracle database, hence an 
    ethernet connection within the Wageningen University network is required!

    """
    import cx_Oracle
    from operator import itemgetter as operator_itemgetter

    # test the connection:
    try:
        connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")
    except cx_Oracle.DatabaseError:
        print '\nBEWARE!! The Oracle database is not responding. Probably, you are'
        print 'not using a computer wired within the Wageningen University network.'
        print '--> Get connected with ethernet cable before trying again!'
        sys.exit(2)

    # initialize lists of grid cells
    europ        = np.array([]) # empty array
    europ_arable = list() # empty array

    # for each grid cell of the CGMS database:
    for i,grid_no in enumerate(gridcells):

        # if the grid cell is located in Europe:
        if ((-13.<= lons[i] <= 70.) and (34 <= lats[i] <= 71)):

            # we append the grid cell no to the list of European grid cells:
            europ = np.append(europ,grid_no)

    # now we want to get the list of European grid cells that contains arable land:
    # NB: SQL can only access the info of 1000 grids at a time. We got about
    # 20000 cells to loop over, so we do 20 times the query to get their arable land

    bounds = np.arange(0,len(europ),1000) # bounds of grid cell ID
    # in this 19 iteration, we do not explore ALL grid cells, we must do a last
    # iteration manually (see below)
    for i in range(len(bounds)-1):
        subset = europ[bounds[i]:bounds[i+1]]
        subset_arable = find_grids_with_arable_land(connection, subset)
        # we remove grid_no duplicates, but we conserve the tuple format:
        subset_arable = dict((i[0],i) for i in subset_arable).values()
        # we order the list by decreasing amount of arable land area
        subset_arable = sorted(subset_arable,key=operator_itemgetter(1),reverse=True)
        # we store the list of tuples:
        europ_arable += subset_arable

    # we are still missing the last grid cells: do they have arable land?
    subset = europ[bounds[-1]:len(europ)]
    subset_arable = find_grids_with_arable_land(connection, subset)
    # we remove grid_no duplicates, but we conserve the tuple format:
    subset_arable = dict((i[0],i) for i in subset_arable).values()
    # we order the list by decreasing amount of arable land area
    subset_arable = sorted(subset_arable,key=operator_itemgetter(1),reverse=True)
    # we store the list of tuples:
    europ_arable += subset_arable 

    assert (len(europ) >= len(europ_arable)), 'increased the nb of grid cells???'
    print '\nWe retrieved %i grid cell ids with arable '%len(europ_arable)+\
          'land in Europe.\n'

    return europ_arable

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
def querie_arable_cells_in_NUTS_region(NUTS_reg_code,_threshold=None,_largest_n=None):
#===============================================================================
    
    import cx_Oracle

    # test the connection:
    try:
        connection = cx_Oracle.connect("cgms12eu_select/OnlySelect@eurdas.world")
    except cx_Oracle.DatabaseError:
        print '\nBEWARE!! The Oracle database is not responding. Probably, you are'
        print 'not using a computer wired within the Wageningen University network.'
        print '--> Get connected with ethernet cable before trying again!'
        sys.exit(2)

    # 1- retrieve the NUTS 3 region ID forming the desired NUTS 2 region
    try:
        # the CGMS database uses the 2006 EUROSTAT nomenclature for NUTS_ids
        # we correct a few NUTS names before trying to retrieve information there
        # 1- Greece old country code was 'GR' before it became 'EL'
        if NUTS_reg_code.startswith('EL'):
            NUTS_reg_code = 'GR'+NUTS_reg_code[2:len(NUTS_reg_code)]
        # 3- Italy old code 'ITD' became 'ITH' in 2010
        if NUTS_reg_code.startswith('ITH'):
            NUTS_reg_code = 'ITD'+NUTS_reg_code[3:len(NUTS_reg_code)]
        # 4- Italy old code 'ITE' became 'ITI' in 2010
        if NUTS_reg_code.startswith('ITI'):
            NUTS_reg_code = 'ITE'+NUTS_reg_code[3:len(NUTS_reg_code)]

        regions = find_level3_regions(connection, NUTS_reg_code)
    except Exception as e:
        print 'Region id does not exist?', e
        return None,'all'

    # 2- get the grid cells that are complete
    try:
        complete_grids = find_complete_grid_cells_in_regions(connection, regions)
    # if no cells are wholely contained in the region:
    except cx_Oracle.DatabaseError:
        print "No grid cells are entirely contained in this region. Return: None."
        return None,'all'

    # 3- get the grid cells with arable land
    if _threshold is not None:
        r = find_grids_with_arable_land(connection, complete_grids, threshold=_threshold)
        crit_grid_selec = 'above_%i'%_threshold
        print 'we select cells with arable land > %i m2!'%_threshold
    elif _largest_n is not None:
        r = find_grids_with_arable_land(connection, complete_grids, largest_n=_largest_n)
        crit_grid_selec = 'top_%i'%_largest_n
        print 'we select %i top cells!'%_largest_n
    else:
        crit_grid_selec = 'all'
        try:
            r = find_grids_with_arable_land(connection, complete_grids)
        except cx_Oracle.DatabaseError:
            print "No whole grid cells have arable land in this region. Return: None."
            r = None
	
    return r,crit_grid_selec

#===============================================================================
def find_level3_regions(connection, reg_code):
#===============================================================================
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

#===============================================================================
def find_complete_grid_cells_in_regions(connection, regions):
#===============================================================================
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

#===============================================================================
# Function to select a subset of grid cells within a NUTS region
def select_cells(NUTS_no, crop_no, year, folder_pickle, method='topn', n=3, 
                                                          select_from='arable'):
#===============================================================================
    '''
    This function selects a subset of grid cells contained in a NUTS region
    This function should be used AFTER having listed the available CGMS grid 
    cells of a NUTS region with Allard's SQL routine.

    The function returns a list of (grid_cell_id, arable_land_area) tuples that 
    are sorted by arable_land_area.

    Function arguments:
    ------------------
    NUTS_no        is a string and is the NUTS region code number

    folder_pickle  is a string and is the path to the folder where the CGMS 
                   input pickle files are located (the pickled lists of 
                   available grid cells). To produce these input files, use the
                   script CGMS_input_files_retrieval.py

    method         is a string specifying how we select our subset of grid cells.
                   if 'all': we select all grid cells from the list.
                   if 'topn': we select the top n cells from the list that 
                   contain the most arable land.
                   if 'randomn': we select n grid cells randomly from the list.
                   method='topn' by default.

    n              is an integer specifying the number of grid cells to select 
                   for the method 'topn' and 'randomn'. n=3 by default.

    '''
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    from random import sample as random_sample

    if (select_from == 'arable'):    
        # we first read the list of all 'whole' grid cells of that region
        # NB: list_of_tuples is a list of (grid_cell_id, arable_land_area)
        # tuples, which are already sorted by decreasing amount of arable land
        filename = os.path.join(folder_pickle,'gridlist_objects/',
                   'gridlistobject_all_r%s.pickle'%NUTS_no)
        try:
            NUTS_arable = pickle_load(open(filename, 'rb'))
        except IOError:
            NUTS_arable = querie_arable_cells_in_NUTS_region(NUTS_no)
            pickle_dump(NUTS_arable, open(os.path.join(filename), 'wb'))
        # we select the first item from the file, which is the actual list of tuples
        list_of_tuples = NUTS_arable[0]

    elif (select_from == 'cultivated'):
        # first get the arable cells contained in NUTS region
        filename = os.path.join(folder_pickle,'gridlist_objects/',
                   'gridlistobject_all_r%s.pickle'%NUTS_no)
        try:
            NUTS_arable = pickle_load(open(filename, 'rb'))
        except IOError:
            NUTS_arable = querie_arable_cells_in_NUTS_region(NUTS_no)
            pickle_dump(NUTS_arable, open(os.path.join(filename), 'wb'))
        # then read the European cultivated cells for that year and crop
        filename = os.path.join(folder_pickle,'cropdata_objects/',
                   'cropmask_c%i.pickle'%crop_no)
        culti_cells = pickle_load(open(filename,'rb'))
        # get only the intersection, i.e. the cultivated cells in NUTS region:
        list_of_tuples = list()
        if (NUTS_arable[0] != None):
            for cell in NUTS_arable[0]:
                if cell[0] in [c for c,a in culti_cells[year]]:
                    list_of_tuples += [cell]
        else:
            return None

    if len(list_of_tuples)==0: 
        return None

    # first option: we select the top n grid cells in terms of arable land area
    if (method == 'topn'):
        subset_list   = list_of_tuples[0:n]
    # second option: we select a random set of n grid cells
    elif (method == 'randomn'):
        try: # try to sample n random soil types:
            subset_list   = random_sample(list_of_tuples,n)
        except: # if an error is raised ie. sample size bigger than population 
            subset_list   = list_of_tuples
    # last option: we select all available grid cells of the region
    else:
        subset_list   = list_of_tuples

    #print '\nWe have selected', len(subset_list),'grid cells:',\
    #          [g for g,a in subset_list]

    return subset_list

#===============================================================================
# Function to select a subset of suitable soil types for a list of grid cells
def select_soils(crop_no, grid_cells, folder_pickle, method='topn', n=3):
#===============================================================================
    '''
    This function selects a subset of suitable soil types for a given crop from 
    all the ones contained in a CGMS grid cell. This function should be used 
    AFTER having listed the available CGMS grid cells of a NUTS region, and its
    soil types, with Allard's SQL routine.

    The function returns a dictionary of lists of (smu_no, stu_no, stu_area, 
    soil_data) sets, one list being assigned per grid cell number.

    Function arguments:
    ------------------
    crop_no        is an integer and is the CGMS crop ID number.

    grid_cells     is the list of selected (grid_cell_id, arable_land_area)
                   tuples for which we want to select a subset of soil types.

    folder_pickle  is a string and is the path to the folder where the CGMS 
                   input pickle files are located (the pickled lists of 
                   suitable soil types and lists of soil type data). To produce
                   these input files, use the script CGMS_input_files_retrieval.py

    method         is a string specifying how we select our subset of soil types.
                   if 'all': we select all suitable soil types from the list.
                   if 'topn': we select the top n soil types from the list with 
                   the biggest soil type area.
                   if 'randomn': we select n soil types randomly from the list.

    n              is an integer specifying the number of soil types to select 
                   for the method 'topn' and 'randomn'. n=3 by default.

    '''

    from random import sample as random_sample
    from operator import itemgetter as operator_itemgetter
    from cPickle import load as pickle_load

    # we first read the list of suitable soil types for our chosen crop 
    filename = os.path.join(folder_pickle,'soildata_objects/',
               'suitablesoilsobject_c%d.pickle'%crop_no)
    suitable_soils = pickle_load(open(filename,'rb')) 
 
    dict_soil_types = {}
 
    #for grid in [g for g,a in grid_cells]:
    for grid in grid_cells:
 
        # we read the list of soil types contained within the grid cell
        filename = os.path.join(folder_pickle,'soildata_objects/',
                   'soilobject_g%d.pickle'%grid)
        soil_iterator_ = pickle_load(open(filename,'rb'))
 
        # Rank soils by decreasing area
        sorted_soils = []
        for smu_no, area_smu, stu_no, percentage_stu, soildata in soil_iterator_:
            if stu_no not in suitable_soils: continue
            weight_factor = area_smu * percentage_stu/100.
            sorted_soils = sorted_soils + [(smu_no, stu_no, weight_factor, 
                                                                       soildata)]
        sorted_soils = sorted(sorted_soils, key=operator_itemgetter(2),
                                                                    reverse=True)
       
        # select a subset of soil types to loop over 
        # first option: we select the top n most present soils in the grid cell
        if   (method == 'topn'):
            subset_list   = sorted_soils[0:n]
        # second option: we select a random set of n soils within the grid cell
        elif (method == 'randomn'):
            try: # try to sample n random soil types:
                subset_list   = random_sample(sorted_soils,n)
            except: # if sample size bigger than population, do:
                subset_list   = sorted_soils
        # last option: we select all available soils in the grid cell
        else:
            subset_list   = sorted_soils

        dict_soil_types[grid] = subset_list

        #print 'We have selected',len(subset_list),'soil types:',\
        #       [stu for smu, stu, w, data in subset_list],'for grid', grid
 
    return dict_soil_types

#===============================================================================
# retrieve the crop fraction from EUROSTAT data
def get_EUR_frac_crop(crop_name, NUTS_no, EUROSTATdir_, campaign_years):
#===============================================================================

    import math
    from cPickle import load as pickle_load

    # retrieve the region's crop area and arable area for the years specified by
    # the user (i.e. campaign_years)
    cultiareas = pickle_load(open(os.path.join(EUROSTATdir_,
                                      'preprocessed_culti_areas.pickle'),'rb'))
    arablareas = pickle_load(open(os.path.join(EUROSTATdir_,
                                     'preprocessed_arable_areas.pickle'),'rb'))
    crop_area   = cultiareas[crop_name][NUTS_no]
    arable_area = arablareas[crop_name][NUTS_no]

    # we calculate frac_crop for the years where we both have observations of 
    # arable land AND crop area
    frac_crop = np.array([])
    years     = np.array([])
    for year in campaign_years:
        if ((float(year) in crop_area[1]) and (float(year) in arable_area[1])):
            indx_crop = np.array([math.pow(j,2) for j in \
                                                  (crop_area[1]-year)]).argmin()
            indx_arab = np.array([math.pow(j,2) for j in \
                                                (arable_area[1]-year)]).argmin()
            frac_crop = np.append(frac_crop, crop_area[0][indx_crop] / \
                                                      arable_area[0][indx_arab])
            years     = np.append(years, year)
        else:
            years     = np.append(years, np.nan)
            frac_crop = np.append(frac_crop, np.nan)

    # for years without data, we try to gap-fill the record of cultivated fractions
    # using different methods
    non_masked_len = np.ma.MaskedArray.count(np.ma.masked_invalid(years))
    if non_masked_len < len(campaign_years) and non_masked_len > 4:
        # if we have 5 points or more, we use a linear trend
        print 'we gap fill the cultivated fraction with a linear trend over years'
        mask = ~np.isnan(np.array(years))
        z = np.polyfit(years[mask], frac_crop[mask], 1)
        p = np.poly1d(z)
        for y,year in enumerate(years):
            if np.isnan(year):
                frac_crop[y] = p(campaign_years[y])
        return frac_crop, campaign_years

    elif non_masked_len <= 4 and non_masked_len > 0:
        # if we have 4 points or less, we use the mean
        clim_mean = np.ma.mean(np.ma.masked_invalid(frac_crop))
        print 'we gap fill the cultivated fraction with the mean over years:'+\
              ' %.4f'%clim_mean
        frac_crop = np.array([clim_mean]*len(campaign_years))
        return frac_crop, campaign_years

    elif non_masked_len == 0:
        # if we have no points at all, we use the upper level of NUTS records
        # (i.e. NUTS1 instead of NUTS2, or NUTS0 instead of NUTS1)
        if len(NUTS_no)>2:
            if len(NUTS_no)==3:  level=1
            elif len(NUTS_no)>3: level=2
            print 'NUTS%i records are empty. We use NUTS%i '%(level,level-1) +\
                  'records instead.'
            upper_NUTS_frac = get_EUR_frac_crop(crop_name, NUTS_no[0:level+1],
                  EUROSTATdir_, campaign_years)
            return upper_NUTS_frac
        else:
        # if all gap-filling has failed, we signal the user to skip this entire 
        # country for the optimization, by returning None 
            print 'This country (%s) has no reported cultivated '%NUTS_no+\
                  'fraction in the EUROSTAT records'
            return None

    # if all the years of data are available, return the yearly ratio of crop to 
    # arable land, and the full list of years 
    else:
        print 'all observed years are available in the cultivated area records'
        return frac_crop, campaign_years

#===============================================================================
# Select the list of years over which we optimize the yldgapf
def define_opti_years(opti_year_, obs_years):
#===============================================================================

    if opti_year_ in obs_years:
        # if the year is available in the record of observed yields, we
        # use that particular year to optimize the yldgapf
        print 'We optimize fgap on year', opti_year_
        opti_years = [opti_year_]
    else:
        # if we don't have yield observations for that year, we use the
        # most recent 3 years of observations available to optimize the 
        # yldgapf. The generated yldgapf will be used as proxy of the 
        # opti_year yldgapf
        opti_years = find_consecutive_years(obs_years, 3)
        print 'We use', opti_years, 'as proxy for', opti_year

    return opti_years

#===============================================================================
# Return a list of consecutive years longer than n items
def find_consecutive_years(years, nyears):
#===============================================================================

    # Split the list of years where there are gaps
    years = map(int, years) # convert years to integers
    split_years = np.split(years, np.where(np.diff(years) > 1)[0]+1)

    # Return the most recent group of years that contains at least nyears items
    consecutive_years = np.array([])
    for subset in split_years[::-1]: # [::-1] reverses the array without 
                                     # modifying it permanently
        if (len(subset) >= nyears):
            consecutive_years = np.append(consecutive_years, subset)
            break
        else:
            pass

    # return the last nyears years of the most recent group of years
    return consecutive_years[-nyears:len(consecutive_years)]

#===============================================================================
def fetch_EUROSTAT_NUTS_name(NUTS_no, EUROSTATdir):
#===============================================================================
   
    from csv import reader as csv_reader

# read the EUROSTAT file containing the NUTS codes and names

    # open file, read all lines
    inputpath = os.path.join(EUROSTATdir,'NUTS_codes_2013.csv')
    f=open(inputpath,'rU') 
    reader=csv_reader(f, delimiter=',', skipinitialspace=True)
    lines=[]
    for row in reader:
        lines.append(row)
    f.close()

    # storing headers in list headerow
    headerow=lines[0]

    # deleting rows that are not data
    del lines[0]

    # we keep the string format, we just separate the string items
    dictnames = dict()
    for row in lines:
        #dictnames[row[2]] = row[3]
        dictnames[row[0]] = row[1]

# we fetch the NUTS region name corresponding to the code

    #print "EUROSTAT region name of %s:"%NUTS_no, dictnames[NUTS_no]
    
    return dictnames

#===============================================================================
# Function to detrend the observed EUROSTAT yields or harvests
def detrend_obs( _NUTS_no, _crop_name, 
                uncorrected_yields, _DM_content, base_year=2000,
                obs_type='yield', detrend=True, prod_fig=False, verbose=True):
#===============================================================================

    years = np.array(sorted([float(y) for y in set(uncorrected_yields['Year'])]))

    # select if to detrend yields or harvest:
    if (obs_type == 'yield'):
        header_to_search = 'Yield (100 kg/ha)'
        conversion_factor = 100.*_DM_content
        obs_unit = 'kgDM ha-1'
    elif (obs_type == 'harvest'):
        header_to_search = 'Harvested production (1000 t)'
        conversion_factor = _DM_content
        obs_unit = '1000 tDM'
    elif (obs_type == 'culti_area'):
        header_to_search = 'Area (cultivation/harvested/production) (1000 ha)'
        conversion_factor = 1.
        obs_unit = '1000 ha'
    elif (obs_type == 'arable_area'):
        header_to_search = 'Main area (1000 ha)'
        conversion_factor = 1.
        obs_unit = '1000 ha'
   
    # select yields for the required region, crop and period of time
    # and convert them from kg_humid_matter/ha to kg_dry_matter/ha 
    TARGET = np.array([-9999.]*len(years))
    if (verbose==True): print 'searching for:', _NUTS_no
    print obs_type

    # we preprocess ALL data available from the files
    for j,year in enumerate(sorted(set(uncorrected_yields['Year']))):

        # loop over all the lines of the file:
        for i,region in enumerate(uncorrected_yields['NUTS_id']):

            # if we match the NUTS region code
            if (region == _NUTS_no):
                # if we match the year
                if (uncorrected_yields['Year'][i] == year):
                   # print uncorrected_yields['Crop_name'][i], _crop_name
                    # if we match the crop name
                    if uncorrected_yields['Crop_name'][i] == _crop_name:
                        #print uncorrected_yields['Variable'][i], header_to_search
                        # if we match the variable to search
                        if (uncorrected_yields['Variable'][i]==
                                                      header_to_search):
                            if (verbose==True): print year,\
                            _NUTS_no, uncorrected_yields['Value'][i]
                            TARGET[j] = float(uncorrected_yields['Value'][i])\
                                              *conversion_factor

    if (detrend==True):

        # search for the index of base_year item in the years array
        for r,val in enumerate(years): 
            if val == base_year:
                indref = r

        # fit a linear trend line in the record of observed yields
        OBS = {}
        TREND = {}
        mask = ~np.isnan(TARGET)
        z = np.polyfit(years[mask], TARGET[mask], 1)
        p = np.poly1d(z)
        OBS['ORIGINAL'] = TARGET[mask]
        TREND['ORIGINAL'] = p(years)
        
        # calculate the anomalies to the trend line
        ANOM = TARGET - (z[0]*years + z[1])
        
        # Detrend the observed yield data
        OBS['DETRENDED'] = ANOM[mask] + p(base_year)
        z2 = np.polyfit(years[mask], OBS['DETRENDED'], 1)
        p2 = np.poly1d(z2)
        TREND['DETRENDED'] = p2(years)
    else:
        # no detrending, but we apply a mask still?
        OBS = {}
        mask = ~np.isnan(TARGET)
        OBS['ORIGINAL'] = TARGET[mask]
        
    
    # if needed plot a figure showing the yields before and after de-trending
    if prod_fig==True:
        from matplotlib import pyplot as plt
        plt.close('all')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
        for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
            ax.scatter(years[mask], OBS[var], c='b')
       	    ax.plot(years,TREND[var],'r-')
       	    ax.set_ylabel('%s %s (%s)'%(var,obs_type,obs_unit), fontsize=14)
            ax.set_xlabel('time (year)', fontsize=14)
        fig.savefig('observed_%ss.png'%obs_type)
        print 'the trend line is y=%.6fx+(%.6f)'%(z[0],z[1])
        plt.show()
    
    if (detrend==True):
        obstoreturn = OBS['DETRENDED']
        if (verbose==True): print '\nsuccesfully detrended the %ss!'%obs_type, obstoreturn
    else: 
        obstoreturn = OBS['ORIGINAL']
        print '\nno detrending! returning the original %s'%obs_type, obstoreturn, years[mask]

    return obstoreturn, years[mask]

#===============================================================================
# Function to retrieve the dry matter content of a given crop in a given
# country (this is based on EUROSTAT crop humidity data)
def retrieve_crop_DM_content(crop, NUTS_no_, EUROSTATdir):
#===============================================================================

    from cPickle import load as pickle_load

    # we retrieve the dry matter of a specific crop and country, over the 
    # years 1955-2015
    DM_obs = pickle_load(open(os.path.join(EUROSTATdir, 
                         'preprocessed_obs_DM.pickle'), 'rb'))
    DM_content = DM_obs[crop][NUTS_no_[0:2]]
    # if the retrieved array is not empty, and if it's not sugar beet or potato
    # (we do not trust the reported DM of these crops) then we use the average 
    # reported DM:
    if (np.isnan(DM_content).all() == False) and \
        ((crop!='Sugar beet') or (crop!='Potato')):
        DM_content = np.ma.mean(np.ma.masked_invalid(DM_content))
        print '\nWe use the observed DM content', DM_content
    # otherwise we use the standard EUROSTAT DM content for that crop:
    else:
        DM_standard = pickle_load(open(os.path.join(EUROSTATdir, 
                                'preprocessed_standard_DM.pickle'),'rb'))
        DM_content = DM_standard[crop]
        print '\nWe use the standard DM content', DM_content

    return DM_content

#===============================================================================
# Function to open normal csv files
def open_pcse_csv_output(inpath,filelist):
#===============================================================================

    from csv import reader as csv_reader
    from datetime import date
    from string import split

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        #print "\nOpening %s......"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv_reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[18]

        # getting summary output
        crop_yield = lines[13][0].split(':')
        crop_yield = float(crop_yield[1])


        # deleting rows that are not data (first and last rows of the file)
        del lines[0:19]

        # transforming data from string to float type
        converted_data=[]
        for line in lines:
            datestr = split(line[0], '-')
            a = [date(int(datestr[0]),int(datestr[1]),int(datestr[2])), \
                 float(line[1]), float(line[2]), float(line[3]), float(line[4]), \
                 float(line[5]), float(line[6]), float(line[7]), float(line[8]), \
                 float(line[9])]#, float(line[10]), float(line[11])]
            converted_data.append(a)
        data = np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        #print "Dictionary created!"

    return Dict, crop_yield

#===============================================================================
# Function to open normal csv files
def open_csv(inpath,filelist,convert_to_float=False):
#===============================================================================

    from csv import reader as csv_reader

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        #print "\nOpening %s......"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv_reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[0]

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]

        # transforming data from string to float type
        converted_data=[]
        for line in lines:
            converted_data.append(map(float,line))
        data = np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        #print "Dictionary created!"

    return Dict

#===============================================================================
# Function to open EUROSTAT csv files
def open_csv_EUROSTAT(inpath,filelist,convert_to_float=False,verbose=True,
                                                                data_year=2015):
#===============================================================================

    from csv import reader as csv_reader
    from string import replace as string_replace

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        if (verbose == True): print "\nOpening %s......"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv_reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[0]

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]

        # two possibilities: either convert data from string to float or
        # keep it as is in string type
        if (convert_to_float == True):
            # transforming data from string to float type
            converted_data=[]
            for line in lines:
                if data_year == 2015:
                    if (line[4] != ':'): 
                        a = (line[0:4] + [float(string_replace(line[4], ' ', ''))] 
                                       + [line[5]])
                    else:
                        a = line[0:4] + [float('NaN')] + [line[5]]
                elif data_year == 2016:
                    if (line[5] != ':'): 
                        a = (line[0:5] + [float(string_replace(line[5], ' ', ''))] 
                                       + [line[6]])
                    else:
                        a = line[0:5] + [float('NaN')] + [line[6]]
                converted_data.append(a)
            data = np.array(converted_data)
        else:
            # we keep the string format, we just separate the string items
            datafloat=[]
            for row in lines:
                a = row[0:2]
                datafloat.append(a)
            data=np.array(datafloat)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        if (verbose == True): print "Dictionary created!"

    return Dict

#===============================================================================
def get_crop_names(crop_list, method='short'):
#===============================================================================
    """
    This creates a dictionary 
    - either of ALL WOFOST crops to read the EUROSTAT csv files (method 'all')
    - or of a selected shortlist (method 'short')

	The script will associate CGMS crop names with a CGMS_id and EUROSTAT crop
    name crops[CGMS_name] = [CGMS_id, EUROSTAT_name]

    NB: these EUROSTAT crop names are only valid to use with the EUROSTAT files 
    that were updated on feb 2016 (stored in download_2016/ folder)

    """
    crops = dict()
    crops['Winter wheat']    = [1,'Common winter wheat and spelt']
    crops['Spring wheat']    = [np.nan,'Common spring wheat and spelt']
    crops['Grain maize']     = [2,'Grain maize and corn-cob-mix']
    crops['Fodder maize']    = [12,'Green maize']
    crops['Spring barley']   = [3,'Spring barley']
    crops['Winter barley']   = [13,'Winter barley']
    crops['Rye']             = [4,'Rye']
    crops['Sugar beet']      = [6,'Sugar beet (excluding seed)']
    crops['Potato']          = [7,'Potatoes (including seed potatoes)']
    crops['Field beans']     = [8,'Broad and field beans']
    crops['Spring rapeseed'] = [np.nan,'Spring rape and turnip rape seeds']
    crops['Winter rapeseed'] = [10,'Winter rape and turnip rape seeds']
    crops['Sunflower']       = [11,'Sunflower seed']
    crops['Durum wheat']     = [41,'Durum wheat']
    crops['Triticale']       = [43,'Triticale']
    crops['Rapeseed and turnips'] = [46,'Rape and turnip rape seeds']

    if method=='short':
        for key in crops.keys():
            if key not in crop_list:
                del crops[key]

    return crops

