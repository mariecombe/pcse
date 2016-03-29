#!/usr/bin/env python

import sys, os
import numpy as np

# This script maps NUTS ids to EUROSTAT region names, and crop ids to crop names 

#===============================================================================
def main():
#===============================================================================
    """
	This scripts constructs a dictionary of NUTS_ids <--> NUTS names, of
    crop_ids <--> crop names, and a list of years, which are all short-selected 
    by the user, and stores them in pickle files in the current directory.

    SELECTION OF NUTS REGIONS:
    --------------------------

    1: we read the NUTS id codes from the shapefile we use to plot NUTS regions
       we select only the codes that correspond to the NUTS levels we are
       interested in (NUTS 1 or 2, it varies per country).
       ==> we obtain a shortlist of NUTS 1 and 2 id codes we want to do 
       simulations for.

    2: we assign a corrected NUTS name (no accents, no weird alphabets), and a
       latin-1 encoded NUTS name (same encoding as the EUROSTAT observation 
       files) to each NUTS code of the dictionary.

    SELECTION OF CROPS:
    -------------------

    3: we simply manually map the EUROSTAT crop names to each crop code

    SELECTION OF YEARS:
    -------------------

    4: we create a list of (integer) years

    STORING IN MEMORY:
    ------------------

    4: we pickle the produced dictionaries/lists and store them in the 
       current directory

    """
#-------------------------------------------------------------------------------
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    from maries_toolbox import get_crop_names
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    levels_method = 'all' # can be 'all' or 'composite'
                             # if == composite: the script makes a composite of
                             # NUTS 1 and 2 regions to select, as specified 
                             # by the user with variable 'lands_levels'
                             # if == 'all': all NUTS 0-1-2 regions will be 
                             # selected

    # for each country code, we say which NUTS region level we want to 
    # consider (for some countries: level 1, for some others: level 2)
    lands_levels = {'AT':1,'BE':1,'BG':2,'CH':1,'CY':2,'CZ':1,'DE':1,'DK':2,
                    'EE':2,'EL':2,'ES':2,'FI':2,'FR':2,'HR':2,'HU':2,'IE':2,
                    'IS':2,'IT':2,'LI':1,'LT':2,'LU':1,'LV':2,'ME':1,'MK':2,
                    'MT':1,'NL':1,'NO':2,'PL':2,'PT':2,'RO':2,'SE':2,'SI':2,
                    'SK':2,'TR':2,'UK':1} # NB: only works with levels_method = 'composite'

    # list of selected crops of interest:
    crops = ['Winter wheat']#,'Spring wheat','Winter wheat',
            # 'Spring barley','Winter barley','Spring rapeseed','Winter rapeseed',
            # 'Rye','Potato','Sugar beet','Sunflower','Field beans']

    # list of selected years to simulate the c cycle for:
    years = [2000] #list(np.arange(2000,2015,1))#[2006]#,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]

    # input directory path
    inputdir = '/Users/mariecombe/mnt/promise/CO2/wofost/'

    # If you want to check if we match the crop and region names in the EUROSTAT
    # files, set the following files to true, it will print the result to screen
    check_eurostat_file = False

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    currentdir    = os.getcwd()
    EUROSTATdir   = os.path.join(inputdir,'EUROSTATobs')
#-------------------------------------------------------------------------------
# create a temporary directory if it doesn't exist
    if not os.path.exists("../tmp"):
        os.makedirs("../tmp")

#-------------------------------------------------------------------------------
# 1- WE CREATE A DICTIONARY OF REGIONS IDS AND NAMES TO LOOP OVER
#-------------------------------------------------------------------------------
# Select the regions ID in which we are interested in
    # we read the NUTS 0-1-2 codes from pre-processed yield file
    dum = pickle_load(open(os.path.join(EUROSTATdir,'preprocessed_yields.pickle'),'rb'))
    all_NUTS_regions = sorted(dum['Winter wheat'].keys())
    all_NUTS_regions2 = list()
    for reg in all_NUTS_regions:
        # remove the bogus NUTS regions (--Z or --ZZ), regions outside 
        # the European domain (tropical islands), and the EU region
        if not ((reg.endswith('Z') and len(reg) >= 3) or \
        reg.startswith('FRA') or reg.startswith('FR9') or \
        reg.startswith('PT2') or reg.startswith('EU')): 
           all_NUTS_regions2 += [reg]

    # we select a subset of regions to work with:
    if (levels_method == 'all'):
        NUTS_regions = all_NUTS_regions2
        print '\nWe select all NUTS levels (0-1-2): %d'%len(NUTS_regions)
    if (levels_method == 'composite'):
        NUTS_regions = make_NUTS_composite(lands_levels, all_NUTS_regions2)
        print '\nWe select a shortlist of NUTS 0/1/2 regions (user-defined): %d'%len(NUTS_regions)

    print sorted(NUTS_regions)
    NUTS_names_dict = NUTS_regions
#-------------------------------------------------------------------------------
# pickle the produced dictionary in the current directory:
    pathname = os.path.join(currentdir, '../tmp/selected_NUTS_regions.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(NUTS_names_dict, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 2- WE CREATE A DICTIONARY OF CROP IDS AND NAMES TO LOOP OVER
#-------------------------------------------------------------------------------
# Create a dictionary of crop EUROSTAT names, corresponding to the selected crops
    crop_names_dict = get_crop_names(crops, method='short')
    print '\nWe select the following crops:\n', sorted(crop_names_dict.keys())
#-------------------------------------------------------------------------------
# pickle the produced dictionary:
    pathname = os.path.join(currentdir, '../tmp/selected_crops.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(crop_names_dict, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 3- WE CREATE A LIST OF YEARS TO LOOP OVER?
#-------------------------------------------------------------------------------
# pickle the produced dictionary:
    print '\nWe select the following years:\n', years
    pathname = os.path.join(currentdir, '../tmp/selected_years.pickle')
    if os.path.exists(pathname): os.remove(pathname)
    pickle_dump(years, open(pathname,'wb'))

#-------------------------------------------------------------------------------
# 4- FOR INFORMATION: WE CHECK IF WE MATCH THE EUROSTAT NAMES
#-------------------------------------------------------------------------------
    if check_eurostat_file == True:
        check_EUROSTAT_names_match(NUTS_names_dict, crop_names_dict, EUROSTATdir)


#===============================================================================
def make_NUTS_composite(NUTS_levels, all_regions):
#===============================================================================

    NUTS_ids_list = list()

    for country in sorted(NUTS_levels.keys()):
        level = NUTS_levels[country]
        subset_NUTS = [id for id in all_regions if (id[0:2]==country) and 
                       (len(id)<=(level+2))]
        NUTS_ids_list += subset_NUTS

    return NUTS_ids_list

#===============================================================================
def check_EUROSTAT_names_match(NUTS_names_dict, crop_names_dict, EUROSTATdir):
#===============================================================================
    from maries_toolbox import open_csv_EUROSTAT

	# read the yield/ landuse / pop records, try to match previous name with
	# names in this file
    NUTS_filenames = ['agri_yields_NUTS1-2-3_1975-2014.csv',
                     'agri_croparea_NUTS1-2-3_1975-2014.csv',
                     'agri_landuse_NUTS1-2-3_2000-2013.csv',
                     'agri_prodhumid_NUTS1_1955-2015.csv',
                     'agri_harvest_NUTS1-2-3_1975-2014.csv']
    NUTS_data = open_csv_EUROSTAT(EUROSTATdir, NUTS_filenames,
                                     convert_to_float=True, verbose=False)

    # read the NUTS region labels for each EUROSTAT csv file:
    geo_dict = dict()
    crop_dict = dict()
    for record in NUTS_filenames:
        if (record != 'agri_prodhumid_NUTS1_1955-2015.csv'):
            geo_units = NUTS_data[record]['GEO']
            geo_units = list(set(geo_units))
            geo_dict[record] = [u.lower() for u in geo_units]
            #print geo_units, len (geo_units)
        if (record != 'agri_landuse_NUTS1-2-3_2000-2013.csv'):
            crop_names = NUTS_data[record]['CROP_PRO']
            crop_names = list(set(crop_names))
            crop_dict[record] = [u.lower() for u in crop_names]
            #print record, crop_names, len (crop_names)

	# Check if we match all EUROSTAT region names, in all the EUROSTAT
	# observation files:
    for record in NUTS_filenames:
        print '\nChecking EUROSTAT file %s'%record
        if (record != 'agri_prodhumid_NUTS1_1955-2015.csv'):
            counter = 0
            for key in NUTS_names_dict.keys():
                if (NUTS_names_dict[key][1].lower() not in geo_dict[record]):
                    counter +=1
                    if counter ==1: print '        NUTS ID,   Corrected name,   EUROSTAT name'
                    print 'NOT OK: %5s'%key, NUTS_names_dict[key]
            print 'found %i unmatched region names in EUROSTAT file'%counter
        if (record != 'agri_landuse_NUTS1-2-3_2000-2013.csv'):
            if (record == 'agri_prodhumid_NUTS1_1955-2015.csv'):
                counter = 0
                for key in crop_names_dict.keys():
                    if (crop_names_dict[key][2].lower() not in crop_dict[record]):
                        counter +=1
                        if counter ==1: print '        crop ID,   Corrected name,   EUROSTAT name'
                        print 'NOT OK: %5s'%key, crop_names_dict[key]
                print 'found %i unmatched crop names in EUROSTAT file'%counter
            else:
                counter = 0
                for key in crop_names_dict.keys():
                    if (crop_names_dict[key][1].lower() not in crop_dict[record]):
                        counter +=1
                        if counter ==1: print '        crop ID,   Corrected name,   EUROSTAT name'
                        print 'NOT OK: %5s'%key, crop_names_dict[key]
                print 'found %i unmatched crop names in EUROSTAT file'%counter

    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
