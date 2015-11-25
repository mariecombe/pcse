#!/usr/bin/env python

import sys, os
import numpy as np

# This script preprocesses the EUROSTAT observations

#===============================================================================
def main():
#===============================================================================
    """
    This script does not detrend observations, or convert humid matter to dry
    matter for yields and harvests. It just extracts raw observations into a 
    readable dictionary for future use.

    IMPORTANT: This script does not depend on the selection of crop species
    done by running 01_select_crops_n_regions.py!! We pre-process EUROSTAT data
    for ALL crops here, once and for all.
    NB: However, we do use the selection of NUTS regions defined by running
    01_select_crops_n_regions.py!

    To renew the preprocessed observations, delete the corresponding pickle files
    located in the ../model_input_data/ directory, and then re-launch the pre-
    processing.

    """
    from cPickle import load as pickle_load
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, pickledir, crop_dict
#-------------------------------------------------------------------------------
# Define working directories
    currentdir   = os.getcwd()
    EUROSTATdir  = '../observations/EUROSTAT_data/'
    pickledir    = '../model_input_data/'
#-------------------------------------------------------------------------------
# NB: we will loop over ALL crops available, because we want to do the 
# pre-processing once and for all!
    crop_dict    = all_crop_names()
#-------------------------------------------------------------------------------
# 1- PRE-PROCESS EUROSTAT CSV FILES ON AREAS, YIELDS, HARVESTS:
#-------------------------------------------------------------------------------

    try:
        culti_areas  = pickle_load(open(pickledir+'preprocessed_culti_areas.pickle','rb'))
        arable_areas = pickle_load(open(pickledir+'preprocessed_arable_areas.pickle','rb'))
        yields       = pickle_load(open(pickledir+'preprocessed_yields.pickle','rb'))
        harvests     = pickle_load(open(pickledir+'preprocessed_harvests.pickle','rb'))
        print '\nThe yield, harvest, area pickle files exist! Pre-processing aborted.'
    except IOError: # if the files don't exist, we create them:
        print 'WARNING! You are launching the pre-processing of EUROSTAT data'
        print 'this procedure can take up to 20 hours!!!'
        preprocess_EUROSTAT_data()
    except Exception as e:
        print 'Unexpected error:', e

#-------------------------------------------------------------------------------
# 2- PRE-PROCESS EUROSTAT CSV FILE ON DRY MATTER CONTENT:
#-------------------------------------------------------------------------------

    try:
        DM           = pickle_load(open(pickledir+'preprocessed_obs_DM.pickle','rb'))
        DM_standard  = pickle_load(open(pickledir+'preprocessed_standard_DM.pickle','rb'))
        print '\nThe crop humidity content files exist! Pre-processing aborted.'
    except IOError: # if the files don't exist, we create them:
        print '\nNow we launch the pre-processing of the dry matter content data.'
        preprocess_EUROSTAT_DM()
    except Exception as e:
        print 'Unexpected error:', e

    print '\nThe pre-processing of EUROSTAT data is over.'

#===============================================================================
def preprocess_EUROSTAT_DM(plot=False):
#===============================================================================
    from cPickle import load as pickle_load
    from maries_toolbox import open_csv_EUROSTAT
#-------------------------------------------------------------------------------
# we retrieve the record of observed dry matter content

    pathname = os.path.join(pickledir,'preprocessed_obs_DM.pickle')

    if (os.path.exists(pathname)):
        DM = pickle_load(open(pathname, 'rb'))
    else:
        NUTS_data = open_csv_EUROSTAT(EUROSTATdir,['agri_prodhumid_NUTS1_1955-2015.csv'],
                             convert_to_float=True)
        DM = pickle_DM(NUTS_data['agri_prodhumid_NUTS1_1955-2015.csv'], pathname)

#-------------------------------------------------------------------------------
# We retrieve the standard EUROSTAT dry matter information in case there are
# gaps in the observations

    pathname = os.path.join(pickledir,'preprocessed_standard_DM.pickle')

    if (os.path.exists(pathname)):
        DM_standard = pickle_load(open(pathname, 'rb'))
    else:
        DM_standard = pickle_DM_standard(pathname)

#-------------------------------------------------------------------------------
# We can plot a time series of one crop x country DM combination if needed

    if (plot==True):
        list_of_years = np.linspace(1955,2015,61)
        time_series = plot_DM_time_series(DM, list_of_years, 3, 'ES')


#===============================================================================
# Function to store EUROSTAT standard moisture contents in pickle file
def pickle_DM_standard(picklepathname):
#===============================================================================
    from cPickle import dump as pickle_dump

    DM_standard = dict()
    DM_standard['Spring wheat']    = 1. - 0.14
    DM_standard['Winter wheat']    = 1. - 0.14
    DM_standard['Grain maize']     = 1. - 0.14
    DM_standard['Fodder maize']    = 1. - 0.65 # Fodder maize: 50 to 80%, very variable
    DM_standard['Spring barley']   = 1. - 0.14
    DM_standard['Winter barley']   = 1. - 0.14
    DM_standard['Rye']             = 1. - 0.14
    DM_standard['Sugar beet']      = 1. - 0.80 # Sugar beet: no idea
    DM_standard['Potato']          = 1. - 0.80 # Potato: no idea
    DM_standard['Field beans']     = 1. - 0.14
    DM_standard['Spring rapeseed'] = 1. - 0.09
    DM_standard['Winter rapeseed'] = 1. - 0.09
    DM_standard['Sunflower']       = 1. - 0.09

    # we pickle the generated dictionnary containing the dry matter content
    pickle_dump(DM_standard, open(picklepathname, 'wb'))

    return DM_standard

#===============================================================================
# Function to retrieve EUROSTAT humidity content and to store it in pickle file
def pickle_DM(obs_data, picklepathname):
#===============================================================================
    from cPickle import dump as pickle_dump
    from datetime import datetime

    # EUROSTAT lists of crop names and country names
    list_of_crops_EUR     = sorted(set(obs_data['CROP_PRO']))
    list_of_countries_EUR = sorted(set(obs_data['GEO']))
    list_of_years         = sorted(set(obs_data['TIME']))

    # CGMS list of crops was read from file at the beginning of the main() 
    # method
    list_of_crops_CGMS    = crop_dict.keys()

    # we translate the EUROSTAT list of countries into official country codes
    # European & ISO 3166 nomenclature + temporary country code for Kosovo XK
    list_of_countries_CGMS = ['AL','AT','BE','BA','BG','HR','CY','CZ','DK',
                              'EE','EU','FI','MK','FR','DE','EL','HU','IS',
                              'IE','IT','XK','LV','LI','LT','LU','MT','ME',
                              'NL','NO','PL','PT','RO','RS','SK','SI','ES',
                              'SE','CH','TR','UK']

    # we print out some info for the user
    print '\nWe will loop over:\n'
    print list_of_crops_CGMS
    print list_of_countries_CGMS
#-------------------------------------------------------------------------------
    # We add a timestamp at start of the retrieval
    start_timestamp = datetime.utcnow()
#-------------------------------------------------------------------------------

	# we build a dictionnary where we will store the dry matter content per
	# crop and country CGMS ids

    DM = dict()
    for crop in crop_dict.keys():
        DM[crop] = dict()
        print crop
        for k, idc in enumerate(list_of_countries_CGMS):
            print '    ', idc
            DM[crop][idc] = np.zeros(len(list_of_years))
            # we look for the DM content of a specific crop-country combination
            for y, year in enumerate(list_of_years):
                for i,time in enumerate(obs_data['TIME']):
                    if ((time == year)
                    and (obs_data['GEO'][i] == list_of_countries_EUR[k])
                    and (obs_data['CROP_PRO'][i] == crop_dict[crop][1])):
                        DM[crop][idc][y] = 1.-float(obs_data['Value'][i])/100. 

    # we pickle the generated dictionnary containing the dry matter content
    pickle_dump(DM, open(picklepathname, 'wb'))

#-------------------------------------------------------------------------------
    # We add a timestamp at end of the retrieval, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp
#-------------------------------------------------------------------------------

    return DM

#===============================================================================
# Function to plot a time series of crop DM content
def plot_DM_time_series(DM_dict, list_of_years_, crop, country):
#===============================================================================
    from matplotlib import pyplot as plt
    
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    fig.subplots_adjust(0.15,0.2,0.95,0.9,0.4,0.)
    ax.scatter(list_of_years_, DM_dict[crop][country], c='k')
    ax.set_ylabel('Dry matter fraction (-)', fontsize=14)
    ax.set_xlabel('time (year)', fontsize=14)
    ax.set_xlim([1955.,2015.])
    fig.savefig('DM_time_series_%s_%s.png'%(country,crop))
    #plt.show()

    return None

#===============================================================================
def preprocess_EUROSTAT_data():
#===============================================================================
    from maries_toolbox import open_csv_EUROSTAT, detrend_obs
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    from datetime import datetime
#-------------------------------------------------------------------------------
    # We add a timestamp at start of the retrieval
    start_timestamp = datetime.utcnow()
#-------------------------------------------------------------------------------
# Retrieve the observational dataset:
    fileyield    = 'agri_yields_NUTS1-2-3_1975-2014.csv'
    fileharvest  = 'agri_harvest_NUTS1-2-3_1975-2014.csv'
    filecroparea = 'agri_croparea_NUTS1-2-3_1975-2014.csv'
    filelanduse  = 'agri_landuse_NUTS1-2-3_2000-2013.csv'
    filelandusebis  = 'agri_missing_landuse.csv'

    NUTS_data   =  open_csv_EUROSTAT(EUROSTATdir,
            [fileyield, fileharvest, filecroparea, filelanduse, filelandusebis],
             convert_to_float=True, verbose=False)
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    try:
        #crops        = pickle_load(open('selected_crops.pickle','rb'))
        NUTS_regions = pickle_load(open('selected_NUTS_regions.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of regions to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
    # we print out some info for the user
    print '\nWe will loop over:\n'
    print sorted(crop_dict.keys())
    print sorted(NUTS_regions.keys())
#-------------------------------------------------------------------------------
    culti_dict = dict()
    arable_dict = dict()
    yield_dict = dict()
    harvest_dict = dict()
#-------------------------------------------------------------------------------
# LOOP OVER THE CROP NAME
#-------------------------------------------------------------------------------
    for crop in sorted(crop_dict.keys()):
		# NB: we use the first EUROSTAT name for all csv file, except the
		# prodhumid one!!
        EURO_name = crop_dict[crop][0] 
        print '\n%s'%crop

        culti_dict[crop]  = dict()
        arable_dict[crop] = dict()
        yield_dict[crop]  = dict()
        harvest_dict[crop] = dict()
#-------------------------------------------------------------------------------
# LOOP OVER THE REGION NAME
#-------------------------------------------------------------------------------
        for NUTS_no in sorted(NUTS_regions.keys()): 
            print '    ',NUTS_no

            # settings of the detrend_obs function:
            detr = False # do we detrend observations?
            verb = False # verbose
            fig  = False # production of a figure

			# retrieve the region's crop area and arable area for the years
			# 2000-2013. NB: by specifying obs_type='area', we do not remove a
			# long term trend in the observations 
            culti_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name,
                                      NUTS_data[filecroparea], 1., 2000, 
                                      obs_type='area', detrend=detr, 
                                      prod_fig=fig, verbose=verb)
            if ((NUTS_no == 'FI1B') or (NUTS_no == 'FI1C') or (NUTS_no == 'FI1D') or 
                (NUTS_no == 'NO01') or (NUTS_no == 'NO02') or (NUTS_no == 'NO03') or 
                (NUTS_no == 'NO04') or (NUTS_no == 'NO05') or (NUTS_no == 'NO06') or 
                (NUTS_no == 'NO07')):
                arable_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], 'ha: Arable land', 
                                      NUTS_data[filelandusebis], 1., 2000, 
                                      obs_type='area_bis', detrend=detr, 
                                      prod_fig=fig, verbose=verb)
            else:
                arable_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], 'Arable land', 
                                      NUTS_data[filelanduse], 1., 2000, 
                                      obs_type='area', detrend=detr,
                                      prod_fig=fig, verbose=verb)
            harvest_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name, 
                                      NUTS_data[fileharvest], 1., 2000, 
                                      obs_type='harvest', detrend=detr, 
                                      prod_fig=fig, verbose=verb)
            yield_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name, 
                                      NUTS_data[fileyield], 1., 2000, 
                                      obs_type='yield', detrend=detr,
                                      prod_fig=fig, verbose=verb)

    pickle_dump(culti_dict,  open(pickledir+'preprocessed_culti_areas.pickle','wb'))
    pickle_dump(arable_dict, open(pickledir+'preprocessed_arable_areas.pickle','wb'))
    pickle_dump(yield_dict,  open(pickledir+'preprocessed_yields.pickle','wb'))
    pickle_dump(harvest_dict,open(pickledir+'preprocessed_harvests.pickle','wb'))

    # We add a timestamp at end of the retrieval, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#===============================================================================
def all_crop_names():
#===============================================================================
    """
    This creates a dictionary of ALL crop names to read the EUROSTAT csv files:
    crops[crop_short_name] = ['EUROSTAT_name_1', 'EUROSTAT_name_2']

    EUROSTAT_name_1 can be used to retrieve information in the yield, harvest and
    area csv files. 
    EUROSTAT_name_2 should be used in the crop humidity content file only.

    """
    crops = dict()
    crops['Winter wheat']    = ['Common wheat and spelt','Common winter wheat']
    crops['Spring wheat']    = ['Common wheat and spelt','Common spring wheat']
    #crops['Durum wheat']    = ['Durum wheat','Durum wheat']
    crops['Grain maize']     = ['Grain maize','Grain maize and corn-cob-mix']
    crops['Fodder maize']    = ['Green maize','Green maize']
    crops['Spring barley']   = ['Barley','Barley']
    crops['Winter barley']   = ['Barley','Winter barley']
    crops['Rye']             = ['Rye','Rye']
    #crops['Rice']           = ['Rice','Rice']
    crops['Sugar beet']      = ['Sugar beet (excluding seed)','Sugar beet (excluding seed)']
    crops['Potato']          = ['Potatoes (including early potatoes and seed potatoes)',
                                'Potatoes (including early potatoes and seed potatoes)']
    crops['Field beans']     = ['Dried pulses and protein crops for the production '\
                              + 'of grain (including seed and mixtures of cereals '\
                              + 'and pulses)',
                                'Broad and field beans']
    crops['Spring rapeseed'] = ['Rape and turnip rape','Spring rape']
    crops['Winter rapeseed'] = ['Rape and turnip rape','Winter rape']
    crops['Sunflower']       = ['Sunflower seed','Sunflower seed']

    return crops

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
