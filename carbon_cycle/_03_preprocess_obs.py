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
    located in the EUROSTATobs/ directory, and then re-launch the pre-
    processing.

    """
    from cPickle import load as pickle_load
    from maries_toolbox import get_crop_names
#-------------------------------------------------------------------------------
    global EUROSTATdir, crop_dict
#-------------------------------------------------------------------------------
# Define working directories
    EUROSTATdir   = '/Users/mariecombe/mnt/promise/CO2/wofost/EUROSTATobs/'
#-------------------------------------------------------------------------------
# NB: we will loop over ALL crops available, because we want to do the 
# pre-processing once and for all!
    crop_dict    = get_crop_names([], method='all')
#-------------------------------------------------------------------------------
# 1- PRE-PROCESS EUROSTAT CSV FILE ON DRY MATTER CONTENT:
#-------------------------------------------------------------------------------

    try:
        DM           = pickle_load(open(os.path.join(EUROSTATdir,
                                            'preprocessed_obs_DM.pickle'),'rb'))
        DM_standard  = pickle_load(open(os.path.join(EUROSTATdir,
                                       'preprocessed_standard_DM.pickle'),'rb'))
        print '\nThe processed crop humidity files already exist in:'
        print '    %s'%EUROSTATdir
        print 'We abort the pre-processing.'
    except IOError: # if the files don't exist, we create them:
        print '\nYou are launching the pre-processing of the DM content data.'
        dry_matter   = preprocess_EUROSTAT_DM()
        DM           = dry_matter[0]
        DM_standard  = dry_matter[1]
    except Exception as e:
        print 'Unexpected error:', e
        sys.exit()

#-------------------------------------------------------------------------------
# 1- PRE-PROCESS EUROSTAT CSV FILES ON AREAS, YIELDS, HARVESTS:
#-------------------------------------------------------------------------------

    try:
        culti_areas  = pickle_load(open(os.path.join(EUROSTATdir,
                                       'preprocessed_culti_areas.pickle'),'rb'))
        arable_areas = pickle_load(open(os.path.join(EUROSTATdir,
                                      'preprocessed_arable_areas.pickle'),'rb'))
        yields       = pickle_load(open(os.path.join(EUROSTATdir,
                                            'preprocessed_yields.pickle'),'rb'))
        harvests     = pickle_load(open(os.path.join(EUROSTATdir,
                                          'preprocessed_harvests.pickle'),'rb'))
        print '\nThe processed yield, harvest, area files already exist in:'
        print '    %s'%EUROSTATdir
        print 'We abort the pre-processing.'
    except IOError: # if the files don't exist, we create them:
        print '\nWARNING! You are launching the pre-processing of the yield, '+\
              'harvest, and area data \nthis procedure can take up to 20 hours!!!'
        preprocess_EUROSTAT_data()
    except Exception as e:
        print 'Unexpected error:', e
        sys.exit()

    print '\nThis script successfully finished.'

#===============================================================================
def preprocess_EUROSTAT_DM(plot=False):
#===============================================================================
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    from maries_toolbox import open_csv_EUROSTAT
#-------------------------------------------------------------------------------
# we retrieve the record of observed dry matter content

    picklepath      = os.path.join(EUROSTATdir,'preprocessed_obs_DM.pickle')

    if (os.path.exists(picklepath)):
        DM          = pickle_load(open(picklepath, 'rb'))
    else:
        NUTSdatadir = os.path.join(EUROSTATdir, 'download_2016')
        NUTSdata    = open_csv_EUROSTAT(NUTSdatadir,['NUTS12_humidity.csv'],
                                          convert_to_float=True, data_year=2016)
        DM          = compute_DM(NUTSdata['NUTS12_humidity.csv'])
        pickle_dump(DM, open(picklepath, 'wb'))

#-------------------------------------------------------------------------------
# We retrieve the standard EUROSTAT dry matter information in case there are
# gaps in the observations

    pathname = os.path.join(EUROSTATdir,'preprocessed_standard_DM.pickle')

    if (os.path.exists(pathname)):
        DM_standard = pickle_load(open(pathname, 'rb'))
    else:
        DM_standard = compute_standard_DM(pathname)
        pickle_dump(DM_standard, open(pathname, 'wb'))

#-------------------------------------------------------------------------------
# We can plot a time series of one crop x country DM combination if needed

    if (plot==True):
        list_of_years = np.linspace(1955,2015,61)
        time_series = plot_DM_time_series(DM, list_of_years, 3, 'ES')

    return DM, DM_standard

#===============================================================================
# Function to store EUROSTAT standard moisture contents in pickle file
def compute_standard_DM(picklepathname):
#===============================================================================

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
    DM_standard['Durum wheat']     = 1. - 0.14
    DM_standard['Triticale']       = 1. - 0.14
    DM_standard['Rapeseed and turnips'] = 1 - 0.09

    return DM_standard

#===============================================================================
# Function to retrieve EUROSTAT humidity content and to store it in pickle file
def compute_DM(obs_data):
#===============================================================================
    from datetime import datetime

    # EUROSTAT lists of crop names and country names
    list_of_years           = sorted(set(obs_data['Year']))
    list_of_country_ids     = sorted(set([c[0:2] for c in obs_data['NUTS_id']]))

    # CGMS list of crops was read from file at the beginning of the main() 
    # method
    list_of_crops_CGMS    = crop_dict.keys()

    # we print out some info for the user
    print '\nWe will loop over:\n'
    print list_of_crops_CGMS
    print list_of_country_ids
#-------------------------------------------------------------------------------
    # We add a timestamp at start of the retrieval
    start_timestamp = datetime.utcnow()
#-------------------------------------------------------------------------------

    DM = dict()
    # for each CGMS crop
    for crop in crop_dict.keys():
        DM[crop] = dict()
        print crop
        # for each NUTS0 region (countries)
        for NUTS_no in list_of_country_ids:
            DM[crop][NUTS_no] = np.zeros(len(list_of_years))
            print '    ', NUTS_no

            # we retrieve the DM content
            for y, year in enumerate(list_of_years):
                for i,time in enumerate(obs_data['Year']):
                    # if we match the year
                    if ((time == year)
                    # if we match the NUTS id
                    and (obs_data['NUTS_id'][i]   == NUTS_no)
                    # if we match the crop name
                    and (obs_data['Crop_name'][i] == crop_dict[crop][1])):
                        DM[crop][NUTS_no][y] = 1.-float(obs_data['Value'][i])/100. 

    # we store the list of years
    DM['years'] = np.array(list_of_years)

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
    from maries_toolbox import open_csv_EUROSTAT, detrend_obs,\
                               retrieve_crop_DM_content
    from cPickle import load as pickle_load
    from cPickle import dump as pickle_dump
    from datetime import datetime
#-------------------------------------------------------------------------------
    # We add a timestamp at start of the retrieval
    start_timestamp = datetime.utcnow()
#-------------------------------------------------------------------------------
# Retrieve the observational dataset:
    fileyield    = 'NUTS12_yield.csv'
    fileharvest  = 'NUTS12_harvest.csv'
    filearea     = 'NUTS12_area.csv'

    filepath = os.path.join(EUROSTATdir, 'download_2016')
    NUTS_data = open_csv_EUROSTAT(filepath,[fileyield, fileharvest, filearea],
                convert_to_float=True, verbose=False, data_year=2016)
#-------------------------------------------------------------------------------
    NUTS_regions = sorted(set(NUTS_data[fileharvest]['NUTS_id']))
#-------------------------------------------------------------------------------
    # we print out some info for the user
    print '\nWe will loop over:\n'
    print sorted(crop_dict.keys())
    print NUTS_regions
#-------------------------------------------------------------------------------
    cultia = dict()
    arable = dict()
    yields = dict()
    harvest = dict()
#-------------------------------------------------------------------------------
# LOOP OVER THE CROP NAME
#-------------------------------------------------------------------------------
    for crop in sorted(crop_dict.keys()):
		# NB: we use the first EUROSTAT name for all csv file, except the
		# prodhumid one!!
        EURO_name = crop_dict[crop][1] 
        print '\n====================================='
        print '%s'%crop
        print '====================================='

        cultia[crop]  = dict()
        arable[crop] = dict()
        yields[crop]  = dict()
        harvest[crop] = dict()
#-------------------------------------------------------------------------------
# LOOP OVER THE REGION NAME
#-------------------------------------------------------------------------------
        for NUTS_no in NUTS_regions: 
            print '    ',NUTS_no

            # We retrieve the average dry matter content of that crop over the
            # years in that country, or the standard content for that crop
            DM = retrieve_crop_DM_content(crop, NUTS_no, EUROSTATdir)

            # settings of the detrend_obs function:
            detr = False # do we detrend observations?
            verb = False # verbose
            fig  = False # production of a figure

			# retrieve the region's crop area and arable area for the years
			# 2000-2013. NB: by specifying obs_type='area', we do not remove a
			# long term trend in the observations 
            cultia[crop][NUTS_no] = detrend_obs(NUTS_no,
                                      EURO_name, NUTS_data[filearea], 1., 
                                      obs_type='culti_area', detrend=detr, 
                                      prod_fig=fig, verbose=verb)
            arable[crop][NUTS_no] = detrend_obs(NUTS_no, 
                                      'Arable land', NUTS_data[filearea], 1., 
                                      obs_type='arable_area', detrend=detr,
                                      prod_fig=fig, verbose=verb)
            harvest[crop][NUTS_no] = detrend_obs(NUTS_no, 
                                      EURO_name, NUTS_data[fileharvest], DM, 
                                      obs_type='harvest', detrend=detr, 
                                      prod_fig=fig, verbose=verb)
            dummy_yield = detrend_obs(NUTS_no, EURO_name, NUTS_data[fileyield], DM,
                                      obs_type='yield', detrend=detr,
                                      prod_fig=fig, verbose=verb)
            # we recalculate the yields from the harvest and area data if necessary
            yields[crop][NUTS_no] = compute_yield_from_harvest_over_area(dummy_yield, 
                                      harvest[crop][NUTS_no], cultia[crop][NUTS_no])
            print 'final yields:', yields[crop][NUTS_no][0], yields[crop][NUTS_no][1], '\n'
            print '--------------------------------------'

    pickle_dump(cultia,  open(os.path.join(EUROSTATdir,
                                       'preprocessed_culti_areas.pickle'),'wb'))
    pickle_dump(arable,  open(os.path.join(EUROSTATdir,
                                      'preprocessed_arable_areas.pickle'),'wb'))
    pickle_dump(yields,  open(os.path.join(EUROSTATdir,
                                            'preprocessed_yields.pickle'),'wb'))
    pickle_dump(harvest, open(os.path.join(EUROSTATdir,
                                          'preprocessed_harvests.pickle'),'wb'))

    # We add a timestamp at end of the retrieval, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp

    return cultia, arable, harvest, yields

#===============================================================================
def compute_yield_from_harvest_over_area(yields, harvests, cultias):
#===============================================================================
    # we compute the yields from harvest / area if we have less than 15 
    # years of yield reported and we have harvest and area data available
    if (len(yields[0])<15) and not (len(harvests[0])==0 and len(cultias[0])==0): 
        print "We try to recalculate yield as harvest / area"
        new_yields = []
        new_years  = []
        # loop over the years 2000-2014:
        for y,yearx in enumerate(np.arange(2000.,2015.,1.)):
            # if the yield already exists at year X, we store it
            if (yearx in yields[1]):
                yindx = np.argmin(np.abs(yields[1] - yearx))
                new_yields += [yields[0][yindx]]
                new_years  += [yearx]
            # if the yield doesn't exists at year X, we create it
            else:
                # 1- if and only if we have BOTH the harvest and culti area
                # we calculate it from them
                if (yearx in harvests[1] and yearx in cultias[1]):
                    hindx = np.argmin(np.abs(harvests[1] - yearx))
                    aindx = np.argmin(np.abs( cultias[1] - yearx))
                    # if the harvest or area = 0., we set yield = 0.
                    if (cultias[0][aindx]==0. or harvests[0][hindx]==0.):
                        calc = 0.
                    # and if the harvest or area are not = 0.
                    else: 
                        calc = harvests[0][hindx] / cultias[0][aindx]
                    new_yields += [calc*1000.] # kgDM ha-1
                    new_years  += [yearx]
                    print yearx, harvests[0][hindx], cultias[0][aindx], calc*1000. 
                # 2- if either one of the harvest or culti area is available
                elif (yearx in harvests[1] or yearx in cultias[1]):
                    hhindx = np.abs(harvests[1] - yearx)
                    aaindx = np.abs( cultias[1] - yearx)
                    if (0. in hhindx): # if the year exists in the harvest record
                        hindx = np.argmin(hhindx)
                        # if the harvest is 0. we set the yield to 0.
                        if (harvests[0][hindx] == 0.): 
                            calc=0.
                            new_yields += [calc*1000.] # kgDM ha-1
                            new_years  += [yearx]
                            print yearx, calc*1000. 
                    if (0. in aaindx): # if the year exists in the culti record
                        aindx = np.argmin(aaindx)
                        # if the area is 0. we set the yield to 0.
                        if (cultias[0][aindx] == 0.):
                            calc=0.
                            new_yields += [calc*1000.] # kgDM ha-1
                            new_years  += [yearx]
                            print yearx, calc*1000. 
        new_yields = np.array(new_yields)
        new_years  = np.array(new_years)

        return new_yields, new_years

    else:
        return yields

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
