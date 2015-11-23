#!/usr/bin/env python

import sys, os
import numpy as np

# This script scans through the EUROSTAT files to see the information coverage

#===============================================================================
def main():
#===============================================================================
    """
    This file does not  detrend observations, or convert humid matter to dry
    matter for yields and harvests. It just extracts observations into a 
    readable dictionary.

    """
    from cPickle import load as pickle_load
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, folderpickle, lands_levels
#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir   = '../observations/EUROSTAT_data/'
    folderpickle  = '../model_input_data/'

#-------------------------------------------------------------------------------
# Get the 2000-2013 EUROSTAT data per region and per crop:

    #data_type = ['yield', 'harvest', 'arableland', 'cultiland']
    data_type = ['cultiland']

    try:
        culti_areas  = pickle_load(open(folderpickle+'temp_cultivated_areas.pickle','rb'))
        arable_areas = pickle_load(open(folderpickle+'saved_temp_arable_areas.pickle','rb'))
        yields       = pickle_load(open(folderpickle+'saved_temp_yields.pickle','rb'))
        harvests     = pickle_load(open(folderpickle+'saved_temp_harvests.pickle','rb'))
        print '\nLoaded all pickle files!'
    except:
        print 'WARNING! You are launching the pre-processing of EUROSTAT data'
        print 'this procedure can take up to 6 hours!!!'
        preprocess_EUROSTAT_data()
        print '\nThe pre-processing is over, re-launch scan_observations_coverage.py'
        print 'to complete the analysis.'
        sys.exit()

#-------------------------------------------------------------------------------
# for each country code, we say which NUTS region level we want to 
# consider (for some countries: level 1, for some others: level 2)

    lands_levels = {'AT':1,'BE':1,'BG':2,'CH':1,'CY':2,'CZ':1,'DE':1,'DK':2,
                    'EE':2,'EL':2,'ES':2,'FI':2,'FR':2,'HR':2,'HU':2,'IE':2,
                    'IS':2,'IT':2,'LI':1,'LT':2,'LU':1,'LV':2,'ME':1,'MK':2,
                    'MT':1,'NL':1,'NO':2,'PL':2,'PT':2,'RO':2,'SE':2,'SI':2,
                    'SK':2,'TR':2,'UK':1}

# retrieve the regions of interest corresponding to these levels

    filename = folderpickle + 'EUROSTAT_regions_names.pickle'
    NUTS_regions = pickle_load(open(filename,'rb'))
    NUTS_regions_keys = sorted(NUTS_regions.keys())

#-------------------------------------------------------------------------------
# retrieve the list of crops

    crops = set_crop_standard_info()
    crop_name_keys = sorted(crops.keys())

#-------------------------------------------------------------------------------
# FOR OPTIMIZATION QUALITY CONTROL: PLOT MAPS OF DATA COVERAGE 

# We look at the spatial and temporal data coverage of a number of variables:

    for dtype in data_type: # data_type = yields, harvests, areas
        if dtype=='yield': results_dict = yields
        if dtype=='harvest': results_dict = harvests
        if dtype=='arableland': results_dict = arable_areas
        if dtype=='cultiland': results_dict = culti_areas

# for each crop

        for crop in crop_name_keys:
            print crop
            coverage = {}

# we compute the data coverage % over the years 2000-2013

            for NUTS_id in NUTS_regions_keys:
                print NUTS_id, NUTS_regions[NUTS_id]
                coverage[NUTS_id] = compute_european_data_coverage(results_dict, 
                                                                  crop, NUTS_id)
                print 'Coverage of %s in %s: %.2f'%(crop,NUTS_id,coverage[NUTS_id])

# we create a map of EUROSTAT data coverage

            fig = create_data_coverage_map(dtype,crop,coverage)

#-------------------------------------------------------------------------------
# TO MAKE A CHOICE ON THE NUMBER OF CROPS TO INCLUDE:
# crop cultivated area: % of total arable area: bar plots over Europe, evolution over year?
# crop harvest: % of total harvest in DM over Europe, evolutaion over years?



#===============================================================================
def create_data_coverage_map(dtype,crop,coverage):
#===============================================================================

    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import PathPatch

# Define the shapefile path and filename:

    shape_path = '../observations/EUROSTAT_data/EUROSTAT_website_2010_shapefiles/'
    shape_filename = 'NUTS_RG_03M_2010'# NUTS regions 

# create a basic map with coastlines

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,7))
    map = Basemap(projection='laea', lat_0=48, lon_0=16, llcrnrlat=30, 
                  llcrnrlon=-10, urcrnrlat=65, urcrnrlon=70)
    map.drawcoastlines()

# Read a shapefile and its metadata

    name = 'NUTS'
    # read the shapefile data WITHOUT plotting its shapes
    NUTS_info = map.readshapefile(shape_path + shape_filename, name, drawbounds=False) 
           
# Plot polygons and the corresponding pickled data:

    # retrieve the list of patches to fill and its data to plot
    patches    = []
    yield_data = []
    pickle_dict = list()
    # for each polygon of the shapefile
    for info, shape in zip(map.NUTS_info, map.NUTS):
        # we get the NUTS number of this polygon:
        NUTS_no = info['NUTS_ID']
        # if the NUTS level of the polygon corresponds to the desired one:
        if (info['STAT_LEVL_'] == lands_levels[NUTS_no[0:2]]):
            # we append the polygon to the patch collection
            patches.append( Polygon(np.array(shape), True) )
            # and we associate a yield to it
            try:
                yield_data.append( float(coverage[NUTS_no]) )
            except KeyError: #for CH0, LI0, HR04
                if (NUTS_no == 'HR04'):
                    HR01 = coverage['HR01']
                    HR02 = coverage['HR02']
                    yield_data.append(float(HR01)+float(HR02))
                else:
                    print "we don't have data for:", NUTS_no
                    yield_data.append(float(np.nan))
 
    # create a color scale that fits the data
    # NB: the data supplied needs to be normalized (between 0. and 1.)
    cmap = plt.get_cmap('RdYlGn')
    cmap.set_bad('w',1.)
    norm_data = np.array(yield_data)/max(yield_data)
    colors = cmap(norm_data)
 
    # add the polygons to the map
    collection = PatchCollection(patches, cmap = cmap, facecolors=colors, 
                                 edgecolor='k', linewidths=1., zorder=2) 
   		     
    # so that the colorbar works, we specify which data array we use to
    # construct it. Here the data should not be normalized.
    # NB: this overrides the colors specified in the collection line!
    data = np.ma.array(yield_data, mask=np.isnan(yield_data))
    collection.set_array(data) #data used for the colorbar
    collection.set_clim(0.,1.) #limits of the colorbar
    plt.colorbar(collection)
    ax.add_collection(collection)
    plt.title('%s %s\nFraction of years reported between 2000-2013'%(crop.upper(),
                                                                  dtype.upper()))
  
    fig.savefig('../figures/coverage_%s_%s.png'%(dtype,crop.lower())) 

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
             convert_to_float=True)
    #print NUTS_data[yields]  # this is a dictionary

#-------------------------------------------------------------------------------
# retrieve the regions of interest

    filename = folderpickle + 'EUROSTAT_regions_names.pickle'
    NUTS_regions = pickle_load(open(filename,'rb'))
    NUTS_regions_keys = sorted(NUTS_regions.keys())

#-------------------------------------------------------------------------------
    crops = set_crop_standard_info()
    crop_name_keys = sorted(crops.keys())

    culti_dict = dict()
    arable_dict = dict()
    yield_dict = dict()
    harvest_dict = dict()
#-------------------------------------------------------------------------------
# retrieve data per crop
    for crop in crop_name_keys:
        EURO_name = crops[crop][0] 
        print '\n%s'%EURO_name

        culti_dict[crop]  = dict()
        arable_dict[crop] = dict()
        yield_dict[crop]  = dict()
        harvest_dict[crop] = dict()
#-------------------------------------------------------------------------------
# retrieve data per region
        for NUTS_no in NUTS_regions_keys: 
#-------------------------------------------------------------------------------
# Retrieve the crop dry matter content
            DM = crops[crop][1]

            print '    ',NUTS_no
#-------------------------------------------------------------------------------
            # settings of the detrend_obs function:
            detr = False # do we detrend observations?
            verb = False # verbose
            fig  = False # production of a figure

            # retrieve the region's crop area and arable area for the years 2000-2013
            # NB: by specifying obs_type='area', we do not remove a long term trend in
            # the observations 
            culti_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name,
                                      NUTS_data[filecroparea], 1., 2000, obs_type='area',
                                      detrend=False, prod_fig=fig, verbose=verb)
            if ((NUTS_no == 'FI1B') or (NUTS_no == 'FI1C') or (NUTS_no == 'FI1D') or 
                (NUTS_no == 'NO01') or (NUTS_no == 'NO02') or (NUTS_no == 'NO03') or 
                (NUTS_no == 'NO04') or (NUTS_no == 'NO05') or (NUTS_no == 'NO06') or 
                (NUTS_no == 'NO07')):
                arable_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], 'ha: Arable land', 
                                      NUTS_data[filelandusebis], 1., 2000, obs_type='area_bis',
                                      detrend=detr, prod_fig=fig, verbose=verb)
            else:
                arable_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], 'Arable land', 
                                      NUTS_data[filelanduse], 1., 2000, obs_type='area',
                                      detrend=detr, prod_fig=fig, verbose=verb)
                
            # the following variables will be detrended:
            harvest_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name, 
                                      NUTS_data[fileharvest], DM, 2000, 
                                      obs_type='harvest',
                                      detrend=detr, prod_fig=fig, verbose=verb)
            yield_dict[crop][NUTS_no] = detrend_obs(2000, 2013, 
                                      NUTS_regions[NUTS_no][1], EURO_name, 
                                      NUTS_data[fileyield], DM, 2000, 
                                      obs_type='yield',
                                      detrend=detr, prod_fig=fig, verbose=verb)

    pickle_dump(culti_dict,open(folderpickle+'temp_cultivated_areas.pickle','wb'))
    #pickle_dump(arable_dict,open(folderpickle+'temp_arable_areas.pickle','wb'))
    #pickle_dump(yield_dict,open(folderpickle+'temp_yields.pickle','wb'))
    #pickle_dump(harvest_dict,open(folderpickle+'temp_harvests.pickle','wb'))

    # We add a timestamp at end of the retrieval, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the retrieval:', end_timestamp-start_timestamp

#===============================================================================
def compute_european_data_coverage(results_dict, crop, NUTS_id):
#===============================================================================
    counter = 0
    #if NUTS_id == 'CH0': return np.nan

    print results_dict[crop][NUTS_id][0]
    if len(results_dict[crop][NUTS_id][1])>0:#if the region was found in EUROSTAT
        if (sum(results_dict[crop][NUTS_id][0]) == -139986.0):# if the EUROSTAT records were empty
            coverage_frac = 0.
        else:
            coverage_frac = float(len(results_dict[crop][NUTS_id][0]))/\
                                float(len(range(2000,2014)))
    else:# if the region was absent from the EUROSTAT records
        coverage_frac = np.nan

    return coverage_frac

#===============================================================================
def set_crop_standard_info():
#===============================================================================
# dictionary of crop information to read the EUROSTAT csv files:
# crops[crop_short_name] = ['EUROSTAT_crop_name', DM_content_of_crop]
    crops = dict()
    crops['soft wheat']  = ['Common wheat and spelt',1.]
    crops['durum wheat'] = ['Durum wheat',1.]
    crops['rye']         = ['Rye',1.]
    crops['barley']      = ['Barley',1.]
    crops['grain maize'] = ['Grain maize',1.]
    crops['fodder maize']= ['Green maize',1.]
    crops['rice']        = ['Rice',1.]
    crops['peas and beans'] = ['Dried pulses and protein crops for the production '\
                            + 'of grain (including seed and mixtures of cereals '\
                            + 'and pulses)',1.]
    crops['potatoes']    = ['Potatoes (including early potatoes and seed potatoes)',1.]
    crops['sugar beet']  = ['Sugar beet (excluding seed)',1.]
    crops['rape']        = ['Rape and turnip rape',1.]
    crops['sunflower']   = ['Sunflower seed',1.]
    crops['linseed']     = ['Linseed (oil flax)',1.]

    return crops

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
