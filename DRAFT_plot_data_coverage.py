#!/usr/bin/env python

import sys, os
import numpy as np

# This script scans through the EUROSTAT files to see the information coverage

#===============================================================================
def main():
#===============================================================================
    """
    PBs of this file: 
    (1) Does not use observed DM data, only self-defined standard DM contents

    """
    from cPickle import load as pickle_load
    from maries_toolbox import all_crop_names
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, pickledir,custom_crop_order,\
           custom_crop_names, NUTS_regions
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # select which part of the script to execute:
    plot_crop_share             = True
    plot_spatial_data_coverage  = False
    plot_temporal_data_coverage = False

    # We choose to loop over ALL years (2000-2013) to scan the data coverage
    years = np.linspace(2000,2013,14)

    # We select to loop only over the following crops, because the EUROSTAT
    # observations do not separate between spring and winter crops (at least for
    # the yields, harvests and areas datasets)
    custom_crop_order = ['Winter wheat','Spring barley',
                         'Grain maize','Fodder maize',
                         'Sugar beet','Potato','Rye','Spring rapeseed',
                         'Sunflower','Field beans']
    custom_crop_names = ['Common wheat','Barley',
                         'Grain maize','Fodder maize',
                         'Sugar beet','Potato','Rye','Rapeseed',
                         'Sunflower','Field beans']

# ==============================================================================
#-------------------------------------------------------------------------------
# Define working directories
    currentdir    = os.getcwd()
    EUROSTATdir   = '../observations/EUROSTAT_data/'
    pickledir     = '../model_input_data/'
#-------------------------------------------------------------------------------
# Get the pre-processed EUROSTAT data per region and per crop:
    try:
        culti_areas  = pickle_load(open(pickledir+'preprocessed_culti_areas.pickle','rb'))
        arable_areas = pickle_load(open(pickledir+'preprocessed_arable_areas.pickle','rb'))
        yields       = pickle_load(open(pickledir+'preprocessed_yields.pickle','rb'))
        harvests     = pickle_load(open(pickledir+'preprocessed_harvests.pickle','rb'))
        dry_matter   = pickle_load(open(pickledir+'preprocessed_obs_DM.pickle','rb'))
        dry_matter_st= pickle_load(open(pickledir+'preprocessed_standard_DM.pickle','rb'))
        print '\nLoaded all pickle files!'
    except IOError:
        print '\nFiles do not exist! Run 03_preprocess_obs.py before you try again.'
        sys.exit()
    except Exception as e:
        print '\nUnexpected error:', e
        sys.exit()
#-------------------------------------------------------------------------------
# we retrieve the NUTS regions selected by the user (for the maps)
    try:
        NUTS_regions = pickle_load(open('selected_NUTS_regions.pickle','rb')) 
    except IOError:
        print '\nYou have not yet selected a shortlist of NUTS regions to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
    except Exception as e:
        print '\nUnexpected error:', e
        sys.exit()
#-------------------------------------------------------------------------------
# We download the dictionary of all crop names and crop standard dry matters
    crop_dict = all_crop_names()
#-------------------------------------------------------------------------------
# 1. CROP SPECIES CHOICE VALIDATION
    if (plot_crop_share):
#-------------------------------------------------------------------------------
# 1.1. CROP FRACTIONS OF TOTAL HARVEST OVER THE YEARS
# 1.2. CROP AREA FRACTIONS OF TOTAL CULTIVATED AREA OVER THE YEARS
#-------------------------------------------------------------------------------

        # compute the harvest fraction of all crops over the years
        harvest_absol = compute_data_frac(harvests, crop_dict, years, NUTS_regions, 
                                                          'abs', convert_to_dm=True)
        harvest_fracs = compute_data_frac(harvests, crop_dict, years, NUTS_regions, 
                                                         'frac', convert_to_dm=True)
 
        # compute the cultivated area fraction of all crops over the years
        culti_absol   = compute_data_frac(culti_areas, crop_dict, years, NUTS_regions,
                                                                              'abs')
        culti_fracs   = compute_data_frac(culti_areas, crop_dict, years, NUTS_regions,
                                                                             'frac')
 
        # 1.1. plot a bar plot of crop harvest: fractions or absolute share
        fig = create_stacked_bar_plot(harvest_fracs,years,'harvest','fraction','-')
        fig = create_stacked_bar_plot(harvest_absol,years,'harvest','absolute part',
                                                                    '1000 t$_{DM}$')
 
        # 1.2. plot a bar plot of crop cultivated fractions or absolute share
        fig = create_stacked_bar_plot(culti_fracs, years, 'cultivated area',
                                                                    'fraction', '-')
        fig = create_stacked_bar_plot(culti_absol, years, 'cultivated area',
                                                         'absolute part', '1000 ha')
 
        print '\nStacked bar plots are saved in folder ../figures/'

#-------------------------------------------------------------------------------
# 2. OPTIMIZATION ROUTINE QUALITY CONTROL 
#-------------------------------------------------------------------------------
# 2.1. ARE THE SELECTED NUTS LEVEL OK? CHECK SPATIAL DATA COVERAGE 
    if (plot_spatial_data_coverage):
#-------------------------------------------------------------------------------
        # ARABLE LAND: it is not dependent on the crop species
        # we supply a boggus crop name to the functions
        spatial_coverage = {}
        # Loop over regions
        for NUTS_id in sorted(NUTS_regions.keys()):
            #print NUTS_id, NUTS_regions[NUTS_id]
 
            # compute the regional data coverage % over the years 2000-2013
            spatial_coverage[NUTS_id] = compute_spatial_data_coverage(arable_areas, 
                                                              'Rye', NUTS_id)
            print 'Coverage of arable land in %s: %.2f'%(NUTS_id,
                                                          spatial_coverage[NUTS_id])
 
        # plot a European map of data coverage
        fig = create_data_coverage_map('arableland', 'Barley', spatial_coverage)
 
#-------------------------------------------------------------------------------
    # YIELD, HARVEST, CULTI AREA: it is dependent on the crop
    # We look at the spatial and temporal data coverage of a number of variables:
    data_type = ['yield', 'harvest', 'cultiland']
 
    for dtype in data_type: # data_type = yields, harvests, areas
        if dtype=='yield': 
            results_dict = yields
            name = 'yield'
        if dtype=='harvest': 
            results_dict = harvests
            name = 'harvest'
        if dtype=='cultiland': 
            results_dict = culti_areas
            name = 'cultivated area'

        # Loop over crops
        #for crop in sorted(crop_dict.keys()):
        for c,crop in enumerate(custom_crop_order):
            print crop

#-------------------------------------------------------------------------------
            if (plot_spatial_data_coverage):
#-------------------------------------------------------------------------------
                spatial_coverage = dict()
                # Loop over regions
                for NUTS_id in sorted(NUTS_regions.keys()):
                    #print NUTS_id, NUTS_regions[NUTS_id]
 
                    # compute the regional data coverage % over the years
                    spatial_coverage[NUTS_id] = compute_spatial_data_coverage(
                                                   results_dict, crop, NUTS_id)
                    print 'Coverage of %s in %s: %.2f'%(custom_crop_names[c],
                                             NUTS_id, spatial_coverage[NUTS_id])
 
                # plot a European map of data coverage
                fig = create_data_coverage_map(dtype, custom_crop_names[c],
                                                         spatial_coverage, name)
 
#-------------------------------------------------------------------------------
# 2.2. ARE THE SELECTED YEARS OK? CHECK TEMPORAL DATA COVERAGE 
            if (plot_temporal_data_coverage):
#-------------------------------------------------------------------------------
                temporal_coverage = list()
                # loop over years:
                for year in years:
                    # compute the yearly data coverage % over the years
                    temporal_coverage += [compute_temporal_data_coverage(
                                                      results_dict, year, crop)]
                # plot a bar plot of data coverage
                create_bar_plot(temporal_coverage, years, dtype, crop, name)

#-------------------------------------------------------------------------------
    if (plot_temporal_data_coverage):
#-------------------------------------------------------------------------------
        # ARABLE LAND: it is not dependent on the crop species
        # we supply a boggus crop name to the functions
        name = 'arable land'
        temporal_coverage = list()

        # loop over years:
        for year in years:
            # compute the yearly data coverage % over the years
            temporal_coverage += [compute_temporal_data_coverage(
                                              arable_areas, year,'Rye')]
        # plot a bar plot of data coverage
        create_bar_plot(temporal_coverage, years, 'arableland', 'Rye', name)

#===============================================================================
def compute_temporal_data_coverage(results_dict,year,crop):
#===============================================================================
    counter = 0

    # Loop over regions
    for NUTS_id in sorted(NUTS_regions.keys()):
        #if the region was found in EUROSTAT:
        if len(results_dict[crop][NUTS_id][1])>0:
            # if we find the year in the record:
            if int(year) in [int(y) for y in results_dict[crop][NUTS_id][1]]:
                 counter += 1
        # if the region was absent from the EUROSTAT records
        else:
            counter += 0
    coverage_frac = float(counter) / float(len(NUTS_regions))

    return coverage_frac

#===============================================================================
def compute_spatial_data_coverage(results_dict, crop, NUTS_id):
#===============================================================================
    counter = 0
    #if NUTS_id == 'CH0': return np.nan

    #if the region was found in EUROSTAT:
    if len(results_dict[crop][NUTS_id][1])>0:
        # if the EUROSTAT records were empty:
        if (sum(results_dict[crop][NUTS_id][0]) == -139986.0):
            coverage_frac = 0.
        else:
            coverage_frac = float(len(results_dict[crop][NUTS_id][0]))/\
                                float(len(range(2000,2014)))
    else:# if the region was absent from the EUROSTAT records
        coverage_frac = np.nan

    return coverage_frac

#===============================================================================
def compute_data_frac(results_dict, crops_dict, years, NUTS_regions, proc, 
                                                           convert_to_dm=False):
#===============================================================================
    # we create a matrix (2D array) with dimensions N years, X crops:
    N = len(years)
    X = len(results_dict.keys())
    crop_frac = np.zeros((X,N))

    #for c, crop in enumerate(sorted(results_dict.keys())):
    for c, crop in enumerate(custom_crop_order):
        #print c, crop
        for y, year in enumerate(years):
            # total harvest or area of that crop:
            total_crop = 0.
            # we add up the harvest or area over Europe:
            for NUTS_id in NUTS_regions:
                if year in results_dict[crop][NUTS_id][1]:
                    indx_year = results_dict[crop][NUTS_id][1]-year
                    indx_year = indx_year.argmin()
                    HM        = results_dict[crop][NUTS_id][0][indx_year]
                    if (convert_to_dm==True):
                        DM    = crops_dict[crop][0]
                        total_crop += HM*DM 
                    else:
                        total_crop += HM
                else:
                    total_crop += 0.
            crop_frac[c,y] = total_crop

    # if we want to compute the fractions of the total:
    if (proc=='frac'):
        for y,year in enumerate(years):
            total_year = 0.
            # we compute the total harvest or area at year y
            #for c,crop in enumerate(sorted(results_dict.keys())):
            for c, crop in enumerate(custom_crop_order):
                total_year += crop_frac[c,y]
            # now that the total is known we compute fractions of the total
            #for c,crop in enumerate(sorted(results_dict.keys())):
            for c, crop in enumerate(custom_crop_order):
                crop_frac[c,y] /= total_year
            assert sum(crop_frac[:,y]), 'The sum of fractions is not 1!!'

    #return sorted(results_dict.keys()), crop_frac
    return custom_crop_order, crop_frac

#===============================================================================
def create_data_coverage_map(dtype,crop,coverage):
#===============================================================================

    from matplotlib import pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import PathPatch

    plt.close('all')
# Define the shapefile path and filename:

    shape_path = '../observations/EUROSTAT_data/EUROSTAT_website_2010_shapefiles/'
    shape_filename = 'NUTS_RG_03M_2010'# NUTS regions 

    # for each country code, we say which NUTS region level we want to 
    # consider (for some countries: level 1, for some others: level 2)
    lands_levels = {'AT':1,'BE':1,'BG':2,'CH':1,'CY':2,'CZ':1,'DE':1,'DK':2,
                    'EE':2,'EL':2,'ES':2,'FI':2,'FR':2,'HR':2,'HU':2,'IE':2,
                    'IS':2,'IT':2,'LI':1,'LT':2,'LU':1,'LV':2,'ME':1,'MK':2,
                    'MT':1,'NL':1,'NO':2,'PL':2,'PT':2,'RO':2,'SE':2,'SI':2,
                    'SK':2,'TR':2,'UK':1}
# create a basic map with coastlines

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,7))
    map = Basemap(projection='laea', lat_0=48, lon_0=16, llcrnrlat=30, 
                  llcrnrlon=-10, urcrnrlat=65, urcrnrlon=70)
    map.drawcoastlines()

# Read a shapefile and its metadata

    name = 'NUTS'
    # read the shapefile data WITHOUT plotting its shapes
    NUTS_info = map.readshapefile(shape_path + shape_filename, name, 
                                                              drawbounds=False) 
           
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
    if (dtype=='arableland'):
        plt.title('%s\nFraction of years reported between 2000-2013'%(
                                                              name_var.upper()))
        fig.savefig('../figures/spatial_coverage_%s.png'%(dtype)) 
    else:
        plt.title('%s %s\nFraction of years reported between 2000-2013'%(
                                                crop.upper(), name_var.upper()))
        fig.savefig('../figures/spatial_coverage_%s_%s.png'%(dtype, 
                                                                  crop.lower())) 
    return None

#===============================================================================
def create_bar_plot(data_list,years_list,dtype,crop,name_var):
#===============================================================================
    from matplotlib import pyplot as plt

    plt.close('all')
    width = 1.
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,7))
    fig.subplots_adjust(0.1,0.1,0.95,0.88,0.4,0.6)

    plt.bar(np.arange(len(years_list)), data_list, width, color='grey')
    plt.xticks(np.arange(len(years_list))+width/2., [int(y) for y in years_list], 
               rotation=45, fontsize=14)
    plt.ylim(0.,1.)
    plt.xlim(0.,14.)
    if (dtype=='arableland'):
        plt.title('Fraction of NUTS regions reporting %s\n'%(name_var), 
                   fontsize=18)
        fig.savefig('../figures/temporal_coverage_%s.png'%dtype)
    else:
        plt.title('%s\nFraction of NUTS regions reporting %s\n'%(crop,name_var), 
                   fontsize=18)
        fig.savefig('../figures/temporal_coverage_%s_%s.png'%(dtype,crop.lower()))

#===============================================================================
def create_stacked_bar_plot(data_frac,years,dtype,proc,unit,name_var):
#===============================================================================
    from matplotlib import pyplot as plt

    plt.close('all')
    co = ['gold','b','aqua','k','r','forestgreen','magenta',
          'saddlebrown','springgreen','orange','firebrick','pink','grey']
    co2 = ['gold','b','r','forestgreen','magenta',
          'saddlebrown','springgreen','firebrick','aqua','grey']
    width = 0.5
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10,7))
    fig.subplots_adjust(0.1,0.1,0.75,0.88,0.4,0.6)
    C = len(data_frac[0])

    bot  = np.zeros(len(years))
    bars = [0.]*len(data_frac[0])
    for c,crop in enumerate(data_frac[0]):
        bars[c] = plt.bar(np.arange(len(years)), data_frac[1][c,:], width,
                          color = co2[c], bottom=bot)
        bot += np.array(data_frac[1][c,:])
    if (proc=='fraction'):
        plt.ylim(0.,1.)
        plt.plot(np.arange(-1.,len(years)+1),[0.9]*(len(years)+2), lw=2,
                 ls='--',c='k')
    plt.xticks(np.arange(len(years))+width/2., [int(y) for y in years], 
               rotation=45)
    plt.xlim(-0.5,14.)
    plt.legend([p[0] for p in bars[::-1]], custom_crop_names[::-1], ncol=1,
               bbox_to_anchor = (1.39,0.8), fontsize=14)
    plt.title('%s of the total %s (%s)\n'%(proc,dtype,unit), fontsize=18)
    fig.savefig('../figures/%s_total_%s.png'%(proc[0:4],dtype[0:5]))

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
