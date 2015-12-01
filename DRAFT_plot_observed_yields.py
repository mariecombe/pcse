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

    # observed variable to plot
    dtype = 'cultiland'    # can be 'yield', 'harvest', or 'cultiland'

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
        crop_dict = pickle_load(open('selected_crops.pickle','rb'))
        #years     = pickle_load(open('selected_years.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of NUTS regions to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
    except Exception as e:
        print '\nUnexpected error:', e
        sys.exit()
#-------------------------------------------------------------------------------
# We download the dictionary of all crop names and crop standard dry matters
    #crop_dict = all_crop_names()
#-------------------------------------------------------------------------------
   
    # define which variable to plot on the map 
    if dtype=='yield': 
        results_dict = yields
        name = 'yield'
        units = 'kgDM ha-1'
    if dtype=='harvest': 
        results_dict = harvests
        name = 'harvest'
        units = '1000 tDM'
    if dtype=='cultiland': 
        results_dict = culti_areas
        name = 'cultivated area'
        units = '1000 ha'

    for c,crop in enumerate(sorted(crop_dict.keys())):
    #for c,crop in enumerate(custom_crop_order):
        print crop
        #if crop=='Potato': var_max = 55000.
        for year in [int(y) for y in years]:
            print year

            # initialize the yield dictionary of year X:
            reg_yield = dict()

            for NUTS_id in sorted(NUTS_regions.keys()):
                #print NUTS_id, NUTS_regions[NUTS_id]
                # retrieve the regional data at year X
                reg_yield[NUTS_id] = retrieve_regional_data_at_year_x(
                                               results_dict, year, crop, NUTS_id)
                #print '%s in %s: %.2f'%(custom_crop_names[c],
                #                     NUTS_id, reg_yield[NUTS_id])
            # plot a European map of data coverage
            fig = create_observed_data_map(dtype, crop,
                                                    reg_yield, name,units,year)
        sys.exit() 

#===============================================================================
def retrieve_regional_data_at_year_x(results_dict,year,crop,NUTS_no):
#===============================================================================

    # if we have any data for that year in the records:
    if int(year) in [int(y) for y in results_dict[crop][NUTS_no][1]]:
        indx_year = results_dict[crop][NUTS_no][1] - int(year)
        indx_year = (np.abs(indx_year)).argmin()
        data_year = results_dict[crop][NUTS_no][0][indx_year]
    else:
        data_year = np.nan
#    print NUTS_no, year, data_year
#    print results_dict[crop][NUTS_no]
#    print ''

    return data_year

#===============================================================================
def create_observed_data_map(dtype,crop,results,name_var,units,year,var_max=1.):
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

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(9,7))
    fig.subplots_adjust(0.05,0.05,0.85,0.88,0.4,0.6)
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
                yield_data.append( float(results[NUTS_no]) )
            except KeyError: #for CH0, LI0, HR04
                if (NUTS_no == 'HR04'):
                    HR01 = results['HR01']
                    HR02 = results['HR02']
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
   	
    var_max = max(yield_data)
	     
    # so that the colorbar works, we specify which data array we use to
    # construct it. Here the data should not be normalized.
    # NB: this overrides the colors specified in the collection line!
    data = np.ma.array(yield_data, mask=np.isnan(yield_data))
    collection.set_array(data) #data used for the colorbar
    collection.set_clim(0.,var_max) #limits of the colorbar
    cb = plt.colorbar(collection, fraction=0.041, pad=0.04)
    cb.set_label(label = '\nobserved %s (%s)'%(name_var,units), size=16, 
                 style='italic')
    cb.ax.tick_params(labelsize=16)
    ax.add_collection(collection)
    plt.title('%s - %s'%(crop.upper(),year),
              fontsize=20)
    fig.savefig('../figures/observed_%s_%s_%s.png'%(dtype,year,crop.lower())) 
    
    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
