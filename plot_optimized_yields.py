#!/usr/bin/env python

import os, sys
import numpy as np

from maries_toolbox import get_crop_name, open_csv, open_csv_EUROSTAT,\
                           detrend_obs, define_opti_years,\
                           retrieve_crop_DM_content, get_EUR_frac_crop,\
                           open_pcse_csv_output
from matplotlib import pyplot as plt

#===============================================================================
# This script plots optimum yields gap factors obtained with the sensitivity
# analysis
def main():
#===============================================================================
    global currentdir, EUROSTATdir, RMSEdir, ForwardSimdir
#-------------------------------------------------------------------------------
# the user selects the figures to plot:

    # figures for the sensitivity analysis on the n combi:
    figure_1 = False # figure showing RMSE = f(YLDGAPF) for various top n combi 
    figure_2 = False # figure showing yldgapf = f(various n cell x soil combi)

    # figures for the sensitivity analysis to the value of the yldgapf
    figure_3 = False # figure showing the relative deviation of the simulated 
                    # yield from the observed yield

    # figures for the forward simulations:
    figure_4 = False # figure showing a time series of opt vs. obs yields
    figure_5 = True # figure showing a map of yield over a region
    figure_6 = False # comparison with fluxnet data
    figure_7 = False

#-------------------------------------------------------------------------------
# Define general working directories
    currentdir = os.getcwd()
	# directories on my local MacBook:
    EUROSTATdir   = '../Cbalance/EUROSTAT_data'
    FLUXNETdir    = '../Cbalance/FluxNet_data'
    # directories on capegrim:
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"
    pcseoutputdir = '/Users/mariecombe/mnt/promise/CO2/marie/pcse_output'
#-------------------------------------------------------------------------------
# Close all figures
    plt.close('all')
#-------------------------------------------------------------------------------
# Plot the figures for the sensitivity analysis on the number of grid cell 
# and soil types combinations

    # user defined:
    crops     = [3]
    regions   = ['ES41','ES43'] # the regions for which we have performed
                               # the sensitivity analysis
    cases     = ['all_2gx2s_yield_2006-2006', 'topn_10gx10s_yield_2006-2006',
                 'topn_5gx5s_yield_2006-2006', 'topn_3gx3s_yield_2006-2006']
                            # these are search terms to
                            # find the results files of the sensitivity analysis
    labels    = ['all', 'top 10', 'top 5', 'top 3'] 
    colors    = ['k','r','b','g']#,'cyan','orange','yellow']
    RMSEdir   = os.path.join(currentdir, 'pcse_summary_output') # directory where the
                            # sensitivity analysis results files are located

    # open the results from the sensitivity analysis
    #RMSE      = retrieve_RMSE_and_YLDGAPF(regions, cases)

    # calculate some other variables from the user input:
    crop_name = get_crop_name(crops)
    NUTS_name = dict() # this should not be hardcoded
    NUTS_name['ES41'] = 'Castilla y Leon'
    NUTS_name['ES42'] = 'Castilla-la Mancha'
    NUTS_name['ES43'] = 'Extremadura'

    for crop in crops:
    
        # Plot the RMSE = f(YLDGAPF) graph for each region x crop combination
        if (figure_1 == True):
            print '\n============================== tackling figure 1 =============================='
            figs1 = plot_RMSE_asf_YLDGAPF(crop_name[crop][0], regions, cases, 
                                          labels, colors, RMSE[0])

        # Plot the optimum YLDGAPF = f(case) for each region x crop combination
        if (figure_2 == True):
            print '\n============================== tackling figure 2 =============================='
            figs2 = plot_optiYLDGAPF_asf_cases(crop_name[crop][0], regions, 
                                               cases, labels, colors, RMSE[1])

#-------------------------------------------------------------------------------
# Plot the figure showing the relative deviation of the simulated yield from the
# observed yield

    # user defined:
    crops     = [3]
    regions   = ['ES41'] # the regions for which we have performed
                               # the sensitivity analysis
    start_year  = 2000     # start_year and end_year define the period of time
    end_year    = 2014     # for which we did our forward simulations
    opti_year   = 2006       # nb of years for which we optimized the YLDGAPF
    ForwardSimdir = os.path.join (currentdir, 'pcse_summary_output') # directory where
                             # the results files of the forward simulations are
                             # located

    if (figure_3 == True):
        print '\n============================== tackling figure 3 =============================='
        # retrieve the observed and simulated yields and aggregate into regional 
        # values
        Regional_yield = compute_regional_yields_series(crops, regions, start_year,
                                                   end_year, 'yield', ForwardSimdir)
 
        #rel_dif_top3 = (Regional_yield[1][crops[0]][regions[0]]['opt-0.05'] -
        #                 Regional_yield[1][crops[0]][regions[0]]['opt']) / \
        #                   Regional_yield[1][crops[0]][regions[0]]['opt']
        #print rel_dif_top3 
 
        # Plot the time series of opt vs. non-optimized yields, and observed yields
        # for each crop x region
        fig3 = plot_precision_yldgapf(crops, regions, start_year, 
                                 end_year, opti_year, Regional_yield, 'yield')

#-------------------------------------------------------------------------------
# Plot the figure showing a time series of opt vs. obs yields

    # list the regions and crops for which we do that figure
    crops      = [3]      # crop ID numbers
    regions    = ['ES41', 'ES42', 'ES43']
                          # regions ID numbers for which we have performed
                          # the sensitivity analysis
    val_type   = 'harvest'  # regional variable we want to plot
                          # can be 'yield', 'harvest', 'area' (cultivated)
    start_year = 2000     # start_year and end_year define the period of time
    end_year   = 2014     # for which we did our forward simulations
    opti_year  = 2006     # nb of years for which we optimized the YLDGAPF
    ForwardSimdir = os.path.join (currentdir, 'output_data') # directory where
                          # the results files of the forward simulations are
                          # located

    if (figure_4 == True):
        print '\n========================== tackling figure 4 =========================='
        # retrieve the observed and simulated yields and aggregate into regional 
        # values
        Regional_values = compute_regional_yield(crops, regions, start_year,
                                              end_year, val_type, ForwardSimdir)

        # Plot the time series of opt vs. non-optimized yields, and observed 
        # yields for each crop x region
        fig4 = plot_yield_time_series(crops, regions, start_year, 
                                 end_year, opti_year, Regional_values, val_type)

#-------------------------------------------------------------------------------
# Plot the figure showing a map of yield over a region

    if (figure_5 == True):
        print '\n========================== tackling figure 5 =========================='
        from mpl_toolkits.basemap import Basemap
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import PathPatch
        from osgeo import ogr # to add data to a shapefile

# Define the shapefile path and filename:

        path = '../Cbalance/EUROSTAT_data/EUROSTAT_website_2010_shapefiles/'
        filename = 'NUTS_RG_03M_2010'# NUTS regions 
        #filename = 'NUTS_RG_03M_2010_copy'# NUTS regions 

# define the NUTS regions of interest:

        lands_levels = {'AT':1,'BE':1,'BG':2,'CH':1,'CY':2,'CZ':1,'DE':1,'DK':2,
                        'EE':2,'EL':2,'ES':2,'FI':2,'FR':2,'HR':2,'HU':2,'IE':2,
                        'IS':2,'IT':2,'LI':1,'LT':2,'LU':1,'LV':2,'ME':1,'MK':2,
                        'MT':1,'NL':1,'NO':2,'PL':2,'PT':2,'RO':2,'SE':2,'SI':2,
                        'SK':2,'TR':2,'UK':1}

        from cPickle import load as pickle_load
        list_of_regions = pickle_load(open('list_of_NUTS_regions.pickle','rb'))

        from random import random
        pickled_yield_data = {}
        for reg in list_of_regions:
            pickled_yield_data[reg] = random()*1000.

# first add an attribute to the shapefile (make a function for that)
# this attribute will be the yield data! observed, simulated, anomaly, etc...

#        # open a shapefile and get field names
#        source = ogr.Open(os.path.join(path + filename), 1) # 1 is read/write
#        layer = source.GetLayer()
#        layer_defn = layer.GetLayerDefn()
#        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in 
#                       range(layer_defn.GetFieldCount())]
#        print 'nb of fields:', len(field_names), 'yield' in field_names
#        # add a new field
#        new_field = ogr.FieldDefn('yield', ogr.OFTReal) # could be OFTInteger or 
#                                                        # OFTString, depending on
#                                                        # desired data type
#        layer.CreateField(new_field)
#        field_names = [layer_defn.GetFieldDefn(i).GetName() for i in
#                       range(layer_defn.GetFieldCount())]
#        print 'nb of fields:', len(field_names), 'yield' in field_names
#        # close the shapefile
#        source = None
#        sys.exit(2)

# create a map of Europe with coast lines

        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,7))
        map = Basemap(projection='laea', lat_0=48, lon_0=16, llcrnrlat=30, 
                      llcrnrlon=-10, urcrnrlat=65, urcrnrlon=45)
        map.drawcoastlines()

# plot specific scatter points (e.g. FluxNet data sites)

        lons = [-5, 0]
        lats = [40, 48]
        x, y = map(lons,lats)
        #map.scatter(x, y, marker='D', color='m')

# Read a shapefile and its metadata

        name = 'NUTS'
        # read the shapefile data WITHOUT plotting its shapes
        NUTS_info = map.readshapefile(path + filename, name, drawbounds=False) 

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
                yield_data.append( float(pickled_yield_data[NUTS_no]) )
                       
        ## following lines produce the pickled list of NUTS regions
        #        if NUTS_no not in pickle_dict:
        #            pickle_dict += [NUTS_no]
        #from cPickle import dump as pickle_dump
        #pickle_dump(pickle_dict, open('list_of_NUTS_regions.pickle','wb')) 

        # create a color scale that fits the data
        # NB: the data supplied needs to be normalized (between 0. and 1.)
        cmap = plt.get_cmap('RdYlBu')
        colors = cmap(np.array(yield_data)/1000.)

        # add the patches on the map
        collection = PatchCollection(patches, cmap = cmap, facecolors=colors, 
                                     edgecolor='k', linewidths=1., zorder=2) 
        # so that the colorbar works, we specify which data array we use to
        # construct it. Here the data should not be normalized.
        # NB: this overrides the colors specified in the collection line
        collection.set_array(np.array(yield_data))
        collection.set_clim(min(yield_data),max(yield_data))
        plt.colorbar(collection)
        ax.add_collection(collection)
       
        plt.show()
#-------------------------------------------------------------------------------
# Plot the FluxNet GPP, Reco from various sites

    if (figure_6 == True):
        print '\n========================== tackling figure 6 =========================='
        sites = ['DE-Kli', 'ES-ES2']
        years = [2006]
        for site in sites:
            listoffiles = [f for f in os.listdir(FLUXNETdir) if (site in f)]
            filename = listoffiles[0]
            FNdata = open_csv(FLUXNETdir, listoffiles)
            fig = plt.subplots(ncols=1, nrows=1, figsize=(14,7))
            for var in ["GPP_f","Reco","NEE_f"]:
                plt.plot(FNdata[filename]['DoY'],FNdata[filename][var])
            
            plt.xlabel('Time (DOY)')
            plt.ylabel('Carbon fluxes (gC m-2 d-1)')
        #fig.suptitle(site)
        plt.show()

#-------------------------------------------------------------------------------
# Plot the FluxNet GPP, Reco from various sites

    if (figure_7 == True):
        from operator import itemgetter as operator_itemgetter
        print '\n========================== tackling figure 7 =========================='
        # open file with grid cell coordinates:
        CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
        all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
        lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
        lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']

        # construct a list of all available results (runs did not finish entirely)
        listoffiles = [f for f in os.listdir(pcseoutputdir) if ('.csv' in f) and ('timeseries' in f)]
        listoffiles = sorted(listoffiles)
        list_of_gridcells = list()
        list_of_results   = list()
        for filename in listoffiles:
            args    = filename.split('_')  
            grid_no = float(args[4].replace('g',''))
            #stu_no  = args[5].replace('.csv','').replace('s','')
            # construct a 2-D array with the results
            i   = np.argmin(np.absolute(all_grids - grid_no))
            lon = lons[i]
            lat = lats[i]
            res = open_pcse_csv_output(pcseoutputdir, [filename])
            TSO = res[1]
            #  am going to quickly exclude repeated soils
            if grid_no not in list_of_gridcells:
                list_of_gridcells += [grid_no]
                list_of_results   += [(grid_no, lon, lat, TSO)]
        #list_of_results = sorted(list_of_results, key=operator_itemgetter(0))
        from cPickle import dump as pickle_dump
        pickle_dump(list_of_results,open('list_of_grid_lon_lat_yield.pickle','wb'))
        print 'pickled results!'
        sys.exit(2)
        # set up the grid and create a 2D array with same dimension containing the results
        lons = [lon for g,lon,lat,y in list_of_results]
        lats = [lat for g,lon,lat,y in list_of_results]
        #lons = np.array(sorted(list(set(lons))))
        #lats = np.array(sorted(list(set(lats))))
        #nx   = len(lons)
        #ny   = len(lats)
        #crop_yield = np.zeros((nx,ny))
        #for g,lo,la,y in list_of_results:
        #    print g,lo,la,y
        #    for iy, lat in enumerate(lats):
        #        for ix, lon in enumerate(lons):
        #            if (lon==lo) and (lat==la):
        #                crop_yield[ix,iy] = y
        #print crop_yield, lons, lats, nx, ny
        #'''crop_yield = np.zeros((180,360))
        #lons = -179.5 + np.arange(360)
        #lats = -89.5 + np.arange(180)
        #nx   = 360
        #ny   = 180
        #crop_yield[:,182] = 1'''
        # set up the map
        from mpl_toolkits.basemap import Basemap
        #m = Basemap(projection='laea', lat_0=48, lon_0=16, llcrnrlat=30, 
        #              llcrnrlon=-10, urcrnrlat=65, urcrnrlon=45)
        #m = Basemap(llcrnrlon=-180., llcrnrlat=-90., urcrnrlon=180., urcrnrlat=90.,
        #            lon_0=0.0,lat_0 = 0.0,
        #            rsphere=6371200., projection='cyl')
        m = Basemap(llcrnrlon=-10., llcrnrlat=26., urcrnrlon=72., urcrnrlat=68.,
                    lon_0=18.0,lat_0 = 40.0,
                    rsphere=6371200., projection='aea')
        # convert the data to map coordinates
        print lons,lats
        X, Y = m(lons, lats)
        #yield_map = m.transform_scalar(crop_yield, lons, lats, nx, ny)
        #print yield_map

        # show the map
        #pl = m.imshow(yield_map, cmap=plt.get_cmap('RdYlGn'))
        pl = m.plot(X, Y, 'bo')
        #plt.colorbar(pl)
        m.drawcoastlines()
        m.drawcountries()
        plt.show()

        
#===============================================================================
def plot_precision_yldgapf(list_of_crops, list_of_regions, start_year_, 
                                  end_year_, opti_year_, yield_dict, obs_type_):
#===============================================================================

    crop_name = get_crop_name(list_of_crops)
    # THIS SHOULD NOT BE HARDCODED:
    NUTS_name = dict() # this should not be hardcoded
    NUTS_name['ES41'] = 'Castilla y Leon'
    #####
    nb_years = int(end_year_ - start_year_ + 1.)
    campaign_years = np.linspace(int(start_year_),int(end_year_),nb_years)

    for c,crop in enumerate(list_of_crops):
        for region in list_of_regions:
			# Retrieve the most recent N years of continuous yield data that 
			# have been used for the optimization of the yield gap factor.
            opti_years = define_opti_years(opti_year_, 
                                           yield_dict[0][crop][region][1])

            # produce the plot:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,3))
            fig.subplots_adjust(0.15,0.3,0.95,0.88,0.4,0.6)
            ax.scatter(campaign_years, yield_dict[1][crop][region]['nonopt'], 
                       c='w', marker='v', s=50, label='non-optimized sims')
            ax.fill_between(campaign_years, 
                            yield_dict[1][crop][region]['opt-0.05'],
                            yield_dict[1][crop][region]['opt+0.05'], color='blue', alpha=0.2)
            ax.fill_between(campaign_years, 
                            yield_dict[1][crop][region]['opt-0.02'],
                            yield_dict[1][crop][region]['opt+0.02'], color='green', alpha=0.4)
            ax.fill_between(campaign_years, 
                            yield_dict[1][crop][region]['opt-0.01'],
                            yield_dict[1][crop][region]['opt+0.01'], color='red', alpha=0.6)
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt-0.05'], 
            #           c='k', marker='.', s=50)
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt-0.02'], 
            #           c='k', marker='s', s=50)
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt-0.01'], 
            #           c='k', marker='o', s=50)
            ax.scatter(campaign_years, yield_dict[1][crop][region]['opt'], 
                       c='k', marker='*', s=50, label='optimized sims')
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt+0.01'], 
            #           c='k', marker='o', s=50)
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt+0.02'], 
            #           c='k', marker='s', s=50)
            #ax.scatter(campaign_years, yield_dict[1][crop][region]['opt+0.05'], 
            #           c='k', marker='.', s=50)
            ax.scatter(yield_dict[0][crop][region][1], 
                       yield_dict[0][crop][region][0], c='k', marker='+', 
                       linewidth=2, s=70, label='detrended obs')
            ylims = frame_at_order_of_magnitude(0., 
                              max(yield_dict[1][crop][region]['nonopt']), 1000.)
            ystep = ylims[1] / 5.
            yticks = np.arange(0.,ylims[1]+0.5,ystep)
            ax.set_ylim(ylims)
            ax.set_xlim([start_year_-0.5, end_year_+0.5])
            ax.set_yticks(yticks)
            if (obs_type_=='yield'):
                ax.set_ylabel(r'yield (kg$_{DM}$ ha$^{-1}$)')
            elif (obs_type_=='harvest'):
                ax.set_ylabel(r'harvest (1000 tDM)')
            elif (obs_type_=='area'):
                ax.set_ylabel(r'area (1000 ha)')
            ax.set_xlabel('time (year)')
            ax.set_title('%s (%s) - %s'%(region, NUTS_name[region], 
                                                            crop_name[crop][0]))
            ax.axvspan(opti_years[0]-0.5, opti_years[-1]+0.5, color='grey',
                                                                      alpha=0.3)
            ax.annotate('yldgapf=%.2f'%yield_dict[2][crop][region], xy=(0.89,0.1), 
                        xycoords='axes fraction',horizontalalignment='center',
                        verticalalignment='center')
            plt.legend(loc='best', ncol=3, fontsize=12,
                       bbox_to_anchor = (1.03,-0.25))
            fig.savefig('precision_yldgapf_%s_crop%s_region%s.png'%(obs_type_, crop,\
                                                                        region))

    return None

#===============================================================================
def plot_yield_time_series(list_of_crops, list_of_regions, start_year_, 
                              end_year_, opti_year_, yield_dict, obs_type_):
#===============================================================================

    crop_name = get_crop_name(list_of_crops)
    # THIS SHOULD NOT BE HARDCODED:
    NUTS_name = dict() # this should not be hardcoded
    NUTS_name['ES41'] = 'Castilla y Leon'
    NUTS_name['ES42'] = 'Castilla-la Mancha'
    NUTS_name['ES43'] = 'Extremadura'
    #####
    nb_years = int(end_year_ - start_year_ + 1.)
    campaign_years = np.linspace(int(start_year_),int(end_year_),nb_years)

    for c,crop in enumerate(list_of_crops):
        for region in list_of_regions:
			# Retrieve the most recent N years of continuous yield data that 
			# have been used for the optimization of the yield gap factor.
            opti_years = define_opti_years(opti_year_, 
                                           yield_dict[0][crop][region][1])
            # produce the plot:
            fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(7,3))
            fig.subplots_adjust(0.15,0.3,0.95,0.88,0.4,0.6)
            ax.scatter(campaign_years, yield_dict[1][crop][region]['nonopt'], 
                       c='b', marker='v', s=50, label='non-optimized sims')
            ax.scatter(campaign_years, yield_dict[1][crop][region]['opt'], 
                       c='k', marker='o', s=50, label='optimized sims')
            ax.scatter(yield_dict[0][crop][region][1], 
                       yield_dict[0][crop][region][0], c='r', marker='+', 
                       linewidth=2, s=70, label='detrended obs')
            if (obs_type_=='area'):
                ylims = frame_at_order_of_magnitude(0., 
                          max(max(yield_dict[0][crop][region][0]), 
                              max(yield_dict[1][crop][region]['nonopt'])), 100.)
            else:
                ylims = frame_at_order_of_magnitude(0., 
                              max(yield_dict[1][crop][region]['nonopt']), 1000.)
            ystep = ylims[1] / 5.
            yticks = np.arange(0.,ylims[1]+0.5,ystep)
            ax.set_ylim(ylims)
            ax.set_xlim([start_year_-0.5, end_year_+0.5])
            ax.set_yticks(yticks)
            if (obs_type_=='yield'):
                ax.set_ylabel(r'yield (kg$_{DM}$ ha$^{-1}$)')
            elif (obs_type_=='harvest'):
                ax.set_ylabel(r'harvest (1000 tDM)')
            elif (obs_type_=='area'):
                ax.set_ylabel(r'area (1000 ha)')
            ax.set_xlabel('time (year)')
            ax.set_title('%s (%s) - %s'%(region, NUTS_name[region], 
                                                            crop_name[crop][0]))
            ax.axvspan(opti_years[0]-0.5, opti_years[-1]+0.5, color='grey',
                                                                      alpha=0.3)
            ax.annotate('yldgapf=%.3f'%yield_dict[2][crop][region], xy=(0.89,0.1), 
                        xycoords='axes fraction',horizontalalignment='center',
                        verticalalignment='center')
            plt.legend(loc='best', ncol=3, fontsize=12,
                       bbox_to_anchor = (1.03,-0.25))
            fig.savefig('time_series_%s_crop%s_region%s.png'%(obs_type_, crop,\
                                                                        region))

    return None

#===============================================================================
def plot_optiYLDGAPF_asf_cases(crop_name_, list_of_regions, list_of_cases, 
                                        list_of_labels, list_of_colors, opti_f):
#===============================================================================

# need to loop over the crops within the function, not outside

    for r,region in enumerate(list_of_regions):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.92,0.4,0.05)
        ind = np.arange(len(list_of_cases)) # the x location for the groups
        width = 1. # width of the bars
        ax.bar(ind, opti_f[region][::-1], width, color=list_of_colors[::-1])
        ax.set_ylabel('optimum YLDGAPF (-)')
        ylims = frame_at_order_of_magnitude(min(opti_f[region]), 
                                            max(opti_f[region]), 0.1)
        ax.set_ylim(ylims)
        ax.set_xlim([-0.5,len(list_of_cases)+0.5])
        ax.set_xticks(ind + width/2.)
        xtickNames = ax.set_xticklabels(list_of_labels[::-1])
        plt.setp(xtickNames, rotation=45)
        plt.title('%s - %s'%(region,crop_name_))
        fig.savefig('OYLDGAPF_asf_case_'+region+'.png')

#===============================================================================
def plot_RMSE_asf_YLDGAPF(crop_name_, list_of_regions, list_of_cases,
                           list_of_labels, list_of_colors, RMSE_dict_of_tuples):
#===============================================================================

# need to loop over the crops within the function, not outside

    for region in list_of_regions:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.92,0.4,0.)
        for i,case in enumerate(list_of_cases):
            x,y = zip(*RMSE_dict_of_tuples[region][case])
            ax.plot(x, y, label=list_of_labels[i], color=list_of_colors[i])
            ax.set_xlabel('yldgapf (-)')
            ax.set_ylabel('RMSE')
        plt.title('%s - %s'%(region,crop_name_))
        plt.legend(loc='best')
        fig.savefig('RMSE_asf_YLDGAPF_'+region+'.png')

# add a range of optimum yield gap factor on the graph

#===============================================================================
# function that returns a range that rounds down the min value at first decimal
# and rounds up the maximum value at the first decimal
def frame_at_order_of_magnitude(minvalue, maxvalue, roundvalue):
#===============================================================================
    
    a = np.arange(0.,10000.00000001,roundvalue)
    floor_ind = np.argmax([n for n in (a-minvalue) if n<=0.]) 
    ceil_ind  = np.argmax([n for n in (a-maxvalue) if n<0.]) 
    frame =list([a[floor_ind], a[ceil_ind+1]])

    return frame

#===============================================================================
# Function to retrieve yields from forward runs (optimized and non-optimized) 
# and to aggregate those into regional yields
def compute_regional_yields_series(list_of_crops, list_of_regions, start_year_, 
                                             end_year_, obs_type_, folder_sims):
#===============================================================================

    # calculating some other variables from user input:
    crop_name = get_crop_name(list_of_crops)
    # THIS SHOULD NOT BE HARDCODED:
    NUTS_name = dict() # this should not be hardcoded
    NUTS_name['ES41'] = 'Castilla y Leon'
    #####
    nb_years = int(end_year_ - start_year_ + 1.)
    campaign_years = np.linspace(int(start_year_),int(end_year_),nb_years)

    # Retrieve the observational dataset:
    NUTS_data   =  open_csv_EUROSTAT(EUROSTATdir,
                                     ['agri_yields_NUTS1-2-3_1975-2014.csv'],
                                     convert_to_float=True)

    # we simplify the dictionaries keys:
    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
        
    opti_yldgapf   = dict()
    Regional_yield = dict()
    detrend        = dict()

    for crop in list_of_crops:

        opti_yldgapf[crop]   = dict()
        Regional_yield[crop] = dict()
        detrend[crop]        = dict()

        for region in list_of_regions: # for each region of our list:

            # Retrieve the crop fraction of each region:
            frac_crop = get_EUR_frac_crop(crop_name[crop][1], NUTS_name[region], 
                        EUROSTATdir)
            print 'EUROSTAT crop area fraction:', frac_crop

            DM_content = retrieve_crop_DM_content(crop, region)
            # detrend the yield observations
            detrend[crop][region] = detrend_obs(1975, 2014, NUTS_name[region], 
                                    crop_name[crop][1], NUTS_data['yields'], 
                                    DM_content, 2000, obs_type=obs_type_, 
                                    prod_fig=False)
        
            # Retrieve the forward simulation datasets (one for the optimized
            # and one for the non-optimized simulations)
            list_of_files = [f for f in os.listdir(ForwardSimdir) if ((region 
                         in f) and ('ForwardSim' in f)) and not ('.swp' in f)]
            print list_of_files
            Forward_Sim   = open_csv(ForwardSimdir, list_of_files)

            # we identify the dictionaries keys:
            key_opt1   = [f for f in list_of_files if ('_Opt_0.575_' in f)][0]
            key_opt2   = [f for f in list_of_files if ('_Opt_0.605_' in f)][0]
            key_opt3   = [f for f in list_of_files if ('_Opt_0.615' in f)][0]
            key_opt4   = [f for f in list_of_files if ('_Opt_crop' in f)][0]
            key_opt5   = [f for f in list_of_files if ('_Opt_0.635_' in f)][0]
            key_opt6   = [f for f in list_of_files if ('_Opt_0.645_' in f)][0]
            key_opt7   = [f for f in list_of_files if ('_Opt_0.675_' in f)][0]
            key_nonopt = [f for f in list_of_files if ('_Non-Opt_crop' in f)][0]

            # aggregate the individual yields into a regional yield
            Regional_yield[crop][region] = dict()
            Regional_yield[crop][region]['opt-0.05'] = \
                     calc_regional_yields(Forward_Sim[key_opt1], campaign_years)
            Regional_yield[crop][region]['opt-0.02'] = \
                     calc_regional_yields(Forward_Sim[key_opt2], campaign_years)
            Regional_yield[crop][region]['opt-0.01'] = \
                     calc_regional_yields(Forward_Sim[key_opt3], campaign_years)
            Regional_yield[crop][region]['opt'] = \
                     calc_regional_values(Forward_Sim[key_opt4], campaign_years,
                                                             frac_crop, 'yield')
            Regional_yield[crop][region]['opt+0.01'] = \
                     calc_regional_yields(Forward_Sim[key_opt5], campaign_years)
            Regional_yield[crop][region]['opt+0.02'] = \
                     calc_regional_yields(Forward_Sim[key_opt6], campaign_years)
            Regional_yield[crop][region]['opt+0.05'] = \
                     calc_regional_yields(Forward_Sim[key_opt7], campaign_years)
            Regional_yield[crop][region]['nonopt'] = \
                   calc_regional_values(Forward_Sim[key_nonopt], campaign_years,
                                                             frac_crop, 'yield')
            opti_yldgapf[crop][region] = Forward_Sim[key_opt4]['YLDGAPF(-)'][0]

    return detrend, Regional_yield, opti_yldgapf

#===============================================================================
# Function to retrieve yields from forward runs (optimized and non-optimized) 
# and to aggregate those into regional yields
def compute_regional_yield(list_of_crops, list_of_regions, start_year_, 
                                             end_year_, obs_type_, folder_sims):
#===============================================================================

    # calculating some other variables from user input:
    crop_name = get_crop_name(list_of_crops)
    # THIS SHOULD NOT BE HARDCODED:
    NUTS_name = dict() # this should not be hardcoded
    NUTS_name['ES41'] = 'Castilla y Leon'
    NUTS_name['ES42'] = 'Castilla-la Mancha'
    NUTS_name['ES43'] = 'Extremadura'
    #####
    nb_years = int(end_year_ - start_year_ + 1.)
    campaign_years = np.linspace(int(start_year_),int(end_year_),nb_years)


    # Retrieve the observational dataset:
    if (obs_type_ == 'harvest'):
        NUTS_filename = 'agri_harvest_NUTS1-2-3_1975-2014.csv'
    elif (obs_type_ == 'yield'):
        NUTS_filename = 'agri_yields_NUTS1-2-3_1975-2014.csv'
    elif (obs_type_ == 'area'):
        NUTS_filename = 'agri_croparea_NUTS1-2-3_1975-2014.csv'
    NUTS_data   =  open_csv_EUROSTAT(EUROSTATdir, [NUTS_filename],
                                     convert_to_float=True)

    opti_yldgapf   = dict()
    Regional_yield = dict()
    detrend        = dict()

    for crop in list_of_crops:

        opti_yldgapf[crop]   = dict()
        Regional_yield[crop] = dict()
        detrend[crop]        = dict()

        for region in list_of_regions: # for each region of our list:

            # Retrieve the crop fraction of each region:
            frac_crop = get_EUR_frac_crop(crop_name[crop][1], NUTS_name[region], 
                                          EUROSTATdir)
            print 'EUROSTAT crop area fraction:', frac_crop

            DM_content = retrieve_crop_DM_content(crop, region)
            # detrend the yield observations (except area)
            detrend[crop][region] = detrend_obs(1975, 2014, NUTS_name[region], 
                                    crop_name[crop][1], NUTS_data[NUTS_filename], 
                                    DM_content, 2000, obs_type=obs_type_, 
                                    prod_fig=False)
            # Retrieve the forward simulation datasets (one for the optimized
            # and one for the non-optimized simulations)
            list_of_files = [f for f in os.listdir(ForwardSimdir) if ((region 
                             in f) and ('ForwardSim' in f) and ('Opt_crop' in f)
                             and not ('.swp' in f))]
            Forward_Sim   = open_csv(ForwardSimdir, list_of_files)

            # we identify the dictionaries keys:
            key_opt    = [f for f in list_of_files if ('_Opt_crop' in f)][0]
            key_nonopt = [f for f in list_of_files if ('_Non-Opt_crop' in f)][0]

            # aggregate the individual yields into a regional yield
            Regional_yield[crop][region] = dict()
            Regional_yield[crop][region]['opt'] = \
                                   calc_regional_values(Forward_Sim[key_opt], 
                                                        campaign_years, 
                                                        frac_crop,
                                                        obs_type_)
            Regional_yield[crop][region]['nonopt'] = \
                                   calc_regional_values(Forward_Sim[key_nonopt], 
                                                        campaign_years, 
                                                        frac_crop,
                                                        obs_type_)
            opti_yldgapf[crop][region] = Forward_Sim[key_opt]['YLDGAPF(-)'][0]

    return detrend, Regional_yield, opti_yldgapf

#===============================================================================
# Function to load the RMSE pickle file generated during the sensitivity 
# analysis
def retrieve_RMSE_and_YLDGAPF(list_of_regions, list_of_cases):
#===============================================================================

    from cPickle import load as pickle_load
    from operator import itemgetter as operator_itemgetter

    RMSE      = dict()
    O_YLDGAPF = dict()
    for region in list_of_regions:
        RMSE[region]      = dict()
        optimum_yldgapf   = []
        for case in list_of_cases:
            # find the filename that matches the region and case
            filename = [f for f in os.listdir(RMSEdir) if (f.startswith('RMSE_')
                        and (region in f) and (case in f))][0]
            # make the file path
            filepath = os.path.join(RMSEdir, filename)
            # load the file
            RMSE[region][case] = pickle_load(open(filepath, 'rb'))
            # we sort the list of tuples by value of RMSE, and retrieve the 
            # yldgapf corresponding to the min RMSE:
            optimum_yldgapf = optimum_yldgapf + [sorted(RMSE[region][case], 
                                            key = operator_itemgetter(1))[0][0]]
        O_YLDGAPF[region] = list(optimum_yldgapf) 

    return RMSE, O_YLDGAPF

#===============================================================================
# Function to aggregate the individual yields into the yearly regional yields
def calc_regional_values(FINAL_YLD, campaign_years_, frac_crop_, _obs_type_):
#===============================================================================

    import math

    TSO_regional = np.zeros(len(campaign_years_))
    # we calculate one regional value per year
    for y,year in enumerate(campaign_years_):
        indx_crop            = np.array([math.pow(j,2) for j in \
                                         (frac_crop_[1]-year)]).argmin()
        frac_crop            = frac_crop_[0][indx_crop]
        sum_weighted_yields  = 0.
        sum_weights          = 0.
        # we read each line of the ouput file
        for l in range(0,len(FINAL_YLD['year'])):
            # if the line concerns the year we are interested in
            if (FINAL_YLD['year'][l] == year):
                # we multiply the yield with the area cultivated for the crop
                frac_arable         = FINAL_YLD['arable_area(ha)'][l] / 62500.
                frac_culti          = frac_arable * frac_crop
                area_culti          = FINAL_YLD['stu_area(ha)'][l] * frac_culti   
                sum_weighted_yields = sum_weighted_yields + \
                                      FINAL_YLD['TSO(kgDM.ha-1)'][l] * area_culti 
                # we sum the cultivated area
                sum_weights         = sum_weights + area_culti
            else:
                pass
        if (_obs_type_ == 'yield'):
            # area-weighted average of the yield
            TSO_regional[y] = sum_weighted_yields / sum_weights # kgDM/ha
        elif (_obs_type_ == 'harvest'):
            # sum of the individual harvests
            TSO_regional[y] = sum_weighted_yields / 1000000. # 1000 tDM
        elif (_obs_type_ == 'area'):
            # sum of the individual cultivated areas
            TSO_regional[y] = sum_weights / 1000. # 1000 ha

    return TSO_regional

#===============================================================================
# Function to aggregate the individual yields into the yearly regional yields
def calc_regional_yields(FINAL_YLD, campaign_years_):
#===============================================================================

    TSO_regional = np.zeros(len(campaign_years_)) # empty 1D array
    for y,year in enumerate(campaign_years_):
        sum_weighted_yields  = 0.
        sum_weights          = 0.
        for l in range(0,len(FINAL_YLD['year'])):
            if (FINAL_YLD['year'][l] == year):
                # adding weighted 2D-arrays in the empty array sum_weighted_yields
                sum_weighted_yields = sum_weighted_yields + \
                        FINAL_YLD['weight(-)'][l]*FINAL_YLD['TSO(kgDM.ha-1)'][l] 
                # computing the sum of the soil type area weights
                sum_weights         = sum_weights + FINAL_YLD['weight(-)'][l]
            else:
                pass
        TSO_regional[y] = sum_weighted_yields / sum_weights # weighted average 

    return TSO_regional

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
