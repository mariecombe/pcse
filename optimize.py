#!/usr/bin/env python

#import pcse
from pcse.db.cgms11 import WeatherObsGridDataProvider, TimerDataProvider
from pcse.db.cgms11 import SoilDataIterator, CropDataProvider, STU_Suitability
from pcse.db.cgms11 import SiteDataProvider
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider
from random import shuffle, sample
#import matplotlib.pyplot as plt
#import sqlalchemy as sa  #only required to connect to the oracle database
from sqlalchemy import MetaData, select, Table
#import pandas as pd
import os
from csv import reader as csv_reader
from datetime import date
import numpy as np
from string import replace as string_replace
from pickle import load as pickle_load
#import math
#import sys, os, getopt
#from matplotlib import pyplot
#from scipy import ma

#===============================================================================
# Start of the script
def main():
#===============================================================================

#    try:                                
#        opts, args = getopt.getopt(sys.argv[1:], "-h")
#    except getopt.GetoptError:           
#        print "Error"
#        sys.exit(2)      
#    
#    for options in opts:
#        options=options[0].lower()
#        if options == '-h':
#            helptext = """
#    This script execute the WOFOST runs for one location
#
#                """
#            print helptext
#            sys.exit(2)     
 
#-------------------------------------------------------------------------------
# Define working directories

    currentdir   = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir  = '/Users/mariecombe/Documents/Work/Research project 3/'\
				   +'EUROSTAT_data'
    folderpickle = 'pickled_CGMS_input_data/'
	
    # directories on capegrim:
    #folderpickle = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    #EUROSTATdir  = "/Users/mariecombe/Cbalance/EUROSTAT_data"

#------------------------------------------------------------------------------
# Define the settings of the run

    # option (A)
    # switch on if you want to do forward runs without optimization:
    # returns a pickle file containing simulated yields 
    yield_sim_no_optimize                  = True 
    # option (B)
	# switch on if you want to do individual optimization of YLDGAP:
	# returns - a pickle file containing the list of optimum YLDGAPF
	#         - and a pickle file containing the simulated optimized yields
    yield_sim_each_optimize                = False

    # option (C)
	# switch on if you want to optimize the YLDGAP factor of the most average 
    # grid cell x soil combination of the region
	# returns the optimized YLDGAPF and simulated yield printed on screen
	# NB: can only be done after doing option (A)
    optimize_YLDGAPF_of_average_combi      = False
    # option (D)
	# switch on if you want to do optimization of the aggregated yield/harvest
	# returns the optimized YLDGAPF and simulated yield printed on screen
	# NB: can only be done after doing forward runs without optimization
    optimize_YLDGAPF_of_aggregated_yield   = False
    optimize_YLDGAPF_of_aggregated_harvest = False
 
    # Region information:
    NUTS_no    = 'ES41'
    NUTS_name  = 'Castilla y Leon'
    # Crop information:
    crop_no    = 3
    crop_name  = 'Barley'
    DM_content = 0.9  # this is the DM fraction in the reported observed yields
    # Period of time:
    start_year = 1975
    end_year   = 2014
    # Yield gap factor optimization options:
    yldgapf_range = np.linspace(0.1,1.,5.)
    nb_yldgapf = 5

#------------------------------------------------------------------------------
# Calculate key variables from the user input

    nb_years   = int(end_year - start_year + 1.)
    campaign_years = np.linspace(int(start_year),int(end_year),nb_years)

#------------------------------------------------------------------------------
# Retrieve observed yield datasets
    
    NUTS_data     = open_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'],
                             convert_to_float=True)
    NUTS_ids      = open_csv(EUROSTATdir,['NUTS_codes_2013.csv'],
                             convert_to_float=False)
    
    # we simplify the dictionaries keys:
    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    del NUTS_ids['NUTS_codes_2013.csv']
    
#------------------------------------------------------------------------------
# Detrend the yield observations

    detrend   = detrend_obs_yields(start_year, end_year, NUTS_name, crop_name,
								   NUTS_data['yields'], DM_content, 2000)

    
#------------------------------------------------------------------------------
# Retrieve or select the grid cells to loop over

    # we get the list of all grid cells contained in that region
    # the list of grid cells has been pickled, and is already ranked by size of
    # arable land area in the pickle file
    filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
    grid_list_tuples = pickle_load(open(filename,'rb'))
    # NB: grid_list_tuples[0] contains the (grid,arable_land_area) tuples
    #     grid_list_tuples[1] contains the criteria of selection of grids 
    #     (e.g. 'all' if all grid cells are listed in the pickle file)
    print 'Reading data from pickle file %s'%filename

    n = int(10.)
    # We select a subset of those grid cells
    # first option: we select the top n grid cells in terms of arable land area
    if (select_crit_grid=='topn'):
        grid_list = [g for g,a in grid_list_tuples[0][0:n]]
    # second option: we select a random set of n grid cells
    elif (select_crit_grid == 'randomn'):
        mix_list = random.shuffle(grid_list_tuples[0])
        grid_list = [g for g,a in mix_list]
        #grid_list = sample(grid_list_tuples[0], n)
    # last option: we select all available grid cells of the region
    else:
        grid_list = [g for g,a in grid_list_tuples[0]]
    print 'list of selected grids:', grid_list

    sys.exit(2)      

#------------------------------------------------------------------------------
# Retrieve or select the soil types to loop over  


#------------------------------------------------------------------------------
# Perform optimization of YLDGAPF and yield

    if yield_sim_no_optimize == True: 
	    perform_yield_sims_no_optimize(NUTS_no, crop_no, campaign_years,
									   grid_list_tuples, soil_list_tuples)

#------------------------------------------------------------------------------
# Do forward simulations for the grid cells x soil type combinations


#------------------------------------------------------------------------------
# 


#===============================================================================
# Function to detrend the observed yields
def detrend_obs_yields( _start_year, _end_year, _NUTS_name, _crop_name, 
                       uncorrected_yields_dict, _DM_content, base_year, 
                       prod_fig=False):
#===============================================================================

    nb_years = int(_end_year - _start_year + 1.)
    campaign_years = np.linspace(int(_start_year), int(_end_year), nb_years)
    OBS = {}
    TREND = {}
    
    # search for the index of base_year item in the campaign_years array
    for i,val in enumerate(campaign_years): 
        if val == base_year:
            indref = i
    
    # select yields for the required region, crop and period of time
    # and convert them from kg_humid_matter/ha to g_dry_matter/m2 
    TARGET = np.array([0.]*nb_years)
    for j,year in enumerate(campaign_years):
        for i,region in enumerate(uncorrected_yields_dict['GEO']):
            if region.startswith(_NUTS_name[0:12]):
                if uncorrected_yields_dict['CROP_PRO'][i]==_crop_name:
                    if (uncorrected_yields_dict['TIME'][i]==str(int(year))):
                        if (uncorrected_yields_dict['STRUCPRO'][i]==
                                                      'Yields (100 kg/ha)'):
                            TARGET[j] = float(uncorrected_yields_dict['Value'][i])\
                                               *10.*_DM_content
    print 'observed dry matter yields:', TARGET

    # fit a linear trend line in the record of observed yields
    mask = ~np.isnan(TARGET)
    z = np.polyfit(campaign_years[mask], TARGET[mask], 1)
    p = np.poly1d(z)
    OBS['ORIGINAL'] = TARGET[mask]
    TREND['ORIGINAL'] = p(campaign_years)
    
    # calculate the anomalies to the trend line
    ANOM = TARGET - (z[0]*campaign_years + z[1])
    
    # Detrend the observed yield data
    OBS['DETRENDED'] = ANOM[mask] + p(base_year)
    z2 = np.polyfit(campaign_years[mask], OBS['DETRENDED'], 1)
    p2 = np.poly1d(z2)
    TREND['DETRENDED'] = p2(campaign_years)
    
    # if needed plot a figure showing the yields before and after de-trending
    if prod_fig==True:
        pyplot.close('all')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
        for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
            ax.scatter(campaign_years[mask], OBS[var], c='b')
       	    ax.plot(campaign_years,TREND[var],'r-')
       	    ax.set_ylabel('%s yield (gDM m-2)'%var, fontsize=14)
            ax.set_xlabel('time (year)', fontsize=14)
        fig.savefig('observed_yields.png')
        print 'the trend line is y=%.6fx+(%.6f)'%(z[0],z[1])
        pyplot.show()
    
    print 'detrended dry matter yields:', OBS['DETRENDED']
    
    return OBS['DETRENDED'], campaign_years[mask]


#===============================================================================
# Function to open csv files
def open_csv(inpath,filelist,convert_to_float=False):
#===============================================================================

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        print "Opening %s"%(namefile)

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
        print headerow

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]

        # two possibilities: either convert data from string to float or
        # keep it as is in string type
        if (convert_to_float == True):
            # transforming data from string to float type
            converted_data=[]
            for line in lines:
                if (line[4] != ':'): 
                    a = (line[0:4] + [float(string_replace(line[4], ' ', ''))] 
                                   + [line[5]])
                else:
                    a = line[0:4] + [float('NaN')] + [line[5]]
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
    
        print "Dictionary created!\n"

    return Dict

#===============================================================================
if __name__=='__main__':
  main()
#===============================================================================
