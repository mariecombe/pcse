#!/usr/bin/env python

import sys, os
import numpy as np
from csv import reader as csv_reader
from pickle import load as pickle_load
from random import sample as random_sample
from string import replace as string_replace
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider


#===============================================================================
# This script execute the optimized WOFOST runs for one NUTS region
def main():
#===============================================================================

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
    DM_content = 0.9  # this is the observed DM fraction
    # Period of time for the WOFOST runs:
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
# Retrieve observed yield datasets
    
#    NUTS_data     = open_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'],
#                             convert_to_float=True)
#    NUTS_ids      = open_csv(EUROSTATdir,['NUTS_codes_2013.csv'],
#                             convert_to_float=False)
#    
#    # we simplify the dictionaries keys:
#    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
#    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
#    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
#    del NUTS_ids['NUTS_codes_2013.csv']
    
#------------------------------------------------------------------------------
# Detrend the yield observations

#    detrend   = detrend_obs_yields(start_year, end_year, NUTS_name, crop_name,
#								   NUTS_data['yields'], DM_content, 2000)
    
#------------------------------------------------------------------------------
# Select the grid cells to loop over

    # we first read the list of all 'whole' grid cells contained in that region
	# NB: grid_list_tuples[0] is a list of (grid_cell_id, arable_land_area)
	# tuples, which are already sorted by decreasing amount of arable land
    filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
    grid_list_tuples = pickle_load(open(filename,'rb'))
    print 'Reading data from pickle file %s'%filename

    # we select a subset of grid cells to loop over
    selected_grid_cells = select_grid_cells(grid_list_tuples[0],method='topn',n=10)
    #selected_grid_cells=select_grid_cells(grid_list_tuples[0],method='randomn',n=10)
    #selected_grid_cells=select_grid_cells(grid_list_tuples[0],method='all')

#------------------------------------------------------------------------------
# Select the soil types to loop over  

    # we first read the list of suitable soil types for our chosen crop 
    filename = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no
    suitable_soil_types = pickle_load(open(filename,'rb')) 
    print 'Reading data from pickle file %s'%filename

    selected_soil_types = {}

    for grid, area_arable in selected_grid_cells:

        # we read the entire list of soil types contained within the grid cell
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        print 'Reading data from pickle file %s'%filename
        soil_iterator = pickle_load(open(filename,'rb'))

        # We select a subset of soil types to loop over
#        selected_soil_types[grid] = select_soil_types(grid, soil_iterator)

#------------------------------------------------------------------------------
# Perform optimization of YLDGAPF and yield

    sys.exit(2)      
    
    if yield_sim_no_optimize == True: 
	    perform_yield_sims_no_optimize(NUTS_no, crop_no, campaign_years,
									   grid_list_tuples, soil_list_tuples)





#------------------------------------------------------------------------------
# Do forward simulations for the grid cells x soil type combinations


#------------------------------------------------------------------------------
# 

#===============================================================================
# Function to select a subset of soil types within a grid cell
def select_soil_types(grid_cell_id, soil_iterator_, method='topn', n=3):
#===============================================================================

    list_soils = []
    
    # loop over the soil types
    for smu_no, area_smu, stu_no, percentage, soildata in soil_iterator_:

        # NB: we remove all unsuitable soils from the iteration
        if (stu_no not in suitable_stu): 
            continue
        else:
            # calculate the cultivated area of the crop in this grid
            area_soil      = area_smu*percentage/100.
            perc_area_soil = area_soil/625000000
                
            if (perc_area_soil <= 0.05):
                continue 
            else:
                list_soils = list_soils + [(smu_no,stu_no)]
            
    selected_soils = list_soils

    print 'list of selected soil types:', [st for sm,st in selected_soils]

    return selected_soils

#===============================================================================
# Function to select a subset of grid cells within a NUTS region
def select_grid_cells(list_of_tuples, method='topn', n=3):
#===============================================================================

	# NB: list_of_tuples is a list of (grid_cell_id, arable_land_area) tuples,
	# which are already sorted by decreasing amount of arable land
    
    # first option: we select the top n grid cells in terms of arable land area
    if   (method == 'topn'):
        subset_list   = list_of_tuples[0:n]
    # second option: we select a random set of n grid cells
    elif (method == 'randomn'):
        subset_list   = random_sample(list_of_tuples,n)
    # last option: we select all available grid cells of the region
    else:
        subset_list   = list_of_tuples
    print 'list of selected grids:', [g for g,a in subset_list]

    return subset_list

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
    #print 'observed dry matter yields:', TARGET

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
    
    #print 'detrended dry matter yields:', OBS['DETRENDED']
    
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
