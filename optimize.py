#!/usr/bin/env python

import sys, os
import numpy as np
from csv import reader as csv_reader
from pickle import load as pickle_load
from random import sample as random_sample
from string import replace as string_replace
from operator import itemgetter as operator_itemgetter
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider


#===============================================================================
# This script executes WOFOST runs for one NUTS region and optimizes its YLDGAPF
def main():
#===============================================================================

# Define the settings of the run
 
    # NUTS region:
    NUTS_no       = 'ES41'
    NUTS_name     = 'Castilla y Leon'

    # Crop:
    crop_no       = 3        # CGMS crop number
    crop_name     = 'Barley' # EUROSTAT crop name (check EUROSTAT nomenclature)
    DM_content    = 0.9      # EUROSTAT dry matter fraction
                             # should be read from file rather than hardcoded
    # Period of time:
    start_year    = 1975
    end_year      = 2014

    # options for the selection of grid cell x soil types combinations
    selec_method  = 'topn'   # can be 'topn' or 'randomn' or 'all'
    ncells        = 10       # number of selected grid cells within a region
    nsoils        = 3        # number of selected soil types within a grid cell

    # options for the yield gap factor optimization
    optimization  = False    # if False: we assign a YLDGAPF = 1.
                             # if True: we optimize YLDGAPF with one of the 
                             # following methods
    opti_method   = 'average_combi' # can be 'individual' or 'average_combi'
                             # or 'aggregated_yield' or 'aggregated_harvest'
    opti_nyears   = 3

#-------------------------------------------------------------------------------
# Calculate key variables from the user input

    nb_years      = int(end_year - start_year + 1.)
    campaign_years = np.linspace(int(start_year),int(end_year),nb_years)

#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research project 3/'\
				   +'EUROSTAT_data'
    folderpickle  = 'pickled_CGMS_input_data/'
	
    # directories on capegrim:
    #folderpickle  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"

#-------------------------------------------------------------------------------
# Retrieve the observed yield and remove the technological trend 
    
    NUTS_data     = open_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'],
                             convert_to_float=True)
    NUTS_ids      = open_csv(EUROSTATdir,['NUTS_codes_2013.csv'],
                             convert_to_float=False)
    
    # we simplify the dictionaries keys:
    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    del NUTS_ids['NUTS_codes_2013.csv']
    
    # Detrend the yield observations
    detrend   = detrend_obs_yields(start_year, end_year, NUTS_name, crop_name,
								   NUTS_data['yields'], DM_content, 2000)
    
#-------------------------------------------------------------------------------
# Select the grid cells to loop over

    # we first read the list of all 'whole' grid cells contained in that region
	# NB: grid_list_tuples[0] is a list of (grid_cell_id, arable_land_area)
	# tuples, which are already sorted by decreasing amount of arable land
    filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
    grid_list_tuples = pickle_load(open(filename,'rb'))
    print 'Reading grid cells list from pickle file %s'%filename

    # we select a subset of grid cells to loop over
    selected_grid_cells = select_grid_cells(grid_list_tuples[0], 
                                            method=selec_method, n=ncells)

#-------------------------------------------------------------------------------
# Select the soil types to loop over  

    # we first read the list of suitable soil types for our chosen crop 
    filename = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no
    suit_soils = pickle_load(open(filename,'rb')) 
    print 'Reading suitable soil types list from pickle file %s'%filename

    selected_soil_types = {}

    for grid, area_arable in selected_grid_cells:

        # we read the entire list of soil types contained within the grid cell
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        print 'Reading soil data from pickle file %s'%filename
        soils = pickle_load(open(filename,'rb'))

        # We select a subset of soil types to loop over
        selected_soil_types[grid] = select_soils(grid, soils, suit_soils, 
                                                 method=selec_method, n=nsoils)

#-------------------------------------------------------------------------------
# Perform optimization of the yield gap factor

    if (optimization == True):
        if (opti_method == 'average_combi'):
            # 1- do forward simulations for all grid cells x soil combinations
            # 2- select the most average combination from it
            # 3- optimize YLDGAPF using this one combi only
        else:
            # for all other methods (individual or aggregated):
            optimum_yldgapf = optimize_yldgapf_dyn(crop_no, campaign_years,
                                                   selected_grid_cells,
                                                   selected_soil_types,
                                                   method=opti_method,
                                                   nyears=opti_years)
    else:
        optimum_yldgapf = 1.

#-------------------------------------------------------------------------------
# Do forward simulations for all available grid cells x soil type combinations

    sys.exit(2)
    simulated_yields = perform_yield_sims(crop_no, campaign_years, 
                                          yldgapf=optimum_yldgapf, 
                                          selected_grid_cells,
                                          selected_soil_types)
#    pickle.dump(sim_yields, open( "list_of_sim_yld_%i-%i.pickle"
#                             %(campaign_years_[0],campaign_years_[-1]), "wb") )



#===============================================================================
#===============================================================================
# Function to optimize the yield gap factor within a NUTS region
# this function iterates dynamically to find the optimum YLDGAPF
def optimize_yldgapf_dyn(crop_no_, campaign_years_, selected_grid_cells_,
                     selected_soil_types_, method='aggregated_yield', nyears=3):
#===============================================================================

    # 1- calculate on which years we will loop (last observed year - nyears)

    # 2- code the optimization for an aggregated method
    # NB: the individual method has already been coded so copy it here

    # think about the structure: do we do dynamical simulation with 9 iterations
    # or do we simulate all parameter space which we analyze afterwards
 
            #yldgapf_range = np.linspace(0.1,1.,5.)
            #nb_yldgapf    = 5

                if (optimize_ == True):
                    lowestf  = yldgapf_range[0]
                    highestf = yldgapf_range[-1]
		            f_step   = highestf - lowestf
	                while (f_step >= 0.1):
                        print '...'
	                    # we build a range of YLDGAPF to explore until we obtain
                        # a good enough precision on the YLDGAPF
	                    f_step = (highestf - lowestf)/2.
	                    middlef = lowestf + f_step
	                    f_range = [lowestf, middlef, highestf]

    # 3- add a timestamp to time the function

    # 4- return the optimized YLDGAPF

#===============================================================================
# Function to optimize the yield gap factor within a NUTS region
# this function explores a whole matrix of factors to find the optimum YLDGAPF
def optimize_yldgapf_matrix(crop_no_, campaign_years_, selected_grid_cells_,
                     selected_soil_types_, method='aggregated_yield', nyears=3):
#===============================================================================

    return None

#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sims(crop_no_, campaign_years_, yldgapf=1., 
                       selected_grid_cells_, selected_soil_types_):
#===============================================================================

    FINAL_YLD = []
    for grid in selected_grid_cells_:

        # Retrieve the weather data of one grid cell (all years are in one file) 
        filename = folderpickle+'weatherobject_g%d.pickle'%grid
        weatherdata = WeatherDataProvider()
        print 'Reading data from pickle file %s'%filename
        weatherdata._load(filename)
                    
        # Retrieve the soil data of one grid cell 
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soil_iterator = pickle.load(open(filename,'rb'))
        print 'Reading data from pickle file %s'%filename

        for y, year in enumerate(campaign_years_): 

            # Retrieve calendar data of one year for one grid cell
            filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'%(grid,
                                                                  crop_no_,year)
            timerdata = pickle.load(open(filename,'rb'))
            print 'Reading data from pickle file %s'%filename
                            
            # Retrieve crop data of one year for one grid cell
            filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'%(grid,
                                                                  crop_no_,year)
            cropdata = pickle.load(open(filename,'rb'))
            print 'Reading data from pickle file %s'%filename

            # Set the yield gap factor
            cropdata['YLDGAPF'] = yldgapf

            for soil_type in selected_soil_types_[grid]:
                
                # Retrieve the site data of one soil, one year, one grid cell
                filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'%(
                                                      grid,crop_no_,year,stu_no)
                sitedata = pickle.load(open(filename,'rb'))
                print 'Reading data from pickle file %s'%filename

                # run WOFOST
                wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, 
                                                cropdata, weatherdata)
                wofost_object.run_till_terminate()
    
                # get the yield (in gDM.m-2) and the yield gap
                TSO = wofost_object.get_variable('TWSO')
            
                FINAL_YLD = FINAL_YLD + [(grid, year, stu_no, TSO)]

	
    print "All forward simulations completed!"

    return FINAL_YLD

#===============================================================================
# Function to select a subset of soil types within a grid cell
def select_soils(grid_cell_id, soil_iterator_, suitable_soils, method='topn', n=3):
#===============================================================================

    # Rank soils by decreasing area
    sorted_soils = []
    for smu_no, area_smu, stu_no, percentage_stu, soildata in soil_iterator_:
        if stu_no not in suitable_soils: continue
        weight_factor  =  area_smu * percentage_stu
        sorted_soils   =  sorted_soils + [(smu_no, stu_no, weight_factor)]
    sorted_soils = sorted(sorted_soils, key=operator_itemgetter(2), reverse=True)
   
    # select a subset of soil types to loop over 
    # first option: we select the top n most present soils in the grid cell
    if   (method == 'topn'):
        subset_list   = sorted_soils[0:n]
    # second option: we select a random set of n soils within the grid cell
    elif (method == 'randomn'):
        subset_list   = random_sample(sorted_soils,n)
    # last option: we select all available soils in the grid cell
    else:
        subset_list   = sorted_soils
    #print 'list of selected soils:', [st for sm,st,w in subset_list]
    
    return subset_list

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
    #print 'list of selected grids:', [g for g,a in subset_list], '\n'

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
