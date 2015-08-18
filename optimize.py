#!/usr/bin/env python

import sys, os
import math
import numpy as np
from csv import reader as csv_reader
from pickle import load as pickle_load
from random import sample as random_sample
from string import replace as string_replace
from operator import itemgetter as operator_itemgetter
from datetime import datetime
from matplotlib import pyplot as plt
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider


#===============================================================================
# This script executes WOFOST runs for one NUTS region and optimizes its YLDGAPF
def main():
    global currentdir, EUROSTATdir, folderpickle, detrend
#===============================================================================

# Define the settings of the run
 
    # NUTS region:
    NUTS_no       = 'ES43'
    NUTS_name     = 'Extremadura'

    # Crop:
    crop_no       = 3        # CGMS crop number
    crop_name     = 'Barley' # EUROSTAT crop name (check EUROSTAT nomenclature)
    DM_content    = 0.9      # EUROSTAT dry matter fraction
                             # should be read from file rather than hardcoded

    # yield gap factor optimization:
    optimization  = True     # if False: we assign a YLDGAPF = 1.
                             # if True: we optimize YLDGAPF with one of the 
                             # following methods
    # selection of grid cell x soil types combinations for the optimization
    selec_method  = 'topn'   # can be 'topn' or 'randomn' or 'all'
    ncells        = 2        # number of selected grid cells within a region
    nsoils        = 2        # number of selected soil types within a grid cell
    # optimization method options
    opti_method   = 'aggregated_yield' # can be 'individual' or 'average_combi'
                             # or 'aggregated_yield' or 'aggregated_harvest'
    opti_nyears   = 3        # nb of years considered for the optimization
                             # minimum 3 years, maximum 30.

    # forward crop growth simulations:
    forward_sims  = False
    start_year    = 2000     # start_year and end_year define the period of time
    end_year      = 2014     # for which we do forward simulations

#-------------------------------------------------------------------------------
# Calculate key variables from the user input

    nb_years      = int(end_year - start_year + 1.)
    campaign_years = np.linspace(int(start_year),int(end_year),nb_years)

#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'
    folderpickle  = '/Users/mariecombe/Documents/Work/Research_project_3/'\
                   +'pcse/pickled_CGMS_input_data/'

    # directories on capegrim:
    #folderpickle  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"

#-------------------------------------------------------------------------------
# Retrieve the observed yield and remove the technological trend 
   
    if (optimization == True): 

        NUTS_data = open_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'],
                                 convert_to_float=True)
        NUTS_ids  = open_csv(EUROSTATdir,['NUTS_codes_2013.csv'],
                                 convert_to_float=False)
    
        # we simplify the dictionaries keys:
        NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
        NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
        del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
        del NUTS_ids['NUTS_codes_2013.csv']
    
        # Detrend the yield observations
        detrend = detrend_obs_yields(start_year, end_year, NUTS_name, crop_name,
		    						   NUTS_data['yields'], DM_content, 2000)

#-------------------------------------------------------------------------------
# Determine on which years we will optimize the yield gap factor

		# Retrieve the most recent N years of continuous yield data that we can
		# use for the optimization of the yield gap factor.
        opti_years = find_consecutive_years(detrend[1], opti_nyears)

#-------------------------------------------------------------------------------
# Select the grid cells to loop over for the optimization

        # we first read the list of all 'whole' grid cells of that region
        # NB: grid_list_tuples[0] is a list of (grid_cell_id, arable_land_area)
        # tuples, which are already sorted by decreasing amount of arable land
        filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
        grid_list_tuples = pickle_load(open(filename,'rb'))
 
        # we select a subset of grid cells to loop over
        selected_grid_cells = select_grid_cells(grid_list_tuples[0], 
                                                method=selec_method, n=ncells)
        print '\nWe have selected', len(selected_grid_cells),'grid cells:',\
                      [g for g,a in selected_grid_cells]

#-------------------------------------------------------------------------------
# Select the soil types to loop over for the optimization

        # we first read the list of suitable soil types for our chosen crop 
        filename = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no
        suit_soils = pickle_load(open(filename,'rb')) 
 
        selected_soil_types = {}
 
        for grid in [g for g,a in selected_grid_cells]:
 
            # we read the list of soil types contained within the grid cell
            filename = folderpickle+'soilobject_g%d.pickle'%grid
            soils = pickle_load(open(filename,'rb'))
 
            # We select a subset of soil types to loop over
            selected_soil_types[grid] = select_soils(grid, soils, suit_soils, 
                                                 method=selec_method, n=nsoils)
            print 'We have selected',len(selected_soil_types[grid]),\
                  'soil types:', [stu for smu, stu, w, data in \
                  selected_soil_types[grid]],'for grid', grid

#-------------------------------------------------------------------------------
# Perform optimization of the yield gap factor

        plot_filename = 'crop%i_region%s_%s_%igx%is_%i-%i'%(crop_no, NUTS_no,
                             selec_method, ncells, nsoils, start_year, end_year)
        if (opti_method == 'average_combi'):
            pass
            # 1- do forward simulations for all grid cells x soil combinations
            # 2- select the most average combination from it
            # 3- optimize YLDGAPF using this one combi only
        elif (opti_method == 'individual'):
            optimum_yldgapf = optimize_yldgapf_dyn_individual(crop_no,
                                                   selected_grid_cells,
                                                   selected_soil_types,
                                                   opti_years)
        elif (opti_method == 'aggregated_yield'):
            optimum_yldgapf = optimize_yldgapf_dyn_agyield(crop_no,
                                                   selected_grid_cells,
                                                   selected_soil_types,
                                                   opti_years,
                                                   plot_filename,
                                                   plot_rmse=True)
        elif (opti_method == 'aggregated_harvest'):
            optimum_yldgapf = optimize_yldgapf_dyn_agharvest(crop_no,
                                                   selected_grid_cells,
                                                   selected_soil_types,
                                                   opti_years)
        else:
            print "Optimization method does not exist! Check settings in the"\
                 +" script"
            sys.exit(2)
    else:
        optimum_yldgapf = 1.

#-------------------------------------------------------------------------------
# Do forward simulations for all available grid cells x soil type combinations

    if (forward_sims == True):

        print '\nWe use a yield gap factor of ', optimum_yldgapf, \
              'for our forward runs'

    #---------------------------------------------------------------------------
    # Select the grid cells to loop over for the forward simulations

		# we first read the list of all 'whole' grid cells contained in that
		# region NB: grid_list_tuples[0] is a list of (grid_cell_id,
		# arable_land_area) tuples, which are already sorted by decreasing
		# amount of arable land
        filename = folderpickle+'gridlistobject_all_r%s.pickle'%NUTS_no
        grid_list_tuples = pickle_load(open(filename,'rb'))

        # we select all grid cells
        selected_grid_cells = grid_list_tuples[0]
        print '\nWe have selected all', len(selected_grid_cells),'grid cells:',\
                  [g for g,a in selected_grid_cells]

    #---------------------------------------------------------------------------
    # Select the soil types to loop over for the forward runs

        # we first read the list of suitable soil types for our chosen crop 
        filename = folderpickle+'suitablesoilsobject_c%d.pickle'%crop_no
        suit_soils = pickle_load(open(filename,'rb')) 

        selected_soil_types = {}

        for grid in [g for g,a in selected_grid_cells]:

            # we read the entire list of soil types contained within the grid
            # cell
            filename = folderpickle+'soilobject_g%d.pickle'%grid
            soils = pickle_load(open(filename,'rb'))

            # We select all of them
            selected_soil_types[grid] = select_soils(grid, soils, suit_soils, 
                                                     method='all', n=nsoils)
            print 'We have selected all', len(selected_soil_types[grid]),\
                  'soil types:',\
                  [stu for smu, stu, w, data in selected_soil_types[grid]],\
                  'for grid', grid

    #---------------------------------------------------------------------------
    # we define the name of the file where we will store the results of the
    # forward simulations

        if (optimization == True):
            results_filename  = 'ForwardSim_Optimized_crop%i_region%s_%i-%i.dat'%(
                                 crop_no, NUTS_no, start_year, end_year)
        else:
            results_filename  = 'ForwardSim_Non-Optimized_crop%i_region%s_%i-%i.dat'%(
                                 crop_no, NUTS_no, start_year, end_year)

        # We perform the forward simulations:
        simulated_yields = perform_yield_sims(selected_grid_cells,
                                              selected_soil_types, 
                                              crop_no, campaign_years, 
                                              results_filename,
                                              yldgapf=optimum_yldgapf) 



#===============================================================================
#===============================================================================
# Function to optimize the yield gap factor within a NUTS region using the
# difference between the aggregated regional yield and the observed yield (ie. 1
# gap to optimize). This function iterates dynamically to find the optimum
# YLDGAPF.
def optimize_yldgapf_dyn_agyield(crop_no_, selected_grid_cells_,
                                 selected_soil_types_, opti_years_, plot_name,
                                 plot_rmse=True):
#===============================================================================

    # 1- we add a timestamp to time the function
    print '\nStarted dynamic optimization at timestamp:', datetime.utcnow()

    # aggregated yield method:
    
    # 2- we construct a 2D array with same dimensions as TSO_regional,
    # containing the observed yields
    row = [] # this list will become the row of the 2D array
    for y,year in enumerate(opti_years_):
        index_year = np.argmin(np.absolute(detrend[1]-year))
        row = row + [detrend[0][y]]
    OBS = np.tile(row, (5,1)) # repeats the list as a row 3 times, to get a 
                              # 2D array

    # 3- we calculate all the individual yields from the selected grid cells x
    # soils combinations

    # NB: we explore the range of yldgapf between 0.1 and 1.
    f0  = 0.
    f2  = 0.5
    f4  = 1.
    f_step  = 0.25 
    # Until the precision of the yield gap factor is good enough (i.e. < 0.02)
    # we loop over it. We do 12 iterations in total with this method.
    iter_no = 0
    RMSE_stored = list()
    while (f_step >= 0.02):
        iter_no = iter_no + 1
        # sub-method: looping over the yield gap factors

        # we build a range of 3 yield gap factors to explore one low bound, one
        # high bound, one in the middle
        f_step = (f4 - f0)/4.
        f1 = f0 + f_step
        f3 = f2 + f_step
        f_range = [f0, f1, f2, f3, f4]

        RES = [] # list in which we will store the yields of the combinations

        for grid in [g for g,a in selected_grid_cells_]:
 
            # Retrieve the weather data of one grid cell (all years are in one
            # file) 
            filename = folderpickle+'weatherobject_g%d.pickle'%grid
            weatherdata = WeatherDataProvider()
            weatherdata._load(filename)
                        
            # Retrieve the soil data of one grid cell (all possible soil types) 
            filename = folderpickle+'soilobject_g%d.pickle'%grid
            soil_iterator = pickle_load(open(filename,'rb'))

            for smu, stu_no, weight, soildata in selected_soil_types_[grid]:

                # TSO will store all the yields of one grid cell x soil 
                # combination, for all years and all 3 yldgapf values
                TSO = np.zeros((len(f_range), len(opti_years_)))
        
                for y, year in enumerate(opti_years_): 

                    # Retrieve yearly data 
                    filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    timerdata = pickle_load(open(filename,'rb'))
                    filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    cropdata = pickle_load(open(filename,'rb'))
                    filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'\
                                                    %(grid,crop_no_,year,stu_no)
                    sitedata = pickle_load(open(filename,'rb'))

                    for f,factor in enumerate(f_range):
            
                        cropdata['YLDGAPF']=factor
                       
                        # run WOFOST
                        wofost_object = Wofost71_WLP_FD(sitedata, timerdata,
                                                soildata, cropdata, weatherdata)
                        wofost_object.run_till_terminate()
        
                        # get the yield (in gDM.m-2) 
                        TSO[f,y] = wofost_object.get_variable('TWSO')

                RES = RES + [(grid, stu_no, weight, TSO)]

        # 4- we aggregate the yield into the regional one with array operations

        sum_weighted_yields = np.zeros((len(f_range), len(opti_years_)))
                                    # empty 2D array with same dimension as TSO
        sum_weights         = 0.
        for grid, stu_no, weight, TSO in RES:
            # adding weighted 2D-arrays in the empty array sum_weighted_yields
            sum_weighted_yields = sum_weighted_yields + weight*TSO 
            # computing the sum of the soil type area weights
            sum_weights         = sum_weights + weight 
        TSO_regional = sum_weighted_yields / sum_weights # weighted average

        # 5- we compute the (sim-obs) differences.
        DIFF = TSO_regional - OBS
        
        # Writing more output
        print '\nIteration %i'%iter_no
        print 'OPTIMIZATION MATRICES'
        print 'rows = yldgapf values; cols = optimization years'
        print 'opti years:', opti_years_, ', yldgapf values:', f_range
        print 'matrix of regional yields:'
        print ' '.join(str(f) for f in TSO_regional)
        print 'matrix of observed yields:'
        print ' '.join(str(f) for f in OBS)
        print 'matrix of sim-obs differences:'
        print ' '.join(str(f) for f in DIFF)

        # 6- we calculate the RMSE (root mean squared error) of the 3 yldgapf
        # The RMSE of each yldgapf is based on N obs-sim differences for the N
        # years looped over

        RMSE = np.zeros(len(f_range))
        for f,factor in enumerate(f_range):
            list_of_DIFF = []
            for y, year in enumerate(opti_years_):
                list_of_DIFF = list_of_DIFF + [DIFF[f,y]]
            RMSE[f] = np.sqrt(np.mean( [ math.pow(j,2) for j in
                                                           list_of_DIFF ] ))
        print 'Root Mean Square Error:'
        print ' '.join(str(f) for f in RMSE)

        RMSE_stored = RMSE_stored + [(f_range[1], RMSE[1]), (f_range[3], RMSE[3])]
        if (iter_no == 1):
            RMSE_stored = RMSE_stored + [(f_range[0], RMSE[0]), 
                                         (f_range[2], RMSE[2]),
                                         (f_range[4], RMSE[4])]

        # 7- We update the yldgapf range to explore for the next iteration. 
        # For this we do a linear interpolation of RMSE between the 3 yldgapf
        # explored here, and the next range to explore is the one having the
        # smallest interpolated RMSE

        index_new_center = RMSE.argmin()
        f0 = f_range[index_new_center-1]
        f2 = f_range[index_new_center]
        f4 = f_range[index_new_center+1]

	# when we are finished iterating on the yield gap factor range, we sort the
	# (RMSE, YLDGAPF) tuples by values of YLDGAPF
    RMSE_stored = sorted(RMSE_stored, key=operator_itemgetter(0))
    x,y = zip(*RMSE_stored)

	# when we are finished iterating on the yield gap factor range, we plot the
    # RMSE as a function of the yield gap factor
    if (plot_rmse == True):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        ax.plot(x, y, c='k', marker='o')
        ax.set_xlabel('yldgapf (-)')
        ax.set_ylabel('RMSE')
        fig.savefig('RMSE_'+plot_name+'_dyniter.png')

    # 8- when we are finished iterating on the yield gap factor range, we return
    # the optimum value. We look for the yldgapf with the lowest RMSE
    index_optimum   = RMSE.argmin()
    optimum_yldgapf = f_range[index_optimum] 

    print '\noptimum found:', optimum_yldgapf, '+/-', f_step

    # 9- we add a timestamp to time the function
    print 'Finished dynamic optimization at timestamp:', datetime.utcnow()

    # 10- we return the optimized YLDGAPF
    return optimum_yldgapf

#===============================================================================
# Function to optimize the yield gap factor within a NUTS region using the
# difference between the aggregated regional yield and the observed yield (ie. 1
# gap to optimize). This function explores a whole matrix of yldgapf to find the 
# optimum value 
def optimize_yldgapf_matrix_agyield(crop_no_, selected_grid_cells_,
                                   selected_soil_types_, opti_years_, plot_name,
                                   plot_rmse=True):
#===============================================================================

    # 1- we add a timestamp to time the function
    print '\nStarted matrix optimization at timestamp:', datetime.utcnow()

    # aggregated yield method:
    
	# We want a step of 0.02 on the yldgapf so we explore 51 values betwen 0.
	# and 1. NB: to increase the precision on the yldgapf, just increase
	# nb_f_values
    nb_f_values = 51
    f_range = np.linspace(0.,1.,nb_f_values)

    # 2- we construct a 2D array with same dimensions as TSO_regional,
    # containing the observed yields
    row = [] # this list will become the row of the 2D array
    for y,year in enumerate(opti_years_):
        index_year = np.argmin(np.absolute(detrend[1]-year))
        row = row + [detrend[0][y]]
    OBS = np.tile(row, (nb_f_values,1)) # repeats the list as a row n times,
                                        # to get a 2D array

    # 3- we calculate all the individual yields from the selected grid cells x
    # soils combinations

    # NB: we explore the range of yldgapf between 0.1 and 1.

    RES = []

    for grid in [g for g,a in selected_grid_cells_]:

        # Retrieve the weather data of one grid cell (all years are in one
        # file) 
        filename = folderpickle+'weatherobject_g%d.pickle'%grid
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
                    
        # Retrieve the soil data of one grid cell (all possible soil types) 
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soil_iterator = pickle_load(open(filename,'rb'))

        for smu, stu_no, weight, soildata in selected_soil_types_[grid]:

            # We calculate the (obs - sim) difference for each yldgapf
            TSO = np.zeros((nb_f_values, len(opti_years_)))
    
            for y, year in enumerate(opti_years_): 

                # Retrieve yearly data 
                filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'\
                                                       %(grid,crop_no_,year)
                timerdata = pickle_load(open(filename,'rb'))
                filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'\
                                                       %(grid,crop_no_,year)
                cropdata = pickle_load(open(filename,'rb'))
                filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'\
                                                %(grid,crop_no_,year,stu_no)
                sitedata = pickle_load(open(filename,'rb'))

                for f,factor in enumerate(f_range):
        
                    cropdata['YLDGAPF']=factor
                   
                    # run WOFOST
                    wofost_object = Wofost71_WLP_FD(sitedata, timerdata,
                                            soildata, cropdata, weatherdata)
                    wofost_object.run_till_terminate()
    
                    # get the yield (in gDM.m-2) 
                    TSO[f,y] = wofost_object.get_variable('TWSO')

            RES = RES + [(grid, stu_no, weight, TSO)]

    # 4- we aggregate the yield into the regional one with array operations

    sum_weighted_yields = np.zeros((nb_f_values, len(opti_years_)))
                                # empty 2D arraywith same dimension as DIFF
    sum_weights         = 0.
    for grid, stu_no, weight, TSO in RES:
        # adding weighted 2D-arrays in the empty array sum_weighted_yields
        sum_weighted_yields = sum_weighted_yields + weight*TSO 
        # computing the sum of the soil type area weights
        sum_weights         = sum_weights + weight 
    TSO_regional = sum_weighted_yields / sum_weights # weighted average

    # 5- we compute the (sim-obs) differences.
    DIFF = TSO_regional - OBS
    
    # Writing more output
    print 'OPTIMIZATION MATRICES'
    print 'rows = yldgapf values; cols = optimization years'
    print 'opti years:', opti_years_, ', yldgapf values:', f_range
    print 'matrix of regional yields:'
    print ' '.join(str(f) for f in TSO_regional)
    print 'matrix of observed yields:'
    print ' '.join(str(f) for f in OBS)
    print 'matrix of sim-obs differences:'
    print ' '.join(str(f) for f in DIFF)

    # 6- we calculate the RMSE (root mean squared error) of the 3 yldgapf
    # The RMSE of each yldgapf is based on N obs-sim differences for the N
    # years looped over

    RMSE = np.zeros(nb_f_values)
    for f,factor in enumerate(f_range):
        list_of_DIFF = []
        for y, year in enumerate(opti_years_):
            list_of_DIFF = list_of_DIFF + [DIFF[f,y]]
        RMSE[f] = np.sqrt(np.mean( [ math.pow(j,2) for j in
                                                       list_of_DIFF ] ))
    print 'Root Mean Square Error:'
    print ' '.join(str(f) for f in RMSE)

	# when we are finished iterating on the yield gap factor range, we plot the
    # RMSE as a function of the yield gap factor
    if (plot_rmse == True):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        ax.plot(f_range, RMSE, c='k', marker='o')
        ax.set_xlabel('yldgapf (-)')
        ax.set_ylabel('RMSE')
        fig.savefig('RMSE_'+plot_name+'_matrixiter.png')

    # 8- when we are finished iterating on the yield gap factor range, we return
    # the optimum value. We look for the yldgapf with the lowest RMSE
    index_optimum   = RMSE.argmin()
    optimum_yldgapf = f_range[index_optimum] 

    print '\noptimum found:', optimum_yldgapf, '+/- 0.02'

    # 9- we add a timestamp to time the function
    print 'Finished matrix optimization at timestamp:', datetime.utcnow()

    # 10- we return the optimized YLDGAPF
    return optimum_yldgapf

#===============================================================================
# Function to optimize the yield gap factor within a NUTS region
# this function iterates dynamically to find the optimum YLDGAPF
def optimize_yldgapf_dyn_individual(crop_no_, selected_grid_cells_, 
                                             selected_soil_types_, opti_years_):
#===============================================================================

    # 1- add a timestamp to time the function
    print 'Started dynamic optimization at timestamp:', datetime.utcnow()

    # individual method:
    FINAL_YLDGAPF = []
    for grid in [g for g,a in selected_grid_cells_]:

        # Retrieve the weather data of one grid cell (all years are in one file) 
        filename = folderpickle+'weatherobject_g%d.pickle'%grid
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
                    
        # Retrieve the soil data of one grid cell (all possible soil types) 
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soil_iterator = pickle_load(open(filename,'rb'))
    
        for smu, stu_no, a, soildata in selected_soil_types_[grid]:
       
            # Here we have selected one grid cell x soil combination
            # We will optimize the yield gap factor for that combination
            # We look for the optimum yldgapf between 0.1 and 1.
 
            lowestf  = 0.1
            highestf = 1.
            f_step   = highestf - lowestf
            # Until the precision of the yield gap factor is good enough
            # we loop over it. We do X iterations in total with this method.
            while (f_step >= 0.1):
                
                # we build a range of 3 yield gap factors to explore
                # one low bound, one high bound, one in the middle
                f_step = (highestf - lowestf)/2.
                middlef = lowestf + f_step
                f_range = [lowestf, middlef, highestf]
                RES = {}
                
                for y, year in enumerate(opti_years_): 
    
                    # Retrieve yearly data 
                    filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    timerdata = pickle_load(open(filename,'rb'))
                    filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    cropdata = pickle_load(open(filename,'rb'))
                    filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'\
                                                    %(grid,crop_no_,year,stu_no)
                    sitedata = pickle_load(open(filename,'rb'))

                    DIFF = np.array([0.]*len(f_range))

                    # We calculate the (obs - sim) difference for each yldgapf
                    for f,factor in enumerate(f_range):
        
                        cropdata['YLDGAPF']=factor
                   
                        # run WOFOST
                        wofost_object = Wofost71_WLP_FD(sitedata, timerdata,
                                                soildata, cropdata, weatherdata)
                        wofost_object.run_till_terminate()
    
                        # get the yield (in gDM.m-2) and the yield gap
                        TSO = wofost_object.get_variable('TWSO')
                        DIFF[f] = TSO - detrend[0][y]
				
                    RES[year] = DIFF
	            
                # Calculate the RMSE (root mean squared error) of the 3 yldgapf
                # the RMSE of each yldgapf is based on N obs-sim differences for
                # the N years looped over
                list_of_DIFF = []
                RMSE = np.array([0.]*len(f_range))
                for f,factor in enumerate(f_range):
                    for y, year in enumerate(opti_years_):
                        list_of_DIFF = list_of_DIFF + [RES[year][f]]
                    RMSE[f] = np.sqrt(np.mean( [ math.pow(j,2) for j in
                                                               list_of_DIFF ] ))
				
                # We update the yldgapf range to explore for the next iteration
                # For this we do a linear interpolation of RMSE between the 3 
                # yldgapf explored here, and the next range to explore is the
                # one having the smallest interpolated RMSE
                RMSE_midleft  = (RMSE[0] + RMSE[1]) / 2.
                RMSE_midright = (RMSE[1] + RMSE[2]) / 2.
                if (RMSE_midleft <= RMSE_midright):
                    lowestf = lowestf
                    highestf = middlef
                elif (RMSE_midleft > RMSE_midright):
                    lowestf = middlef
                    highestf = highestf

            # When we are finished iterating on the yield gap factor range, we
            # store the optimum value in a list
            index_optimum   = RMSE.argmin()
            optimum_yldgapf = f_range[index_optimum] 
            FINAL_YLDGAPF = FINAL_YLDGAPF + [(grid, stu_no, optimum_yldgapf)]

    # 3- add a timestamp to time the function
    print 'Finished dynamic optimization at timestamp:', datetime.utcnow()

    # 4- return the optimized YLDGAPF
    print FINAL_YLDGAPF
    return FINAL_YLDGAPF

#===============================================================================
# Function to optimize the yield gap factor within a NUTS region
# this function explores a whole matrix of factors to find the optimum YLDGAPF
def optimize_yldgapf_matrix_individual(crop_no_, selected_grid_cells_,
                                             selected_soil_types_, opti_years_):
#===============================================================================

    # 1- add a timestamp to time the function
    print 'Started matrix optimization at timestamp:', datetime.utcnow()

    # individual method:
    FINAL_YLDGAPF = []
    for grid in [g for g,a in selected_grid_cells_]:

        # Retrieve the weather data of one grid cell (all years are in one file) 
        filename = folderpickle+'weatherobject_g%d.pickle'%grid
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
                    
        # Retrieve the soil data of one grid cell (all possible soil types) 
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soil_iterator = pickle_load(open(filename,'rb'))
    
        for smu, stu_no, a, soildata in selected_soil_types_[grid]:
       
            # Here we have selected one grid cell x soil combination
            # We will optimize the yield gap factor for that combination
            # We look for the optimum yldgapf between 0.1 and 1.
 
			# We want a precision of 0.05 on the yldgapf so we explore 19
			# values betwen 0.1 and 1. NB: to increase the precision on the
			# yldgapf, just increase nb_f_values
            nb_f_values = 19
            f_range = np.linspace(0.1,1.,nb_f_values)

            # We calculate the (obs - sim) difference for each yldgapf
            DIFF = np.zeros((nb_f_values,len(opti_years_)))

            for y, year in enumerate(opti_years_): 

                # Retrieve yearly data 
                filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'\
                                                       %(grid,crop_no_,year)
                timerdata = pickle_load(open(filename,'rb'))
                filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'\
                                                       %(grid,crop_no_,year)
                cropdata = pickle_load(open(filename,'rb'))
                filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'\
                                                %(grid,crop_no_,year,stu_no)
                sitedata = pickle_load(open(filename,'rb'))

                for f,factor in enumerate(f_range):

                    cropdata['YLDGAPF']=factor
                   
                    # run WOFOST
                    wofost_object = Wofost71_WLP_FD(sitedata, timerdata,
                                                soildata, cropdata, weatherdata)
                    wofost_object.run_till_terminate()
    
                    # get the yield (in gDM.m-2) and the yield gap
                    TSO = wofost_object.get_variable('TWSO')
                    DIFF[f,y] = TSO - detrend[0][y]
				
            # Calculate the RMSE (root mean squared error) of the 3 yldgapf
            # the RMSE of each yldgapf is based on N obs-sim differences for
            # the N opti_years_ looped over
            RMSE = np.zeros(nb_f_values)
            for f,factor in enumerate(f_range):
                list_of_DIFF = []
                for y, year in enumerate(opti_years_):
                    list_of_DIFF = list_of_DIFF + [DIFF[f,y]]
                RMSE[f] = np.sqrt(np.mean( [ math.pow(j,2) for j in
                                                               list_of_DIFF ] ))
            # We look for the yldgapf with the lowest RMSE
            index_optimum   = RMSE.argmin()
            optimum_yldgapf = f_range[index_optimum] 

            # When we are finished retrieving the optimum yield gap factor, we
            # store it in a list
            FINAL_YLDGAPF = FINAL_YLDGAPF + [(grid, stu_no, optimum_yldgapf)]

    # 3- add a timestamp to time the function
    print 'Finished matrix optimization at timestamp:', datetime.utcnow()

    # 4- return the optimized YLDGAPF
    print FINAL_YLDGAPF
    return FINAL_YLDGAPF

#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sims(selected_grid_cells_, selected_soil_types_,
                       crop_no_, campaign_years_, Res_filename, yldgapf=1.):
#===============================================================================

    # 0- we open a file to write summary output in it
    if (os.path.isfile(os.path.join(currentdir, 'output_data', Res_filename))):
        os.remove(os.path.join(currentdir, 'output_data', Res_filename))
        print '\nDeleted old file %s in folder output_data/'%Res_filename
    Results = open(os.path.join(currentdir, 'output_data', Res_filename), 'w')
    # we write the header line:
    Results.write('YLDGAPF(-),  grid_no,  year,  stu_no,      weight(-), '\
                 +'TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), '\
                 +'TAGP(kgDM.ha-1)\n')

    print '\nStarted forward runs at timestamp:', datetime.utcnow()
    
    for grid in [g for g,a in selected_grid_cells_]:

        # Retrieve the weather data of one grid cell (all years are in one file) 
        filename = folderpickle+'weatherobject_g%d.pickle'%grid
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
                    
        # Retrieve the soil data of one grid cell 
        filename = folderpickle+'soilobject_g%d.pickle'%grid
        soil_iterator = pickle_load(open(filename,'rb'))

        for y, year in enumerate(campaign_years_): 

            # Retrieve calendar data of one year for one grid cell
            filename = folderpickle+'timerobject_g%d_c%d_y%d.pickle'%(grid,
                                                                  crop_no_,year)
            timerdata = pickle_load(open(filename,'rb'))
                            
            # Retrieve crop data of one year for one grid cell
            filename = folderpickle+'cropobject_g%d_c%d_y%d.pickle'%(grid,
                                                                  crop_no_,year)
            cropdata = pickle_load(open(filename,'rb'))

            # Set the yield gap factor
            cropdata['YLDGAPF'] = yldgapf

            for smu, soil_type, weight, soildata in selected_soil_types_[grid]:
                
                # Retrieve the site data of one soil, one year, one grid cell
                filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'%(
                                                   grid,crop_no_,year,soil_type)
                sitedata = pickle_load(open(filename,'rb'))

                # run WOFOST
                wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, 
                                                cropdata, weatherdata)
                wofost_object.run_till_terminate() #will stop the run when DVS=2
    
                # get major summary output variables for each run
                # total dry weight of - dead and alive - storage organs (kg/ha)
                TSO       = wofost_object.get_variable('TWSO')
                # total dry weight of - dead and alive - leaves (kg/ha) 
                TLV       = wofost_object.get_variable('TWLV')
                # total dry weight of - dead and alive - stems (kg/ha) 
                TST       = wofost_object.get_variable('TWST')
                # total dry weight of - dead and alive - roots (kg/ha) 
                TRT       = wofost_object.get_variable('TWRT')
                # maximum LAI
                MLAI      = wofost_object.get_variable('LAIMAX')
                # rooting depth (cm)
                RD        = wofost_object.get_variable('RD')
                # Total above ground dry matter (kg/ha)
                TAGP      = wofost_object.get_variable('TAGP')

                Results.write('%10.3f, %8i, %5i, %7i, %14.0f, %14.2f, %14.2f, '\
                              '%14.2f, %14.2f, %14.2f, %13.2f, %15.2f\n'
                              %(yldgapf, grid, year, soil_type, weight, TSO, 
                               TLV, TST, TRT, MLAI, RD, TAGP))
    Results.close()	

    print 'Finished forward runs at timestamp:', datetime.utcnow()

    return None

#===============================================================================
# Function to select a subset of soil types within a grid cell
def select_soils(grid_cell_id, soil_iterator_, suitable_soils, method='topn', n=3):
#===============================================================================

    # Rank soils by decreasing area
    sorted_soils = []
    for smu_no, area_smu, stu_no, percentage_stu, soildata in soil_iterator_:
        if stu_no not in suitable_soils: continue
        weight_factor  =  area_smu * percentage_stu
        sorted_soils   =  sorted_soils + [(smu_no, stu_no, weight_factor, soildata)]
    sorted_soils = sorted(sorted_soils, key=operator_itemgetter(2), reverse=True)
   
    # select a subset of soil types to loop over 
    # first option: we select the top n most present soils in the grid cell
    if   (method == 'topn'):
        subset_list   = sorted_soils[0:n]
    # second option: we select a random set of n soils within the grid cell
    elif (method == 'randomn'):
        try: # try to sample n random soil types:
            subset_list   = random_sample(sorted_soils,n)
        except: # if an error is raised ie. sample size bigger than population do
            subset_list   = sorted_soils
    # last option: we select all available soils in the grid cell
    else:
        subset_list   = sorted_soils
    
    return subset_list

#===============================================================================
# Function to select a subset of grid cells within a NUTS region
def select_grid_cells(list_of_tuples, method='topn', n=3):
#===============================================================================

	# NB: list_of_tuples is a list of (grid_cell_id, arable_land_area) tuples,
	# which are already sorted by decreasing amount of arable land
    
    # first option: we select the top n grid cells in terms of arable land area
    if (method == 'topn'):
        subset_list   = list_of_tuples[0:n]
    # second option: we select a random set of n grid cells
    elif (method == 'randomn'):
        try: # try to sample n random soil types:
            subset_list   = random_sample(list_of_tuples,n)
        except: # if an error is raised ie. sample size bigger than population do
            subset_list   = list_of_tuples
    # last option: we select all available grid cells of the region
    else:
        subset_list   = list_of_tuples

    return subset_list

#===============================================================================
# Return a list of consecutive years longer than n items
def find_consecutive_years(years, nyears):
#===============================================================================

    # Split the list of years where there are gaps
    years = map(int, years) # convert years to integers
    split_years = np.split(years, np.where(np.diff(years) > 1)[0]+1)

    # Return the most recent group of years that contains at least nyears items
    consecutive_years = np.array([])
    for subset in split_years[::-1]: # [::-1] reverses the array without 
                                     # modifying it permanently
        if (len(subset) >= nyears):
            consecutive_years = np.append(consecutive_years, subset)
            break
        else:
            pass

    # return the last nyears years of the most recent group of years
    return consecutive_years[-nyears:len(consecutive_years)]

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
    # and convert them from kg_humid_matter/ha to kg_dry_matter/ha 
    TARGET = np.array([0.]*nb_years)
    for j,year in enumerate(campaign_years):
        for i,region in enumerate(uncorrected_yields_dict['GEO']):
            if region.startswith(_NUTS_name[0:12]):
                if uncorrected_yields_dict['CROP_PRO'][i]==_crop_name:
                    if (uncorrected_yields_dict['TIME'][i]==str(int(year))):
                        if (uncorrected_yields_dict['STRUCPRO'][i]==
                                                      'Yields (100 kg/ha)'):
                            TARGET[j] = float(uncorrected_yields_dict['Value'][i])\
                                              *100.*_DM_content
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
        plt.close('all')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
        for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
            ax.scatter(campaign_years[mask], OBS[var], c='b')
       	    ax.plot(campaign_years,TREND[var],'r-')
       	    ax.set_ylabel('%s yield (gDM m-2)'%var, fontsize=14)
            ax.set_xlabel('time (year)', fontsize=14)
        fig.savefig('observed_yields.png')
        print 'the trend line is y=%.6fx+(%.6f)'%(z[0],z[1])
        plt.show()
    
    #print 'detrended dry matter yields:', OBS['DETRENDED']
    
    return OBS['DETRENDED'], campaign_years[mask]


#===============================================================================
# Function to open EUROSTAT csv files
def open_csv(inpath,filelist,convert_to_float=False):
#===============================================================================

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        print "\nOpening %s......"%(namefile)

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
    
        print "Dictionary created!"

    return Dict

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
