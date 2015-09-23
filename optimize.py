#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load # pickle_load is used in almost all my
                                        # methods which is why I import it
                                        # before def main():

#===============================================================================
# This script executes WOFOST runs for one NUTS region and optimizes its YLDGAPF
def main():
#===============================================================================
    from maries_toolbox import open_csv_EUROSTAT, detrend_obs, get_crop_name,\
                               define_opti_years, retrieve_crop_DM_content,\
                               select_cells, select_soils,\
                               fetch_EUROSTAT_NUTS_name
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, folderpickle, pcseoutput, detrend
#-------------------------------------------------------------------------------
# Define the settings of the run:
 
    # NUTS region and crop:
    NUTS_no       = 'NL1'
    crop_no       = 7        # CGMS crop number

    # yield gap factor optimization:
    optimization  = True     # if False: we assign a YLDGAPF = 1.
                             # if True: we optimize YLDGAPF
    opti_year     = 2006     # the year we want to match the yield obs. for
    selec_method  = 'topn'   # can be 'topn' or 'randomn' or 'all'
    ncells        = 10        # number of selected grid cells within a region
    nsoils        = 10        # number of selected soil types within a grid cell

    # forward crop growth simulations:
    forward_sims  = True
    start_year    = 2006     # start_year and end_year define the period of time
    end_year      = 2006     # for which we do forward simulations

#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'
    folderpickle  = '/Users/mariecombe/Documents/Work/Research_project_3/'\
                   +'pcse/pickled_CGMS_input_data/'
    pcseoutput    = '/Users/mariecombe/Documents/Work/Research_project_3/'\
                   +'pcse/pcse_individual_output/'

    # directories on capegrim:
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"
    #folderpickle  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    #pcseoutput    = '/Storage/CO2/mariecombe/pcse_individual_output/'

#-------------------------------------------------------------------------------
# Calculate key variables from the user input

    # we create an array containing the years for which we do forward runs:
    nb_years      = int(end_year - start_year + 1.)
    campaign_years = np.linspace(int(start_year),int(end_year),nb_years)

    #we fetch the EUROSTAT region name corresponding to the NUTS_no
    NUTS_name = fetch_EUROSTAT_NUTS_name(NUTS_no, EUROSTATdir)

    # we fetch the EUROSTAT crop name corresponding to the CGMS crop_no
    # 1-we retrieve both the CGMS and EUROSTAT names of a list of crops
    crop_name  = get_crop_name([crop_no])
    # 2-we select the EUROSTAT name of our specific crop 
    crop_name  = crop_name[crop_no][1]

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# If we optimize the yield gap factor:

    if (optimization == True): 

#-------------------------------------------------------------------------------
# Retrieve the crop dry matter content

        DM_content = retrieve_crop_DM_content(crop_no, NUTS_no)

#-------------------------------------------------------------------------------
# The optimization method and metric are now default options not given as
# choice for the user. THESE OPTIONS SHOULD NOT BE MODIFIED ANYMORE.

        opti_metric   = 'yield'      # can be 'yield' or 'harvest'

#-------------------------------------------------------------------------------
# Retrieve the observed EUROSTAT yield or harvest and remove its technological
# trend 

        if (opti_metric == 'harvest'):
            NUTS_filename = 'agri_harvest_NUTS1-2-3_1975-2014.csv'
        elif (opti_metric == 'yield'):
            NUTS_filename = 'agri_yields_NUTS1-2-3_1975-2014.csv'

        # we retrieve the harvest observed data:
        NUTS_data = open_csv_EUROSTAT(EUROSTATdir, [NUTS_filename],
                                     convert_to_float=True)

        # we detrend the observations:
        # NB: we must use the full 30 years of observed data, otherwise the 
        # technological trend might be a downward trend instead of an upward
        # trend! 
        detrend = detrend_obs(1975, 2014, NUTS_name, crop_name,
                              NUTS_data[NUTS_filename], DM_content, 2000,
                              obs_type=opti_metric, prod_fig=False)

#-------------------------------------------------------------------------------
# Determine on which years we will optimize the yield gap factor

        opti_years = define_opti_years(opti_year, detrend[1])

#-------------------------------------------------------------------------------
# Select the grid cells to loop over for the optimization

        # we select a subset of grid cells from the NUTS region
        selected_grid_cells = select_cells(NUTS_no, folderpickle, 
                              method=selec_method, n=ncells)

#-------------------------------------------------------------------------------
# Select the soil types to loop over for the optimization

        # We select a subset of soil types per selected grid cell
        selected_soil_types = select_soils(crop_no, selected_grid_cells,
                              folderpickle, method=selec_method, n=nsoils)


#-------------------------------------------------------------------------------
# Count the area used to calculate the harvest

        if (opti_metric == 'harvest'):

            areas = calc_cultivated_area_of_crop(selected_grid_cells, 
                                                 selected_soil_types)

        frac_crop = 0.36 # for now is hardcoded

#-------------------------------------------------------------------------------
# Perform optimization of the yield gap factor

        output_tagname = 'crop%i_region%s_%s_%igx%is_%s_%i-%i'%(crop_no, NUTS_no,
                          selec_method, ncells, nsoils, opti_metric, 
                          opti_years[0], opti_years[-1])

        optimum_yldgapf = optimize_regional_yldgapf_dyn(crop_no,
                                                           frac_crop,
                                                           selected_grid_cells,
                                                           selected_soil_types,
                                                           opti_years,
                                                           output_tagname,
                                                           obs_type=opti_metric,
                                                           plot_rmse=False)
    else:
        optimum_yldgapf = 1.

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# If we do forward simulations:

    if (forward_sims == True):

#-------------------------------------------------------------------------------
# Select all available grid cells to loop over for the forward simulations

        # we select a subset of grid cells from the NUTS region
        selected_grid_cells = select_cells(NUTS_no, folderpickle, method='all')

#-------------------------------------------------------------------------------
# Select all the available soil types to loop over for the forward runs

        # We select a subset of soil types per selected grid cell
        selected_soil_types = select_soils(crop_no, selected_grid_cells,
                              folderpickle, method='all')

#-------------------------------------------------------------------------------
# Define the name of the file where we will store the results of the forward
# runs

        if (optimization == True):
            results_filename  = 'ForwardSim_Opt_crop%i_region%s_%i-%i.dat'%(
                                 crop_no, NUTS_no, start_year, end_year)
        else:
            results_filename  = 'ForwardSim_Non-Opt_crop%i_region%s_%i-%i.dat'%(
                                 crop_no, NUTS_no, start_year, end_year)

#-------------------------------------------------------------------------------
# Perform the forward simulations:

        print '\nWe use a YLDGAPF of %.2f for our forward runs'%optimum_yldgapf
        simulated_yields = perform_yield_sims(selected_grid_cells,
                                              selected_soil_types, 
                                              crop_no, campaign_years, 
                                              results_filename,
                                              yldgapf=optimum_yldgapf) 



#===============================================================================
# Function to optimize the regional yield gap factor using the difference
# between the regional simulated and the observed harvest or yield (ie. 1 gap to
# optimize per NUTS region). This function iterates dynamically to find the
# optimum YLDGAPF.
def optimize_regional_yldgapf_dyn(crop_no_, frac_crop, selected_grid_cells_,
           selected_soil_types_, opti_years_, output_tagname_, obs_type='yield',
           plot_rmse=True):
#===============================================================================

    import math
    from cPickle import dump as pickle_dump
    from datetime import datetime
    from operator import itemgetter as operator_itemgetter
    from matplotlib import pyplot as plt
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider

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

        for grid, arable_land in selected_grid_cells_:
 
            frac_arable = arable_land / 625000000.

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
        
                        # get the yield (in kgDM.ha-1) 
                        TSO[f,y] = wofost_object.get_variable('TWSO')

                RES = RES + [(grid, stu_no, weight*frac_arable*frac_crop, TSO)]

        # 4- we aggregate the yield or harvest into the regional one with array
        # operations

        sum_weighted_vals = np.zeros((len(f_range), len(opti_years_)))
                                    # empty 2D array with same dimension as TSO
        sum_weights       = 0.
        for grid, stu_no, weight, TSO in RES:
            # adding weighted 2D-arrays in the empty array sum_weighted_yields
            # NB: variable 'weight' is actually the cultivated area in m2
            sum_weighted_vals   = sum_weighted_vals + (weight/10000.)*TSO 
            # computing the total sum of the cultivated area in ha 
            sum_weights         = sum_weights       + (weight/10000.) 

        if (obs_type == 'harvest'):
            TSO_regional = sum_weighted_vals / 1000000. # sum of the individual 
                                                        # harvests in 1000 tDM
        elif (obs_type == 'yield'):
            TSO_regional = sum_weighted_vals / sum_weights # weighted average of 
                                                        # all yields in kgDM/ha

        # 5- we compute the (sim-obs) differences.
        DIFF = TSO_regional - OBS
        assert (TSO_regional[-1] > OBS[-1]), "Observed yield too high! The DM '+\
               'content is wrong. Check it again."
        
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
        if (obs_type == 'harvest'):
            print ' '.join(str(f) for f in DIFF)
            print 'cultivated area of region: %.2f (in 1000 ha)'%\
                                                             (sum_weights/1000.)

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

        # We store the value of the RMSE for plotting purposes
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
    RMSE_stored  = sorted(RMSE_stored, key=operator_itemgetter(0))
    pickle_dump(RMSE_stored, open(os.path.join(currentdir,'pcse_summary_output',
                                   'RMSE_' + output_tagname_ + '.pickle'),'wb'))       

	# when we are finished iterating on the yield gap factor range, we plot the
    # RMSE as a function of the yield gap factor
    if (plot_rmse == True):
        x,y = zip(*RMSE_stored)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        ax.plot(x, y, c='k', marker='o')
        ax.set_xlabel('yldgapf (-)')
        ax.set_ylabel('RMSE')
        fig.savefig('RMSE_'+output_tagname_+'_dyniter.png')

    # 8- when we are finished iterating on the yield gap factor range, we return
    # the optimum value. We look for the yldgapf with the lowest RMSE
    index_optimum   = RMSE.argmin()
    optimum_yldgapf = f_range[index_optimum] 

    print '\noptimum found: %.2f +/- %.2f'%(optimum_yldgapf, f_step)

    # 9- we add a timestamp to time the function
    print 'Finished dynamic optimization at timestamp:', datetime.utcnow()

    # 10- we return the optimized YLDGAPF
    return optimum_yldgapf

#===============================================================================
# Function to optimize the yield gap factor within a NUTS region using the
# difference between the aggregated regional yield and the observed yield (ie. 1
# gap to optimize). This function iterates dynamically to find the optimum
# YLDGAPF.
def optimize_yldgapf_dyn_agyield(crop_no_, selected_grid_cells_,
            selected_soil_types_, opti_years_, output_tagname_, plot_rmse=True):
#===============================================================================

    import math
    from cPickle import dump as pickle_dump
    from datetime import datetime
    from operator import itemgetter as operator_itemgetter
    from matplotlib import pyplot as plt
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider

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
    RMSE_stored  = sorted(RMSE_stored, key=operator_itemgetter(0))
    pickle_dump(RMSE_stored, open(os.path.join(currentdir,'pcse_summary_output',
                                   'RMSE_' + output_tagname_ + '.pickle'),'wb'))       

	# when we are finished iterating on the yield gap factor range, we plot the
    # RMSE as a function of the yield gap factor
    if (plot_rmse == True):
        x,y = zip(*RMSE_stored)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        ax.plot(x, y, c='k', marker='o')
        ax.set_xlabel('yldgapf (-)')
        ax.set_ylabel('RMSE')
        fig.savefig('RMSE_'+output_tagname_+'_dyniter.png')

    # 8- when we are finished iterating on the yield gap factor range, we return
    # the optimum value. We look for the yldgapf with the lowest RMSE
    index_optimum   = RMSE.argmin()
    optimum_yldgapf = f_range[index_optimum] 

    print '\noptimum found: %.2f +/- %.2f'%(optimum_yldgapf, f_step)

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

    import math
    from datetime import datetime
    from matplotlib import pyplot as plt
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider

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
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sims(selected_grid_cells_, selected_soil_types_,
                       crop_no_, campaign_years_, Res_filename, yldgapf=1.):
#===============================================================================

    from datetime import datetime
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider

    # 0- we open a file to write summary output in it
    if (os.path.isfile(os.path.join(currentdir, 'pcse_summary_output',   
                                                                Res_filename))):
        os.remove(os.path.join(currentdir, 'pcse_summary_output', Res_filename))
        print '\nDeleted old file %s in folder pcse_summary_output/'%Res_filename
    Results = open(os.path.join(currentdir, 'pcse_summary_output', Res_filename), 
                                                                             'w')
    # we write the header line:
    Results.write('YLDGAPF(-),  grid_no,  year,  stu_no, arable_area(ha), stu_area(ha), '\
                 +'TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), '\
                 +'TAGP(kgDM.ha-1)\n')

    print '\nStarted forward runs at timestamp:', datetime.utcnow()
    
    for grid, arable_area in selected_grid_cells_:

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

            for smu, soil_type, stu_area, soildata in selected_soil_types_[grid]:
                
                # Retrieve the site data of one soil, one year, one grid cell
                filename = folderpickle+'siteobject_g%d_c%d_y%d_s%d.pickle'%(
                                                   grid,crop_no_,year,soil_type)
                sitedata = pickle_load(open(filename,'rb'))

                # run WOFOST
                wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, 
                                                cropdata, weatherdata)
                wofost_object.run_till_terminate() #will stop the run when DVS=2

                # get time series of the output and take the selected variables
                #output = wofost_object.get_output()
                #varnames = ["day","GASS","MRES"]
                #timeseries = dict()
                #for var in varnames:
                #    timeseries[var] = [t[var] for t in output]
                wofost_object.store_to_file( pcseoutput +\
                                            "pcse_output_c%s_g%s_s%s_y%i.csv"\
                                            %(crop_no_,grid,soil_type,year))

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

                Results.write('%10.3f, %8i, %5i, %7i, %15.2f, %12.5f, %14.2f, '\
                              '%14.2f, %14.2f, %14.2f, %14.2f, %13.2f, %15.2f\n'
                           %(yldgapf, grid, year, soil_type, arable_area/10000.,
                           stu_area/10000., TSO, TLV, TST, TRT, MLAI, RD, TAGP))
    Results.close()	

    print 'Finished forward runs at timestamp:', datetime.utcnow()

    return None

#===============================================================================
# function to calculate the simulated cultivated area of a NUTS region
def calc_cultivated_area_of_crop(selected_grid_cells_, selected_soil_types_):
#===============================================================================

    from datetime import datetime
    from operator import itemgetter as operator_itemgetter

    # 1- we add a timestamp to time the function
    print '\nStarted counting area at timestamp:', datetime.utcnow()

    # 3- we calculate all the individual yields from the selected grid cells x
    # soils combinations

    grid_land   = 0.
    arable_land = 0.
    soil_land   = 0.

    for grid, arable_area in selected_grid_cells_:
 
        grid_land   = grid_land   + 625000000.  # area of a grid cell is in m2
        arable_land = arable_land + arable_area # grid arable land area is in m2
        
        for smu, stu_no, stu_area, soildata in selected_soil_types_[grid]:

            soil_land = soil_land + stu_area # soil type area is in m2
    
    total_grid   = grid_land   / 10000000. # in 1000 ha
    total_soils  = soil_land   / 10000000. # in 1000 ha
    total_arable = arable_land / 10000000. # in 1000 ha

    print '\nGrid area total: %.2f (in 1000 ha)'%total_grid
    print '\nArable land total: %.2f (in 1000 ha)'%total_arable
    print '\nSoil types area total: %.2f (in 1000 ha)'%total_soils

    # 9- we add a timestamp to time the function
    print 'Finished counting area at timestamp:', datetime.utcnow()

    # 10- we return the land areas
    return total_grid, total_arable, total_soils

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
