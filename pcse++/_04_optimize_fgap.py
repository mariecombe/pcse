#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load # pickle_load is used in almost all my
                                        # methods which is why I import it
                                        # before def main():

#===============================================================================
# This script optimizes the WOFOST fgap for a number of NUTS regions
def main():
#===============================================================================
    """
    This script can be used to optimize the yield gap factor of a specific
    region and crop species. It will return a dictionnary of fgap, with NUTS
    regions as keys

    """
#-------------------------------------------------------------------------------
    global currentdir, EUROSTATdir, inputdir, ecmwfdir, crop_dict,\
           NUTS_regions, years, crop, crop_name, year, observed_data,\
           selec_method, ncells, nsoils, weather, opti_metric
#-------------------------------------------------------------------------------
# Temporarily add parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================
 
    # yield gap factor optimization:
    selec_method  = 'topn'    # can be 'topn' or 'randomn' or 'all'
    ncells        = 1        # number of selected grid cells within a region
    nsoils        = 1        # number of selected soil types within a grid cell
    weather       = 'ECMWF'   # weather data to use for the optimization
                              # can be 'CGMS' or 'ECMWF'

    # input directory paths
    inputdir = '/Users/mariecombe/Documents/Work/Research_project_3/'+\
               'model_input_data/'
    #inputdir = '/Users/mariecombe/mnt/promise/CO2/marie/'

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    currentdir    = os.getcwd()
    EUROSTATdir   = 'EUROSTATobs/'
    ecmwfdir      = inputdir + 'CABO_weather_ECMWF/'
#-------------------------------------------------------------------------------
# we retrieve the crops, regions, and years to loop over:
    try:
        crop_dict    = pickle_load(open('../tmp/selected_crops.pickle','rb'))
        years        = pickle_load(open('../tmp/selected_years.pickle','rb'))
        NUTS_regions = pickle_load(open('../tmp/selected_NUTS_regions.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of crops, regions and '+\
              'years to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# we retrieve the EUROSTAT pre-processed yield observations:
    try:
        filename1 = inputdir + 'preprocessed_yields.pickle'
        filename2 = inputdir + 'preprocessed_harvests.pickle'
        yields_dict   = pickle_load(open(filename1,'rb'))
        harvests_dict = pickle_load(open(filename2,'rb'))
    except IOError:
        print '\nYou have not preprocessed the EUROSTAT observations'
        print 'Run the script 03_preprocess_obs.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# we initialize the dictionary of fgap:
    opti_fgap = dict()
#-------------------------------------------------------------------------------
# The optimization method and metric are now default options not given as
# choice for the user. THESE OPTIONS SHOULD NOT BE MODIFIED ANYMORE.
    opti_metric = 'yield'      # can be 'yield' or 'harvest'
#-------------------------------------------------------------------------------
# Retrieve the observed EUROSTAT yield or harvest and remove its technological
# trend 
    if (opti_metric == 'harvest'):
        observed_data = harvests_dict
    elif (opti_metric == 'yield'):
        observed_data = yields_dict
#-------------------------------------------------------------------------------
# LOOP OVER CROPS:
#-------------------------------------------------------------------------------
    for crop in sorted(crop_dict.keys()):
        crop_name  = crop_dict[crop][1]
        print '\nCrop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name)
        print '================================================'
        opti_fgap[crop] = dict()
#-------------------------------------------------------------------------------
# LOOP OVER YEARS:
#-------------------------------------------------------------------------------
        for year in years:
            print 'Year ', year
            print '================================================'
            opti_fgap[crop][year] = dict()
#-------------------------------------------------------------------------------
# LOOP OVER THE NUTS REGIONS:
#-------------------------------------------------------------------------------
            for NUTS_no in sorted(NUTS_regions):
                opti_fgap[crop][year][NUTS_no] = optimize_fgap(NUTS_no)
            print opti_fgap[crop][year]
            sys.exit()
#-------------------------------------------------------------------------------
	    		# we get the fgap of this NUTS region for all grid cells
	    		# entirely contained in it
#                for grid_no in whole_cells_in_region:
#                    opti_fgap[crop][year][grid_no] = optimum

#-------------------------------------------------------------------------------
# LOOP OVER THE CULTIVATED GRID CELLS OF THIS CROP x YEAR COMBINATION:
#-------------------------------------------------------------------------------
            # loop over all cultivated grid cells:
            for grid_no in cultigridcells:
                # if we have not assigned a yield gap factor yet to the cell
                # i.e. if this is a grid cell shared between NUTS regions
                if grid_no not in opti_fgap[crop][opti_year].keys():
                    # we average the fgap between neighbourgh regions
                    closest_cells = get_neighbor_cells(grid_no)
                    list_closest_fgap = list()
                    for cell_no in closest_cells:
                        try:
                            list_closest_fgap += opti_fgap[crop][opti_year][cell_no]
                        except KeyError: # except cell_no is not cultivated
                            pass
                    optimum = mean(list_closest_fgap)
                    opti_fgap[crop][opti_year][grid_no] = optimum

    pickle_dump(open('opti_fgap_dict.pickle','wb'))

#===============================================================================
def optimize_fgap(NUTS_no):
#===============================================================================
    from maries_toolbox import define_opti_years,\
                               select_cells, select_soils, get_EUR_frac_crop
#-------------------------------------------------------------------------------
    NUTS_name = NUTS_regions[NUTS_no][1]
    print "\nRegion: %s / %s"%(NUTS_no, NUTS_name)
#-------------------------------------------------------------------------------
    # NB: we do NOT detrend the yields anymore, since fgap is not
    # supposed to be representative of multi-annual gap
    detrend = observed_data[crop][NUTS_no]
    #print 'Raw yields observations:'
    #print detrend[0]
    #print detrend[1]
#-------------------------------------------------------------------------------
# if there were no reported yields at all, or no reported yield for the year of
# interest, we skip that region
    if (len(detrend[1])==0) or (year not in detrend[1]):
        print 'No reported yield, optimum cannot be compiled'
        return np.nan
#-------------------------------------------------------------------------------
    # Determine on which years we will optimize the yield gap 
    # factor
    #opti_years = define_opti_years(year, detrend[1])
#-------------------------------------------------------------------------------
    # Select the grid cells to loop over for the optimization
    selected_grid_cells = select_cells(NUTS_no, crop_dict[crop][0], 
                                       year, inputdir+'CGMS/', 
                                       method=selec_method, n=ncells, 
                                       select_from='cultivated')
#-------------------------------------------------------------------------------
# if there were no cells cultivated that year, we skip that region
    if (selected_grid_cells == None):
        print 'No cultivated grid cells, optimum cannot be compiled'
        return np.nan
#-------------------------------------------------------------------------------
	# Select the soil types to loop over for the optimization
    selected_soil_types = select_soils(crop_dict[crop][0],
                                       [g for g,a in selected_grid_cells],
                                       inputdir+'CGMS/',
                                       method=selec_method, 
                                       n=nsoils)
#-------------------------------------------------------------------------------
    # we retrieve the EUROSTAT fraction of arable land 
    # cultivated into the crop
    frac_crop_over_years = get_EUR_frac_crop(crop_name, 
                                         NUTS_name, EUROSTATdir)
 
    # the fraction of cultivation for year X:
    index_year = np.abs(np.array(frac_crop_over_years[1]) - 
                                               float(year))
    index_year = index_year.argmin()
    frac_crop  = frac_crop_over_years[0][index_year]
    print 'We use a cultivated fraction of %.2f'%frac_crop
#-------------------------------------------------------------------------------
    # Count the area used to calculate the harvest
    if (opti_metric == 'harvest'):
        areas = calc_cultivated_area_of_crop(selected_grid_cells, 
                                             selected_soil_types)
#-------------------------------------------------------------------------------
    # Perform optimization of the yield gap factor
    output_tagname = 'crop%i_region%s_%s_%igx%is_%s_%i'%(
                      crop_dict[crop][0], NUTS_no, selec_method, ncells, 
                      nsoils, opti_metric, year)
 
    optimum = optimize_regional_yldgapf_dyn(detrend,crop_dict[crop][0],
                                            frac_crop,
                                            selected_grid_cells,
                                            selected_soil_types,
                                            inputdir+'CGMS/',
                                            [year], output_tagname,
                                            obs_type=opti_metric,
                                            plot_rmse=False)
    return optimum

#===============================================================================
# Function to optimize the regional yield gap factor using the difference
# between the regional simulated and the observed harvest or yield (ie. 1 gap to
# optimize per NUTS region). This function iterates dynamically to find the
# optimum YLDGAPF.
def optimize_regional_yldgapf_dyn(detrend, crop_no_, frac_crop, selected_grid_cells_,
 selected_soil_types_, inputdir, opti_years_, output_tagname_, obs_type='yield',
           plot_rmse=True):
#===============================================================================

    import math
    from cPickle import dump as pickle_dump
    from datetime import datetime
    from operator import itemgetter as operator_itemgetter
    from matplotlib import pyplot as plt
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

    # 1- we add a timestamp to time the function
    start_timestamp = datetime.utcnow()

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
        print f_step
        iter_no = iter_no + 1
        # sub-method: looping over the yield gap factors

        # we build a range of 3 yield gap factors to explore one low bound, one
        # high bound, one in the middle
        f_step = (f4 - f0)/4.
        f1 = f0 + f_step
        f3 = f2 + f_step
        f_range = [f0, f1, f2, f3, f4]

        RES = [] # list in which we will store the yields of the combinations

        counter=0
        for grid, arable_land in selected_grid_cells_:
 
            frac_arable = arable_land / 625000000.

            # Retrieve the weather data of one grid cell (all years are in one
            # file) 
            if (weather == 'CGMS'):
                filename = inputdir+'weatherobject_g%d.pickle'%grid
                weatherdata = WeatherDataProvider()
                weatherdata._load(filename)
            if (weather == 'ECMWF'):
                weatherdata = CABOWeatherDataProvider('%i'%grid,fpath=ecmwfdir)
                        
            # Retrieve the soil data of one grid cell (all possible soil types) 
            filename = inputdir+'soilobject_g%d.pickle'%grid
            soil_iterator = pickle_load(open(filename,'rb'))

            for smu, stu_no, weight, soildata in selected_soil_types_[grid]:

                # TSO will store all the yields of one grid cell x soil 
                # combination, for all years and all 3 yldgapf values
                TSO = np.zeros((len(f_range), len(opti_years_)))

                counter +=1
        
                for y, year in enumerate(opti_years_): 

                    # Retrieve yearly data 
                    filename = inputdir+'timerobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    timerdata = pickle_load(open(filename,'rb'))
                    filename = inputdir+'cropobject_g%d_c%d_y%d.pickle'\
                                                           %(grid,crop_no_,year)
                    cropdata = pickle_load(open(filename,'rb'))
                    filename = inputdir+'siteobject_g%d_c%d_y%d_s%d.pickle'\
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

                    print grid, smu, year, counter, TSO[-1]
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
        #assert (TSO_regional[-1][0] > 0.), 'Problem with the potential WOFOST simulation.'
        #assert (TSO_regional[-1] > OBS[-1]), 'Observed yield too high! The DM '+\
        #       'content is wrong. Check it again.'
        if (TSO_regional[-1] <= OBS[-1]):
            print '\nObserved yield larger than simulated yield. Optimum found: 1'
            return 1.
        
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
        try:
            f0 = f_range[index_new_center-1]
            f2 = f_range[index_new_center]
            f4 = f_range[index_new_center+1]
        except IndexError:
            # if the optimum is close to 1:
            if index_new_center == len(f_range)-1:
                f0 = f_range[index_new_center-2]
                f2 = f_range[index_new_center-1]
                f4 = f_range[index_new_center]
            # if the optimum is close to 0:
            elif index_new_center == 0:
                f0 = f_range[index_new_center]
                f2 = f_range[index_new_center+1]
                f4 = f_range[index_new_center+2]

	# when we are finished iterating on the yield gap factor range, we sort the
	# (RMSE, YLDGAPF) tuples by values of YLDGAPF
    #RMSE_stored  = sorted(RMSE_stored, key=operator_itemgetter(0))
    #pickle_dump(RMSE_stored, open(os.path.join(currentdir,'pcse_summary_output',
    #                               'RMSE_' + output_tagname_ + '.pickle'),'wb'))       

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
    end_timestamp = datetime.utcnow()
    print 'Finished dynamic optimization at timestamp:', end_timestamp-start_timestamp

    # 10- we return the optimized YLDGAPF
    return optimum_yldgapf


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
