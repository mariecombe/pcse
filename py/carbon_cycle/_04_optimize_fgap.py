#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys
from os import path
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import os,glob
from py.tools.initexit import start_logger, parse_options
from py.carbon_cycle._01_select_crops_n_regions import select_crops_regions
import py.tools.rc as rc
import logging as mylogger
import numpy as np

from datetime import datetime
from cPickle import load as pickle_load
from cPickle import dump as pickle_dump

#===============================================================================
# This script optimizes the WOFOST fgap for a number of NUTS regions
def optimize():
#===============================================================================
    """
    This script can be used to optimize the yield gap factor of a specific
    region and crop species. It will return a dictionnary of fgap, with NUTS
    regions as keys

    """
#-------------------------------------------------------------------------------
# all the directory names, lists of selected crop/NUTS/years, pre-processed
# observations, and user input variables are converted to global variables
    global inputdir, cwdir, CGMSdir, EUROSTATdir, ecmwfdir, outputdir, \
           crop_dict, NUTS_regions, years, crop, crop_name, year, observed_data,\
           selec_method, ncells, nsoils, weather, opti_metric, force_optimization,\
           CGMSgrid, CGMSsoil, CGMScropmask, CGMScrop, CGMStimer, CGMSsite, crop_no
#-------------------------------------------------------------------------------

    sys.path.insert(0, "..")

    _ = start_logger(level=mylogger.INFO)

    opts, args = parse_options()

    # First message from logger
    mylogger.info('Python Code has started')
    mylogger.info('Passed arguments:')
    for arg,val in args.iteritems():
        mylogger.info('         %s : %s'%(arg,val) )

    rcfilename = args['rc']
    rcF = rc.read(rcfilename)
    crops = [ rcF['crop'] ]
    years = [int(rcF['year'])]
    outputdir = rcF['dir.output']
    outputdir = os.path.join(outputdir,'ygf')
    inputdir = rcF['dir.wofost.input']
    par_process = (rcF['opt.wofost.parallel'] in ['True','TRUE','true','T'])
    nsoils = int(rcF['opt.wofost.nsoils'])
    ncells = int(rcF['opt.wofost.ncells'])
    weather = rcF['opt.wofost.weather']
    selec_method = rcF['opt.wofost.method']

# ================================= USER INPUT =================================
 
    # yield gap factor optimization:
    force_optimization = False # decides if we recalculate the optimum fgap, in
                               # case the results file already exists

    # input directory paths
    #inputdir = '/Users/mariecombe/Documents/Work/Research_project_3/'+\
    #           'model_input_data/'
    #inputdir = os.path.join('/projects/0/ctdas/input/wofost')

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    currentdir    = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
    EUROSTATdir = os.path.join(inputdir,'EUROSTATobs')
    ecmwfdir    = os.path.join(inputdir,'CABO_weather_ECMWF')
#-------------------------------------------------------------------------------
# create an output directory if it doesn't exist
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
#-------------------------------------------------------------------------------
# we retrieve the crops, regions, and years to loop over:
    NUTS_regions,crop_dict = select_crops_regions(crops,EUROSTATdir)

    if rcF.has_key('nuts.limit'):
        nutslimit = rcF['nuts.limit'].split(',')
        NUTS_selected = []
        for nuts in nutslimit:
            NUTS_selected.extend([ n for n in sorted(NUTS_regions) if nuts.strip() in n] ) 
        mylogger.info('NUTS list limited to length %d based on nuts.limit (%s)'%(len(NUTS_selected),sorted(NUTS_selected)))
        NUTS_regions = NUTS_selected

#-------------------------------------------------------------------------------
# we retrieve the EUROSTAT pre-processed yield observations:
    try:
        filename1 = os.path.join(EUROSTATdir, 'preprocessed_yields.pickle')
        filename2 = os.path.join(EUROSTATdir, 'preprocessed_harvests.pickle')
        yields_dict   = pickle_load(open(filename1,'rb'))
        harvests_dict = pickle_load(open(filename2,'rb'))
    except IOError:
        mylogger.error( 'You have not preprocessed the EUROSTAT observations' )
        mylogger.error( 'Run the script 03_preprocess_obs.py first!' )
        sys.exit() 
#-------------------------------------------------------------------------------
# open the pickle files containing the CGMS input data
    CGMSgrid  = pickle_load(open(os.path.join(CGMSdir,'CGMSgrid.pickle'),'rb'))
    CGMSsoil  = pickle_load(open(os.path.join(CGMSdir,'CGMSsoil.pickle'),'rb'))
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
#-------------------------------------------------------------------------------
# I - OPTIMIZE THE YIELD GAP FACTOR FOR THE CLIMATOLOGICAL PERIOD (2000-2010):
#-------------------------------------------------------------------------------
    for year in years:
        mylogger.info( 'Year %s'% year)
        mylogger.info( '================================================' )
#-------------------------------------------------------------------------------
        for crop in sorted(crop_dict.keys()):
            crop_name  = crop_dict[crop][1]
            mylogger.info( 'Crop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name) )
            if crop_dict[crop][0]==5:
                crop_no = 1
                mylogger.info( 'Modified the internal crop_no from 5 to 1')
            elif crop_dict[crop][0]==13:
                crop_no = 3
                mylogger.info( 'Modified the internal crop_no from 13 to 3')
            elif crop_dict[crop][0]==12:
                crop_no = 2
                mylogger.info( 'Modified the internal crop_no from 12 to 2')
            else:
                crop_no = crop_dict[crop][0]

            mylogger.info( '================================================' )

            if force_optimization == True:
                filelist = [f for f in os.listdir(outputdir)]
                for f in filelist:
                    os.remove(os.path.join(outputdir,f))
                mylogger.info( "We force the optimization: output directory just got emptied" )

            # crop mask (do NOT use the crossover crop_no here)
            filename = os.path.join(CGMSdir, 'cropdata_objects',
                                       'cropmask_c%i.pickle'%crop_dict[crop][0])
            CGMScropmask = pickle_load(open(filename, 'rb'))

            # Other crop parameters (where needed use the crossover crop_no)

            CGMScropmask = pickle_load(open(filename, 'rb'))
            filename = os.path.join(CGMSdir, 'cropdata_objects',
                             'CGMScrop_%i_c%i.pickle'%(year,crop_no))
            CGMScrop = pickle_load(open(filename, 'rb'))
            mylogger.info('Successfully loaded the CGMS crop pickle files: cropmask_c%i.pickle and CGMScrop_%i_c%i.pickle'%(crop_dict[crop][0],year,crop_no) )

            # crop calendars (use crossover crop_no, plus 2 exceptions)
            if crop_dict[crop][0]==5: # spring wheat uses spring barley calendars
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                'CGMStimer_%i_c%i.pickle'%(year,3))
            elif crop_dict[crop][0]==13: # winter barley uses winter wheat calendars
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                'CGMStimer_%i_c%i.pickle'%(year,1))
            else:
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                'CGMStimer_%i_c%i.pickle'%(year,crop_no))

            CGMStimer = pickle_load(open(filename, 'rb'))
            mylogger.info('Successfully loaded the CGMS timer pickle file (%s)'%filename)

            filename = os.path.join(CGMSdir, 'sitedata_objects',
                             'CGMSsite_%i_c%i.pickle'%(year,crop_no))
            CGMSsite = pickle_load(open(filename, 'rb'))
            #print 'Successfully loaded the CGMS site pickle file'
            mylogger.info('Successfully loaded the CGMS site pickle file (%s)'%filename)

#-------------------------------------------------------------------------------
# loop over NUTS regions - this is the parallelized part of the script -
#-------------------------------------------------------------------------------
            # we add a timestamp to time the optimization
            start_timestamp = datetime.utcnow()
         
            # if we do a serial iteration:
            if (not par_process):
                for NUTS_no in sorted(NUTS_regions):  
                    #mylogger.warning("Only testing 32 NUTS regions for now!")
                    optimize_fgap(NUTS_no)
         
            # if we do a parallelization
            if (par_process):
                import multiprocessing
                # get number of cpus available to job
                try:
                    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])/2
                    mylogger.info( "Success reading parallel env %d" % ncpus )
                except KeyError:
                    ncpus = multiprocessing.cpu_count()/2
                    mylogger.info( "Success obtaining processor count %d" % ncpus )
                p = multiprocessing.Pool(ncpus)
                data = p.map(optimize_fgap, sorted(NUTS_regions))
                p.close()

         
            # we time the optimization
            end_timestamp = datetime.utcnow()
            mylogger.info( 'Finished dynamic optimization at timestamp: %s' % ( end_timestamp-start_timestamp ))
         
            # we save the dictionary of optimum fgap in the output folder
            #mylogger.debug('ygf: %4.2f'% opti_fgap)
            #pickle_dump(opti_fgap, open(filename,'wb'))

    sys.exit()

#===============================================================================
def optimize_fgap(NUTS_no):
#===============================================================================
    from maries_toolbox import define_opti_years,\
                               select_cells, select_soils, get_EUR_frac_crop
#-------------------------------------------------------------------------------
    # if the optimization has already been performed and we don't want
    # to redo it, we skip that year x crop
    filename = os.path.join(outputdir,'ygf_%s_%s.pickle'%( NUTS_no,'observed') )
    if (os.path.exists(filename) and force_optimization == False):
        mylogger.info( "We have already calculated the optimum ygf for that year and crop" )
        return None
#-------------------------------------------------------------------------------
    # Get the NUTS region name
    mylogger.info( "Region: %s "%(NUTS_no) )


#-------------------------------------------------------------------------------
    # if there were no "non-shared" cells that were cultivated that year,
    # we skip that region in all further calculations
    shortlist_cells = select_cells(NUTS_no, year, CGMSgrid, CGMScropmask,CGMSsoil,\
                          method=selec_method, n=ncells,
                          select_from='cultivated')

    # we compute the list of whole grid cells contained in the region to write to pickle files if needed
    all_cells = select_cells(NUTS_no, year, CGMSgrid, CGMScropmask,CGMSsoil, 
                      method='all', select_from='cultivated')

    if (shortlist_cells == None):
        mylogger.info( "No cultivated grid cells for %s in region %s"%(crop,NUTS_no) )
        mylogger.info( "This crop/NUTS combination will be skipped from now on")
        filename = os.path.join(outputdir,'ygf_%s_noncultivated.pickle'% NUTS_no )
        outlist = [NUTS_no, 'noncultivated', np.NaN, [] ]
        pickle_dump(outlist, open(filename,'wb'))
        return None

#-------------------------------------------------------------------------------
    # NB: we do NOT detrend the yields anymore, since fgap is not supposed to be
    # representative of multi-annual gap
    try:
        detrend = observed_data[crop][NUTS_no]
    except:
        mylogger.info( 'This region code is unknown and will be skipped')
        filename = os.path.join(outputdir,'ygf_%s_notreported.pickle'% NUTS_no )
        outlist = [NUTS_no, 'missingyield', np.NaN, all_cells ]
        pickle_dump(outlist, open(filename,'wb'))
        return None
#-------------------------------------------------------------------------------
	# if there were no reported yields at all, or no reported yield for the
	# year of interest, we skip that region
    if (len(detrend[1])==0) or (year not in detrend[1]):
        mylogger.info( 'No reported yield of %s in %s in %s, optimum cannot be compiled'%(crop,NUTS_no, year) )
        if len(NUTS_no)==2: #WP NUTS level 0, needs to be gapfilled if missing
            mylogger.info( 'This region will need to be gapfilled')
            filename = os.path.join(outputdir,'ygf_%s_togapfill.pickle'% NUTS_no )
            outlist = [NUTS_no, 'togapfill', np.NaN, all_cells ]
        else:
            mylogger.info( 'This region will be skipped')
            filename = os.path.join(outputdir,'ygf_%s_notreported.pickle'% NUTS_no )
            outlist = [NUTS_no, 'missingyield', np.NaN, all_cells ]
        pickle_dump(outlist, open(filename,'wb'))
        return None
#-------------------------------------------------------------------------------
    # NB: in the optimization routine, we use the observed cultivation
    # fraction of the crop to calculate the soil cultivated areas, and
    # thus to compute the regional yields (= weighted average of yields
    # using soil cultivated areas)


    # if the observed cultivated fraction is zero, we skip that region
    selected_soil_types = select_soils(crop_no,
                               [g for g,a in shortlist_cells],
                               CGMSsoil,
                               method=selec_method, 
                               n=nsoils)

#-------------------------------------------------------------------------------
# Count the area used to calculate the harvest
    if (opti_metric == 'harvest'):
        areas = calc_cultivated_area_of_crop(shortlist_cells, 
                 selected_soil_types)
#-------------------------------------------------------------------------------
    # we set the optimization code (gives us info on how we optimize)
    opti_code = 1 # 1= observations are available for optimization
                  # 2= no obs available 

    mylogger.info( "Proceeding to optimize based on %d grid cells for %s in region %s"%(len(shortlist_cells),crop,NUTS_no) )

    # in all other cases, we optimize the yield gap factor
    optimum = optimize_regional_yldgapf_dyn(NUTS_no, detrend,
                                                crop_no,
                                                shortlist_cells,
                                                selected_soil_types,
                                                CGMSdir,
                                                [year],
                                                obs_type=opti_metric,
                                                plot_rmse=False)


    # pickle the information per NUTS region
    outlist = [NUTS_no, opti_code, optimum, all_cells]
    filename = os.path.join(outputdir,'ygf_%s_%s.pickle'% (NUTS_no, 'observed'))
    pickle_dump(outlist, open(filename,'wb'))

    return None

#===============================================================================
# Function to optimize the regional yield gap factor using the difference
# between the regional simulated and the observed harvest or yield (ie. 1 gap to
# optimize per NUTS region). This function iterates dynamically to find the
# optimum YLDGAPF.
def optimize_regional_yldgapf_dyn(NUTS_no_, detrend, crop_no_, 
    selected_grid_cells_, selected_soil_types_, inputdir, opti_years_, 
    obs_type='yield', plot_rmse=False):
#===============================================================================

    import math
    from cPickle import dump as pickle_dump
    from operator import itemgetter as operator_itemgetter
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

    # aggregated yield method:
    
    # 2- we construct a 2D array with same dimensions as TSO_regional,
    # containing the observed yields
    row = [] # this list will become the row of the 2D array
    for y,year in enumerate(opti_years_):
        index_year = np.argmin(np.absolute(detrend[1]-year))
        row = row + [detrend[0][index_year]]
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

        counter=0
        for grid, arable_land in selected_grid_cells_:
 
            frac_arable = arable_land / 625000000.

            # Retrieve the weather data of one grid cell (all years are in one
            # file) 
            if (weather == 'CGMS'):
                filename = os.path.join(inputdir,'weather_objects/',
                           'weatherobject_g%d.pickle'%grid)
                weatherdata = WeatherDataProvider()
                weatherdata._load(filename)
            if (weather == 'ECMWF'):
                weatherdata = CABOWeatherDataProvider('%i'%grid,fpath=ecmwfdir)
                        
            for smu, stu_no, weight, soildata in selected_soil_types_[grid]:

                # TSO will store all the yields of one grid cell x soil 
                # combination, for all years and all 3 yldgapf values
                TSO = np.zeros((len(f_range), len(opti_years_)))
                HIO = np.zeros((len(f_range), len(opti_years_)))

                counter +=1
        
                for y, year in enumerate(opti_years_): 

                    # Retrieve yearly data 
                    # 2 exceptions for the calendars: crossover across species
                    if crop_dict[crop][0]==5: # spring wheat uses spring barley calendars
                        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid,3,year)]
                    elif crop_dict[crop][0]==13: # winter barley uses winter wheat calendars
                        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid,1,year)]
                    else:
                        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid,crop_no_,year)]

                    # we use a temporary fix for sugar beet simulations:
                    if crop_dict[crop][0]==6:
                        if timerdata['END_DATE'] == timerdata['START_DATE']: 
                            timerdata['END_DATE'] = timerdata['CROP_END_DATE']
                        if timerdata['MAX_DURATION'] == 0: 
                            timerdata['MAX_DURATION']=300

                    cropdata  = CGMScrop['cropobject_g%d_c%d_y%d'%(grid,crop_no_,year)]
                    sitedata  = CGMSsite['siteobject_g%d_c%d_y%d_s%d'%(grid,crop_no_,year,stu_no)]

                    for f,factor in enumerate(f_range):
            
                        cropdata['YLDGAPF']=factor
                       
                        # run WOFOST
                        wofost_object = Wofost71_WLP_FD(sitedata, timerdata,
                                                soildata, cropdata, weatherdata)
                        wofost_object.run_till_terminate()
        
                        # get the yield (in kgDM.ha-1) 
                        hi = wofost_object.get_variable('HI')

                        if hi < 0.15:
                            mylogger.warning ('WARNING: simulated harvest index suspiciously small (%f)'%hi)

                        # get the yield (in kgDM.ha-1) 
                        TSO[f,y] = wofost_object.get_variable('TWSO')
                        HIO[f,y] = wofost_object.get_variable('HI')


                    #print grid, smu, year, counter, TSO[-1]
                RES = RES + [(grid, stu_no, weight*frac_arable, TSO, HIO)]

        # 4- we aggregate the yield or harvest into the regional one with array
        # operations

        sum_weighted_vals = np.zeros((len(f_range), len(opti_years_)))
        sum_weighted_hio = np.zeros((len(f_range), len(opti_years_)))
                                    # empty 2D array with same dimension as TSO
        sum_weights       = 0.
        for grid, stu_no, weight, TSO, HIO in RES:
            # adding weighted 2D-arrays in the empty array sum_weighted_yields
            # NB: variable 'weight' is actually the cultivated area in m2
            sum_weighted_vals   = sum_weighted_vals + (weight/10000.)*TSO 
            sum_weighted_hio   = sum_weighted_hio + (weight/10000.)*HIO 
            # computing the total sum of the cultivated area in ha 
            sum_weights         = sum_weights       + (weight/10000.) 

        if (obs_type == 'harvest'):
            TSO_regional = sum_weighted_vals / 1000000. # sum of the individual 
                                                        # harvests in 1000 tDM
            HIO_regional = sum_weighted_hio / 1000000. # sum of the individual 
                                                        # harvest indices [0-1]
        elif (obs_type == 'yield'):
            TSO_regional = sum_weighted_vals / sum_weights # weighted average of 
                                                        # all yields in kgDM/ha
            HIO_regional = sum_weighted_hio / sum_weights # weighted average of 
                                                        # harvest indices [0-1]

        # 5- we compute the (sim-obs) differences.
        DIFF = TSO_regional - OBS
        if (TSO_regional[-1][0] <= 0.):
            mylogger.warning ('WARNING: no simulated crop growth. We set the optimum ygf to 0.666')
            return 0.666
        
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

        if iter_no == 1: 
            observed_yield  = RMSE[0]
            potential_yield = RMSE[-1] + observed_yield

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
        else:
            f0 = f_range[index_new_center-1]
            f2 = f_range[index_new_center]
            f4 = f_range[index_new_center+1]

	# when we are finished iterating on the yield gap factor range, we plot the
    # RMSE as a function of the yield gap factor
    if (plot_rmse == True):
        from matplotlib import pyplot as plt
        RMSE_stored  = sorted(RMSE_stored, key=operator_itemgetter(0))
        x,y = zip(*RMSE_stored)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        ax.plot(x, y, c='k', marker='o')
        ax.set_xlabel('yldgapf (-)')
        ax.set_ylabel('RMSE')
        fig.savefig('%s_opti_ygf.png'%NUTS_no_)

    # 8- when we are finished iterating on the yield gap factor range, we return
    # the optimum value. We look for the yldgapf with the lowest RMSE
    index_optimum   = RMSE.argmin()
    optimum_yldgapf = f_range[index_optimum] 

    mylogger.info( 'optimum found: %.2f +/- %.2f with RMSE: (%5.2f) and observed yield of %7.2f at Harvest Index of %4.2f'%(optimum_yldgapf, f_step,RMSE.min(),OBS[-1], HIO_regional[index_optimum,0]))

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
    optimize()
#===============================================================================

