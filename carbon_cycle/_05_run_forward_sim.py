#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load
from cPickle import dump as pickle_dump
from operator import itemgetter as operator_itemgetter
from datetime import datetime

#===============================================================================
def main():
#===============================================================================
    """
	This method executes wofost forward runs that are by default using crop,
    soil input data from the CGMS database (12x12 km) and ECMWF weather data put 
    on the CGMS grid. The user can decide if to use an optimum yield gap factor, 
    or to do potential simulations with fgap = 1.

    """
#-------------------------------------------------------------------------------
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
# declare as global variables: folder names, main() method arguments, and a few
# variables/constants passed between functions
    global cwdir, CGMSdir, EUROSTATdir, ecmwfdir, yldgapfdir, wofostdir,\
           potential_sim, force_sim, selec_method, nsoils, weather,\
           crop, crop_dict, crop_no, year, start_timestamp, optimi_code, fgap,\
           CGMSsoil, CGMScropmask, CGMScrop, CGMStimer, CGMSsite
#-------------------------------------------------------------------------------
# Temporarily add the parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # forward run settings:
    potential_sim = True     # decide if to do potential / optimum simulations
    force_sim     = True     # decides if we overwrite the forward simulations,
                              # in case the results file already exists
    selec_method  = 'topn'    # for soils: can be 'topn' or 'randomn' or 'all'
    nsoils        = 3         # number of selected soil types within a grid cell
    weather       = 'ECMWF'   # weather data used for the forward simulations
                              # can be 'CGMS' or 'ECMWF'

    process  = 'serial'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 12          # number of cores used in case of a parallelization

    # input data directory path
    #inputdir = '/Users/mariecombe/mnt/promise/CO2/wofost/'
    inputdir = '/Users/mariecombe/Documents/Work/Research_project_3/model_input_data/'

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    cwdir       = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
    EUROSTATdir = os.path.join(inputdir, 'EUROSTATobs')
    ecmwfdir    = os.path.join(inputdir, 'CABO_weather_ECMWF')
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    try:
        crop_dict    = pickle_load(open('../tmp/selected_crops.pickle','rb'))
        years        = pickle_load(open('../tmp/selected_years.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of crops and years '+\
              'to loop over'
        print 'Run the script _01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# open the pickle files containing the CGMS input data
    CGMSsoil  = pickle_load(open(os.path.join(CGMSdir,'CGMSsoil.pickle'),'rb'))
#-------------------------------------------------------------------------------
# PERFORM FORWARD RUNS:
#-------------------------------------------------------------------------------
# loop over years:
    for year in years:
        print '\nYear ', year
        print '================================================'
#-------------------------------------------------------------------------------
# loop over crops
        for crop in sorted(crop_dict.keys()):
            crop_name  = crop_dict[crop][1]
            print '\nCrop no %i: %s / %s'%(crop_dict[crop][0], crop, crop_name)
            print '================================================'
            if crop_dict[crop][0]==5:
                crop_no = 1
            elif crop_dict[crop][0]==13:
                crop_no = 3
            elif crop_dict[crop][0]==12:
                crop_no = 2
            else:
                crop_no = crop_dict[crop][0]
#-------------------------------------------------------------------------------
# wofost runs output folder: create if needed, wipe if required by user
            if potential_sim:
                subfolder = 'potential' 
            else:
                subfolder = 'optimized' 
            wofostdir = os.path.join(cwdir,"../output/%i/c%i/wofost/"%(year,
                                                 crop_dict[crop][0]), subfolder)
            # create output folder if needed
            if not os.path.exists(wofostdir):
                os.makedirs(wofostdir)
            # empty folder if required by user
            if (os.path.exists(wofostdir) and force_sim == True):
                filelist = [f for f in os.listdir(wofostdir)]
                for f in filelist:
                    os.remove(os.path.join(wofostdir,f))

#-------------------------------------------------------------------------------
# load the CGMS input data for crop parameters and calendars, and site parameters

            # crop mask (do NOT use the crossover crop_no here)
            filename = os.path.join(CGMSdir, 'cropdata_objects',
                                       'cropmask_c%i.pickle'%crop_dict[crop][0])
            CGMScropmask = pickle_load(open(filename, 'rb'))
            # crop parameters (where needed use the crossover crop_no)
            filename = os.path.join(CGMSdir, 'cropdata_objects',
                                        'CGMScrop_%i_c%i.pickle'%(year,crop_no))
            CGMScrop = pickle_load(open(filename, 'rb'))
            print 'Successfully loaded the CGMS crop pickle files:', \
                  'cropmask_c%i.pickle'%crop_dict[crop][0], 'and', \
                  'CGMScrop_%i_c%i.pickle'%(year,crop_no)

            # crop calendars (use crossover crop_no, plus 2 exceptions)
            if crop_dict[crop][0]==5: # spring wheat uses spring barley calendars
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                             'CGMStimer_%i_c%i.pickle'%(year,3))
                CGMStimer = pickle_load(open(filename, 'rb'))
                print 'Successfully loaded the CGMS timer pickle file:',\
                      'CGMStimer_%i_c%i.pickle'%(year,3)
            elif crop_dict[crop][0]==13: # winter barley uses winter wheat calendars
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                             'CGMStimer_%i_c%i.pickle'%(year,1))
                CGMStimer = pickle_load(open(filename, 'rb'))
                print 'Successfully loaded the CGMS timer pickle file:',\
                      'CGMStimer_%i_c%i.pickle'%(year,1)
            else:
                filename = os.path.join(CGMSdir, 'timerdata_objects',
                                       'CGMStimer_%i_c%i.pickle'%(year,crop_no))
                CGMStimer = pickle_load(open(filename, 'rb'))
                print 'Successfully loaded the CGMS timer pickle file:',\
                      'CGMStimer_%i_c%i.pickle'%(year,crop_no)

            # site parameters (use the crossover crop_no)
            filename = os.path.join(CGMSdir, 'sitedata_objects',
                                        'CGMSsite_%i_c%i.pickle'%(year,crop_no))
            CGMSsite = pickle_load(open(filename, 'rb'))
            print 'Successfully loaded the CGMS site pickle file:',\
                  'CGMSsite_%i_c%i.pickle'%(year,crop_no)
#-------------------------------------------------------------------------------
# OPTIMIZED FORWARD RUNS:
#-------------------------------------------------------------------------------
            if not potential_sim:
                print '\nOPTIMIZED MODE: we use the available optimum fgap'
#-------------------------------------------------------------------------------
                # print out some information to user
                if force_sim:
                    print 'FORCE MODE: we just wiped the wofost output directory'
                else:
                    print 'SKIP MODE: we skip any simulation already performed'

                # we retrieve the optimum yield gap factor output files
                yldgapfdir = os.path.join(cwdir,"../output/%i/c%i/fgap/"%(year,
                                                            crop_dict[crop][0]))
                if not os.path.exists(yldgapfdir):
                    print "You haven't optimized the yield gap factor!!"
                    print "Run the script _04_optimize_fgap.py first!"
                    continue
             
                # list the regions for which we have been able to optimize fgap
                filelist = [ f for f in os.listdir(yldgapfdir) if '_optimized' in f]

                #---------------------------------------------------------------
                # loop over NUTS regions - this is the parallelized part -
                #---------------------------------------------------------------
             
                # We add a timestamp at start of the forward runs
                start_timestamp = datetime.utcnow()
                
                # if we do a serial iteration, we loop over the grid cells
                # that contain arable land
                if (process == 'serial'):
                    for f in filelist:
                        forward_sim_per_region(f)
                
                # if we do a parallelization, we use the multiprocessor
                # module to provide series of cells to the function
                if (process == 'parallel'):
                    import multiprocessing
                    p = multiprocessing.Pool(nb_cores)
                    data = p.map(forward_sim_per_region, filelist)
                    p.close()
            
                # We add an end timestamp to time the process
                end_timestamp = datetime.utcnow()
                print '\nDuration of the optimized runs:', end_timestamp - \
                start_timestamp

#-------------------------------------------------------------------------------
# POTENTIAL FORWARD RUNS:
#-------------------------------------------------------------------------------
            if potential_sim:
                print '\nPOTENTIAL MODE: we use a yield gap factor of 1.'
#-------------------------------------------------------------------------------
                # print out some more information to user
                if force_sim:
                    print 'FORCE MODE: we just wiped the wofost output directory\n'
                else:
                    print 'SKIP MODE: we skip any simulation already performed\n'

                # we retrieve the list of cultivated grid cells:
                culti_grid = CGMScropmask
                grid_shortlist = list(set([g for g,a in culti_grid[year]]))

                # we set the yield gap factor to 1
                fgap = 1.

                #---------------------------------------------------------------
                # loop over grid cells - this is the parallelized part -
                #---------------------------------------------------------------

                # We add a timestamp at start of the forward runs
                start_timestamp = datetime.utcnow()
                
                # if we do a serial iteration, we loop over the grid cells that 
                # contain arable land
                if (process == 'serial'):
                    for grid in sorted(grid_shortlist):
                        forward_sim_per_grid(grid)
                
                # if we do a parallelization, we use the multiprocessor module
                # to provide series of cells to the function
                if (process == 'parallel'):
                    import multiprocessing
                    p = multiprocessing.Pool(nb_cores)
                    data = p.map(forward_sim_per_grid, sorted(grid_shortlist))
                    p.close()
            
                # We add an end timestamp to time the process
                end_timestamp = datetime.utcnow()
                print '\nDuration of the potential runs:', end_timestamp - \
                start_timestamp

            # we open a results file to write only summary output (for
            # harvest maps)
            #regroup_summary_output()
            

#===============================================================================
def forward_sim_per_region(fgap_filename):
#===============================================================================
    global fgap

    # get the optimum fgap and the grid cell list for these regions
    optimi_info = pickle_load(open(os.path.join(yldgapfdir,fgap_filename),'rb')) 
    print fgap_filename, optimi_info
    NUTS_no     = optimi_info[0]
    optimi_code = optimi_info[1]
    fgap        = optimi_info[2]
    grid_shortlist = list(set([ g for g,a in optimi_info[3] ]))
    print '\n NUTS region: %s\n'%NUTS_no

    for grid in sorted(grid_shortlist):
        forward_sim_per_grid(grid)

    return None

#===============================================================================
def forward_sim_per_grid(grid_no):
#===============================================================================
    """
    Function to do forward simulations of crop yield for a given grid cell no

    """
    import glob
    from maries_toolbox import select_soils
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

    print '    - grid cell %i, yield gap factor of %.2f'%(grid_no, fgap)

    # skipping already performed forward runs if required by user
    outlist = glob.glob(os.path.join(wofostdir,'wofost_g%i*'%grid_no))
    if (len(outlist)==nsoils and force_sim == False):
        print "        We have already done that forward run! Skipping."
        return None

    # Retrieve the weather data of one grid cell
    if (weather == 'CGMS'):
        filename = os.path.join(CGMSdir,'weather_objects/',
                   'weatherobject_g%d.pickle'%grid_no)
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
    if (weather == 'ECMWF'):
        weatherdata = CABOWeatherDataProvider('%i'%(grid_no),fpath=ecmwfdir)
    #print weatherdata(datetime.date(datetime(2006,4,1)))
 
    # Retrieve the soil types, crop calendar, crop species
    soil_iterator = ['soilobject_g%d'%grid_no]
    if crop_dict[crop][0]==5: # spring wheat uses spring barley calendars
        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid_no,3,year)]
    elif crop_dict[crop][0]==13: # winter barley uses winter wheat calendars
        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid_no,1,year)]
    else:
        timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid_no,crop_no,year)]

    # we use a temporary fix for sugar beet simulations:
    if crop_dict[crop][0]==6:
        if timerdata['END_DATE'] == timerdata['START_DATE']: 
            timerdata['END_DATE'] = timerdata['CROP_END_DATE']
        if timerdata['MAX_DURATION'] == 0: 
            timerdata['MAX_DURATION']=300

    cropdata  = CGMScrop['cropobject_g%d_c%d_y%d'%(grid_no,crop_no,year)]
    cropdata['CRPNAM'] = crop
    cropdata['YLDGAPF'] = fgap
 
    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no], CGMSsoil, 
                                       method=selec_method, n=nsoils)
 
    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        print '        soil type no %i'%stu_no
        
        wofostfile = os.path.join(wofostdir, "wofost_g%i_s%i.txt"\
                     %(grid_no,stu_no))
 
        # Retrieve the site data of one year, one grid cell, one soil type
        sitedata = CGMSsite['siteobject_g%d_c%d_y%d_s%d'%(grid_no,crop_no,year,stu_no)]
 
        # run WOFOST
        wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, 
                                        cropdata, weatherdata)
        wofost_object.run_till_terminate() #will stop the run when DVS=2
 
        # get time series of the output and take the selected variables
        wofost_object.store_to_file(wofostfile)
 
        ## get major summary output variables for each run
        ## total dry weight of - dead and alive - storage organs (kg/ha)
        #TSO       = wofost_object.get_variable('TWSO')
        ## total dry weight of - dead and alive - leaves (kg/ha) 
        #TLV       = wofost_object.get_variable('TWLV')
        ## total dry weight of - dead and alive - stems (kg/ha) 
        #TST       = wofost_object.get_variable('TWST')
        ## total dry weight of - dead and alive - roots (kg/ha) 
        #TRT       = wofost_object.get_variable('TWRT')
        ## maximum LAI
        #MLAI      = wofost_object.get_variable('LAIMAX')
        ## rooting depth (cm)
        #RD        = wofost_object.get_variable('RD')
        ## Total above ground dry matter (kg/ha)
        #TAGP      = wofost_object.get_variable('TAGP')
 
        ##output_string = '%10.3f, %8i, %5i, %7i, %15.2f, %12.5f, %14.2f, '
        #                #%(yldgapf, grid_no, year, stu_no, arable_area/10000.,stu_area/10000.,TSO) 
        #output_string = '%10.3f, %8i, %5i, %7i, %12.5f, %14.2f, '%(yldgapf,
        #                 grid_no, year, stu_no, stu_area/10000., TSO) + \
        #                '%14.2f, %14.2f, %14.2f, %14.2f, %13.2f, %15.2f\n'%(TLV,
        #                 TST, TRT, MLAI, RD, TAGP)
 
        ## we pickle the one-liner summary output
        #filename = 'pcse_oneline_c%i_y%i_g%i_s%i.pickle'%(crop_no,year,grid_no,
        #                                                             stu_no)
        #if os.path.exists(os.path.join(pcseoutputdir,filename)):
        #    os.remove(os.path.join(pcseoutputdir, filename))
        #pickle_dump(output_string,open(os.path.join(pcseoutputdir,filename),'wb'))

    return None

#===============================================================================
def regroup_summary_output():
#===============================================================================

    # we open a results file to write only summary output (for harvest maps)
    res_filename  = 'pcse_output_crop%i_year%i.csv'%(crop_no, year)
    if (os.path.isfile(os.path.join(pcseoutputdir, res_filename))):
        os.remove(os.path.join(pcseoutputdir, res_filename))
        print '\nDeleted old file %s in folder pcse_output/'%res_filename
    Results = open(os.path.join(pcseoutputdir, res_filename), 'w')

    # we write the header line:
    #Results.write('YLDGAPF(-),  grid_no,  year,  stu_no, arable_area(ha), '\
    Results.write('YLDGAPF(-),  grid_no,  year,  stu_no,  '\
                 +'stu_area(ha), TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)\n')

    # get list of files
    list_of_files = [f for f in os.listdir(cwdir) 
                     if ( f.startswith('pcse_oneline') 
                     and ('.pickle' in f) ) ]

    for namefile in list_of_files:
        line = pickle_load(open(namefile,'rb'))
        Results.write(line)
        os.remove(os.path.join(pcseoutputdir,namefile))

    # close the summary results file
    Results.close()	

    return None

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
