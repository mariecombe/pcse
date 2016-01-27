#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys, os
import numpy as np

from cPickle import load as pickle_load
from cPickle import dump as pickle_dump
from operator import itemgetter as operator_itemgetter
from datetime import datetime
from maries_toolbox import select_soils

#===============================================================================
# This script executes forward simulations of WOFOST for all cultivated CGMS 
# grid cells (which ones are depends on crop species and year)
def main():
#===============================================================================
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
    global cwdir, CGMSdir, EUROSTATdir, ecmwfdir,\
           force_fwdsim, selec_method, nsoils, weather,\
           crop, crop_no, year, \
           start_timestamp, optimi_code, fgap
#-------------------------------------------------------------------------------
# Temporarily add parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # forward run settings:
    force_fwdsim = False      # decides if we overwrite the forward simulations,
                              # in case the results file already exists
    selec_method  = 'topn'    # for soils: can be 'topn' or 'randomn' or 'all'
    nsoils        = 3         # number of selected soil types within a grid cell
    weather       = 'ECMWF'   # weather data used for the forward simulations
                              # can be 'CGMS' or 'ECMWF'

    process  = 'parallel'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 12          # number of cores used in case of a parallelization

    # input data directory path
    inputdir = '/Users/mariecombe/mnt/promise/CO2/wofost/'

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
# PERFORM FORWARD RUNS:
#-------------------------------------------------------------------------------
# loop over years:
    for year in years:
        print '\nYear ', year
        print '================================================'
#-------------------------------------------------------------------------------
# loop over crops
        for crop in sorted(crop_dict.keys()):
            crop_no    = crop_dict[crop][0]
            crop_name  = crop_dict[crop][1]
            print '\nCrop no %i: %s / %s'%(crop_no, crop, crop_name)
            print '================================================'
#-------------------------------------------------------------------------------
# loop over NUTS regions
            outputdir = os.path.join(cwdir,"../output/%i/%s/fgap/"%(year,crop))
            if not os.path.exists(outputdir):
                print "You haven't optimized the yield gap factor!!"
                print "Run the script _04_optimize_fgap.py first!"
                continue

            # list the regions for which we have been able to optimize fgap
            filelist = [ f for f in os.listdir(outputdir)]
            for f in filelist:

                # get the optimum fgap and the grid cell list for these regions
                optimi_info = pickle_load(open(os.path.join(outputdir,f),'rb')) 
                NUTS_no     = optimi_info[0]
                optimi_code = optimi_info[1]
                fgap        = optimi_info[2]
                grid_shortlist = list(set([ g for g,a in optimi_info[3] ]))
                print '\n NUTS region: %s\n'%NUTS_no

#-------------------------------------------------------------------------------
# loop over grid cells - this is the parallelized part -
#-------------------------------------------------------------------------------
                # We add a timestamp at start of the forward runs
                start_timestamp = datetime.utcnow()
             
                # if we do a serial iteration, we loop over the grid cells that 
                # contain arable land
                if (process == 'serial'):
                    for grid in grid_shortlist:
                        perform_yield_sim(grid)
             
                # if we do a parallelization, we use the multiprocessor module to 
                # provide series of cells to the function
                if (process == 'parallel'):
                    import multiprocessing
                    p = multiprocessing.Pool(nb_cores)
                    data = p.map(perform_yield_sim, grid_shortlist)
                    p.close()
 
            # we open a results file to write only summary output (for harvest maps)
            #regroup_summary_output()
 
            # We add a timestamp at end of the forward runs, to time the process
            end_timestamp = datetime.utcnow()
            print '\nDuration of the forward runs:', end_timestamp-start_timestamp


#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sim(grid_no):
#===============================================================================
    
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

    print '    - grid cell %i'%grid_no
    #try:

    # construct output folder name
    if (optimi_code == 0):
        subfolder = 'potential' 
    elif (optimi_code >= 1):
        subfolder = 'optimized/%02d/'%optimi_code 
    wofostdir = os.path.join(cwdir,"../output/%i/%s/wofost/"%(year,crop),
                subfolder)
    # create output folder if needed
    if not os.path.exists(wofostdir):
        os.makedirs(wofostdir)
    # empty folder if required by user
    if (os.path.exists(wofostdir) and force_fwdsim == True):
        filelist = [f for f in os.listdir(wofostdir)]
        for f in filelist:
            os.remove(os.path.join(outputdir,f))
        print "We force the forward runs: wofost output directory just got emptied"

    # Retrieve the weather data of one grid cell
    if (weather == 'CGMS'):
        filename = os.path.join(CGMSdir,'weather_objects/',
                   'weatherobject_g%d.pickle'%grid_no)
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
    if (weather == 'ECMWF'):
        weatherdata = CABOWeatherDataProvider('%i'%(grid_no),fpath=ecmwfdir)
    #print weatherdata(datetime.date(datetime(2006,4,1)))

    # Retrieve the soil data of one grid cell 
    filename = os.path.join(CGMSdir,'soildata_objects/',
               'soilobject_g%d.pickle'%grid_no)
    soil_iterator = pickle_load(open(filename,'rb'))

    # Retrieve calendar data of one year for one grid cell
    filename = os.path.join(CGMSdir,
               'timerdata_objects/%i/c%i/'%(year,crop_no),
               'timerobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year))
    timerdata = pickle_load(open(filename,'rb'))
                    
    # Retrieve crop data of one year for one grid cell
    filename = os.path.join(CGMSdir,
               'cropdata_objects/%i/c%i/'%(year,crop_no),
               'cropobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year))
    cropdata = pickle_load(open(filename,'rb'))

    # retrieve the fgap data of one year and one grid cell
    cropdata['YLDGAPF'] = fgap

    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no], CGMSdir, 
                                       method=selec_method, n=nsoils)

    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        print '        soil type no %i'%stu_no
        
        wofostfile = os.path.join(wofostdir, "wofost_g%i_s%i.txt"\
                     %(grid_no,stu_no))
        if (os.path.exists(wofostfile) and force_fwdsim == False):
            print "        We have already done that forward run! Skipping."
            continue

        # Retrieve the site data of one year, one grid cell, one soil type
        if str(grid_no).startswith('1'):
            dum = str(grid_no)[0:2]
        else:
            dum = str(grid_no)[0]
        filename = os.path.join(CGMSdir,
                   'sitedata_objects/%i/c%i/grid_%s/'%(year,crop_no,dum),
                   'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid_no,crop_no,year,
                                                                    stu_no))
        sitedata = pickle_load(open(filename,'rb'))

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

    # if there are missing input crop or calendar files, the crop was not grown that year
    #except IOError:
    #    print 'The crop was not grown that year in grid cell no %i'%grid_no

    ## any other error:
    #except Exception as e:
    #    print 'Unexpected error:', sys.exc_info()[0]
    #    end_timestamp = datetime.utcnow()
    #    print 'Duration of runs until Error:', end_timestamp-start_timestamp, '\n'
        

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
