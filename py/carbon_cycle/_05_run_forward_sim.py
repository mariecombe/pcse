#!/usr/bin/env python

# import modules needed in all the methods of this script:
import sys
from os import path
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import glob, os
import numpy as np
from py.tools.initexit import start_logger, parse_options
from py.carbon_cycle._01_select_crops_n_regions import select_crops_regions
import py.tools.rc as rc
import logging
import tarfile

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
           crop, crop_no, year, start_timestamp, optimi_code, fgap, opt_type, pickle_load,\
           CGMSsoil, CGMScropmask, CGMScrop, CGMStimer, CGMSsite
#-------------------------------------------------------------------------------
# Temporarily add the parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 

    _ = start_logger(level=logging.DEBUG)

    opts, args = parse_options()

    # First message from logger
    logging.info('Python Code has started')
    logging.info('Passed arguments:')
    for arg,val in args.iteritems():
        logging.info('         %s : %s'%(arg,val) )

    rcfilename = args['rc']
    rcF = rc.read(rcfilename)
    crops = [ rcF['crop'] ]
    #crops = [i.strip().replace(' ','_') for i in crops]
    years = [int(rcF['year'])]
    opt_type = rcF['optimize.type']
    outputdir = rcF['dir.output']
    outputdir = os.path.join(outputdir,'wofost')
    inputdir = rcF['dir.wofost.input']
    par_process = (rcF['fwd.wofost.parallel'] in ['True','TRUE','true','T'])
    nsoils = int(rcF['fwd.wofost.nsoils'])
    weather = rcF['fwd.wofost.weather']
    selec_method = rcF['fwd.wofost.method']
    potential_sim = (rcF['fwd.wofost.potential'] in ['True','TRUE','true','T'])

    if opt_type not in ['observed','gapfilled']:
        logging.error('The specified optimization type (%s) in the call argument is not recognized' % opt_type )
        logging.error('Please use either "observed" or "gapfilled" as value in the main rc-file')
        sys.exit(2)

#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================

    # forward run settings:
    force_sim     = False     # decides if we overwrite the forward simulations,
                              # in case the results file already exists

    # input data directory path
    #inputdir = os.path.join('/Users',os.environ["USER"],'mnt/promise/CO2/wofost')

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    cwdir       = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
    EUROSTATdir = os.path.join(inputdir, 'EUROSTATobs')
    ecmwfdir    = os.path.join(inputdir, 'CABO_weather_ECMWF')
#-------------------------------------------------------------------------------
# we retrieve the crops and years to loop over:
    NUTS_regions,crop_dict = select_crops_regions(crops, EUROSTATdir)
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
            crop_no    = crop_dict[crop][0]
            crop_name  = crop_dict[crop][1]
            print '\nCrop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name)
            print '================================================'
#-------------------------------------------------------------------------------
# wofost runs output folder: create if needed, wipe if required by user
            if potential_sim:
                subfolder = 'potential' 
            else:
                subfolder = 'optimized' 
            wofostdir = os.path.join(outputdir, subfolder)
            # create output folder if needed
            if not os.path.exists(wofostdir):
                os.makedirs(wofostdir)
                logging.info("Created new folder for output (%s)"%wofostdir)
            # empty folder if required by user
            if (os.path.exists(wofostdir) and force_sim == True):
                filelist = [f for f in os.listdir(wofostdir)]
                for f in filelist:
                    os.remove(os.path.join(wofostdir,f))

#-------------------------------------------------------------------------------
# load the CGMS input data for crop parameters and calendars, and site parameters

            filename = os.path.join(CGMSdir, 'cropdata_objects',
                                                  'cropmask_c%i.pickle'%crop_no)
            CGMScropmask = pickle_load(open(filename, 'rb'))
            filename = os.path.join(CGMSdir, 'cropdata_objects',
                                        'CGMScrop_%i_c%i.pickle'%(year,crop_no))
            CGMScrop = pickle_load(open(filename, 'rb'))
            print 'Successfully loaded the CGMS crop pickle files'

            filename = os.path.join(CGMSdir, 'timerdata_objects',
                                       'CGMStimer_%i_c%i.pickle'%(year,crop_no))
            CGMStimer = pickle_load(open(filename, 'rb'))
            print 'Successfully loaded the CGMS timer pickle file'

            filename = os.path.join(CGMSdir, 'sitedata_objects',
                                        'CGMSsite_%i_c%i.pickle'%(year,crop_no))
            CGMSsite = pickle_load(open(filename, 'rb'))
            print 'Successfully loaded the CGMS site pickle file'
#-------------------------------------------------------------------------------
# OPTIMIZED FORWARD RUNS:
#-------------------------------------------------------------------------------
            if not potential_sim:
                logging.info( 'OPTIMIZED MODE: we use the available optimum ygf' )
#-------------------------------------------------------------------------------
                # print out some information to user
                if force_sim:
                    logging.info( 'FORCE MODE: we just wiped the wofost output directory' )
                else:
                    logging.info( 'SKIP MODE: we skip any simulation already performed' )

                # we retrieve the optimum yield gap factor output files
                yldgapfdir = os.path.join(outputdir.replace('wofost','ygf') )
                if not os.path.exists(yldgapfdir):
                    logging.error( "You haven't optimized the yield gap factor!!" )
                    logging.error( "Run the script _04_optimize_fgap.py first!" )
                    sys.exit(2)
                    continue
             
                # list the regions for which we have been able to optimize fgap
                filelist = [ f for f in os.listdir(yldgapfdir) if opt_type in f]  #WP Selection for only observed, or only gap-filled NUTS
                logging.info( "Found %d optimized yield gap factor files"%len(filelist) )

                #---------------------------------------------------------------
                # loop over NUTS regions - this is the parallelized part -
                #---------------------------------------------------------------
             
                # We add a timestamp at start of the forward runs
                start_timestamp = datetime.utcnow()
                
                # if we do a serial iteration, we loop over the grid cells
                # that contain arable land
                for f in filelist:
                    # get the optimum fgap and the grid cell list for these regions
                    optimi_info = pickle_load(open(os.path.join(yldgapfdir,f),'rb')) 
                    NUTS_no     = optimi_info[0]
                    optimi_code = optimi_info[1]
                    fgap        = optimi_info[2]
                    grid_shortlist = list(set([ g for g,a in optimi_info[3] ]))
                    logging.info( '    NUTS region (n=%d): %s'%(NUTS_no,len(grid_shortlist) )
                    print ( '    NUTS region (n=%d): %s'%(NUTS_no,len(grid_shortlist) )


                    tarfilelist=[]
                    if (par_process):
                        import multiprocessing
                        # get number of cpus available to job
                        try:
                            ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])/2
                            print "Success reading parallel env %d" % ncpus
                        except KeyError:
                            ncpus = multiprocessing.cpu_count()/2
                            print "Success obtaining processor count %d" % ncpus
                        NUTS_nos = [NUTS_no]*len(grid_shortlist)
                        fgaps = [fgap]*len(grid_shortlist)
                        arguments = zip(grid_shortlist,NUTS_nos, fgaps)
                        p = multiprocessing.Pool(ncpus)
                        tarfilelist = p.map(forward_sim_per_grid, arguments)
                        p.close()

                    else: 
                        for grid in sorted(grid_shortlist):
                            wofostfile = forward_sim_per_grid((grid, NUTS_no, fgap))
                            tarfilelist.extend(wofostfile)

                    outputfile = os.path.join(wofostdir, "wofost_%s_results.tgz" %NUTS_no)
                    if os.path.exists(outputfile):
                        logging.debug('tar output file exists, removing')
                        print ('tar output file exists, removing')
                        os.remove(outputfile)
                        tarmode = 'w:gz'
                    else:
                        logging.debug('tar output file does not exist, creating')
                        print ('tar output file does not exist, creating')
                        tarmode = 'w:gz'

                    with tarfile.open(outputfile,tarmode) as tarf:
                        for f in tarfilelist:
                            tarf.add(f,recursive=False,arcname=os.path.basename(f))
                            os.remove(f)

                
#===============================================================================
def forward_sim_per_grid(arguments):
#===============================================================================
    """
    Function to do forward simulations of crop yield for a given grid cell no

    """
    import glob
    import logging
    from maries_toolbox import select_soils
    from pcse.models import Wofost71_WLP_FD
    from pcse.base_classes import WeatherDataProvider
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider

    grid_no, NUTS_no, fgap = arguments

    _ = start_logger(level=logging.DEBUG)

    logging.info( '    - grid cell %i, yield gap factor of %.2f'%(grid_no, fgap) )
    print '    - grid cell %i, yield gap factor of %.2f'%(grid_no, fgap)

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
    timerdata = CGMStimer['timerobject_g%d_c%d_y%d'%(grid_no,crop_no,year)]
    cropdata  = CGMScrop['cropobject_g%d_c%d_y%d'%(grid_no,crop_no,year)]
    cropdata['YLDGAPF'] = fgap
 
    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no], CGMSsoil, 
                                       method=selec_method, n=nsoils)
 
    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        logging.info( '        soil type no %i'%stu_no )
        print  '        soil type no %i'%stu_no 
        
        wofostfile = os.path.join(wofostdir, "wofost_%s_g%i_s%i_%s.txt"\
                     %(NUTS_no,grid_no,stu_no,opt_type))

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

    return wofostfile

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
