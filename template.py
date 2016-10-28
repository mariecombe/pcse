#!/usr/bin/env python
import sys
import os
import logging
import py.tools.rc as rc
from py.tools.initexit import start_logger, parse_options

# Append path so that modules can be imported
sys.path.append(os.getcwd())

# Set up the basic logging style, start logging from DEBUG and higher

_ = start_logger()

args,opts = parse_options()

# First message from logger
logging.info('Python Code has started')

# Now read the rc-file
rcfile = 'template.rc'
try:
    rcitems = rc.read(rcfile)
except:
    logging.error("Could not open file %s"% rcfile)

logging.debug("  Start date: %s"% rcitems['time.years'])
logging.info("  Crop type : %s"% rcitems['crop.types'])

years = rcitems['time.years'].split(',')  # the years to cover
crops = rcitems['crop.types'].split(',')  # the crop types to consider
projectdir = rcitems['dir.project']
inputdir = rcitems['dir.wofost.input']
if not os.path.exists(inputdir):
    logging.error('Input path specified in the rc-file does not exist, exiting...')
    logging.error('rc-file path: %s'%inputdir)
    sys.exit(2)

rundir = os.path.join(projectdir,'exec')
if not os.path.exists(rundir): os.makedirs(rundir)
outputdir = os.path.join(projectdir,'output')
if not os.path.exists(outputdir): os.makedirs(outputdir)

# Next open platform class

import py.platforms.cartesius as platform

pf = platform.CartesiusPlatform()

# And create a loop over years and crop types

for year in years:
    for crop in crops:

        # create directory structure for optimized output per crop and year
       
        dirname = os.path.join(outputdir,'%s'%year.strip(),crop.strip().replace(' ','_') )  
        if not os.path.exists(dirname): 
            os.makedirs(dirname)
            logging.info('Created new folder: %s'%dirname)

        # Create rc-file for the subjobs to be started, using some run specific and some
        # general rc-keys. The latter are copied from the main rc-file
        jobrc = {'year' : year.strip(),'crop' : crop.strip() , 'dir.output' : dirname}
        for k,v in rcitems.iteritems():
            if not jobrc.has_key(k):
                jobrc[k]=v

        if 'run-opt' in rcitems['steps.todo']:

            rcfilename = 'jobs/runopt-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
            rc.write(rcfilename,jobrc)
            logging.debug('An rc-file was created (%s)' % rcfilename )

            # We first run the ygf optimization and directly do the forward runs as well

            runjobname = 'jobs/runopt-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
            jobopts={'jobname':'%s_%s'%(year.strip(),crop.strip().replace(' ','_')),
                     'jobqueue' : 'normal',
                     'jobtime' : '%s'%jobrc['time.job.limit'],
                     'joblog' : '%s'% runjobname.replace('jb','log') }
            header= pf.get_job_header(joboptions=jobopts)
            #header += 'rm -f /projects/0/ctdas/input/wofost/CABO_weather_ECMWF/*cache\n'
            header += 'export PYTHONPATH="./"\n'
            header += 'python py/carbon_cycle/_04_optimize_fgap.py rc=%s\n'%rcfilename
            pf.e_job(runjobname, header, '999')  
            if 'submit' in rcitems['steps.todo']:
                pf.submit_job(runjobname)

        # Then we gapfill

        if 'gapfill' in rcitems['steps.todo']:

            rcfilename = 'jobs/gapfill-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
            rc.write(rcfilename,jobrc)
            logging.debug('An rc-file was created (%s)' % rcfilename )
            runjobname = 'jobs/gapfill-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
            jobopts={'jobname':'%s_%s'%(year.strip(),crop.strip().replace(' ','_')),
                     'jobqueue' : 'short',
                     'jobtime' : '00:15:00',
                     'joblog' : '%s'% runjobname.replace('jb','log') }
            header= pf.get_job_header(joboptions=jobopts)
            header += 'export PYTHONPATH="./"\n'
            header += 'python py/carbon_cycle/gapfill.py rc=%s\n'%rcfilename
            pf.write_job(runjobname, header, '999')  
            if 'submit' in rcitems['steps.todo']:
                pf.submit_job(runjobname)

        # And finally we run the gapfilled NUTS regions forward, note that we have to use a different rc-file now

        if 'run-fwd' in rcitems['steps.todo']:

            rcfilename = 'jobs/runfwd-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
            rc.write(rcfilename,jobrc)
            logging.debug('An rc-file was created (%s)' % rcfilename )

            runjobname = 'jobs/runfwd-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
            jobopts={'jobname':'%s_%s'%(year.strip(),crop.strip().replace(' ','_')),
                     'jobqueue' : 'normal',
                     'jobtime' : '%s'%jobrc['time.job.limit'],
                     'joblog' : '%s'% runjobname.replace('jb','log') }
            header= pf.get_job_header(joboptions=jobopts)
            header += 'export PYTHONPATH="./"\n'
            header += 'python py/carbon_cycle/_05_run_forward_sim.py rc=%s\n'%rcfilename
            #header += 'python py/carbon_cycle/_06_complete_c_cycle.py rc=%s\n'%rcfilename
            pf.write_job(runjobname, header, '999')  
            if 'submit' in rcitems['steps.todo']:
                pf.submit_job(runjobname)

        if 'ccycle' in rcitems['steps.todo']:

            rcfilename = 'jobs/runccycle-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
            rc.write(rcfilename,jobrc)
            logging.debug('An rc-file was created (%s)' % rcfilename )

            runjobname = 'jobs/runccycle-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
            jobopts={'jobname':'%s_%s'%(year.strip(),crop.strip().replace(' ','_')),
                     'jobqueue' : 'normal',
                     'jobtime' : '%s'%('12:00:00'),
                     'joblog' : '%s'% runjobname.replace('jb','log') }
            header= pf.get_job_header(joboptions=jobopts)
            header += 'export PYTHONPATH="./"\n'
            header += 'python py/carbon_cycle/_06_complete_c_cycle.py rc=%s\n'%rcfilename
            pf.write_job(runjobname, header, '999')  
            if 'submit' in rcitems['steps.todo']:
                pf.submit_job(runjobname)

    if 'gridflux' in rcitems['steps.todo']:

        rcfilename = 'jobs/rungridflux_%s.rc'%(year.strip())
        rc.write(rcfilename,jobrc)
        logging.debug('An rc-file was created (%s)' % rcfilename )

        runjobname = 'jobs/rungridflux_%s.jb'%(year.strip())
        jobopts={'jobname':'gridflux_%s'%(year.strip()),
                 'jobqueue' : 'short',
                 'jobtime' : '%s'%('01:00:00'),
                 'joblog' : '%s'% runjobname.replace('jb','log') }
        header= pf.get_job_header(joboptions=jobopts)
        header += 'export PYTHONPATH="./"\n'
        header += 'python py/carbon_cycle/_07_grid_netcdf.py rc=%s\n'%rcfilename
        pf.write_job(runjobname, header, '999')  
        if 'submit' in rcitems['steps.todo']:
            pf.submit_job(runjobname)


#exit
sys.exit(0)

