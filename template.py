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

        jobrc = {'year' : year.strip(),'crop' : crop.strip() , 'dir.output' : dirname, 'optimize.type': 'observed'}
        rcfilename = 'jobs/runopt-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
        rc.write(rcfilename,jobrc)
        logging.debug('An rc-file was created (%s)' % rcfilename )

        # We first run the ygf optimization and directly do the forward runs as well

        runjobname = 'jobs/runopt-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        jobopts={'jobname':'%s_%s'%(year.strip(),crop.strip().replace(' ','_')),
                 'jobqueue' : 'normal',
                 'joblog' : '%s'% runjobname.replace('jb','log') }
        header= pf.get_job_header(joboptions=jobopts)
        header += 'python py/carbon_cycle/_04_optimize_fgap.py rc=%s\n'%rcfilename
        header += 'python py/carbon_cycle/_05_run_forward_sim.py rc=%s\n'%rcfilename
        pf.write_job(runjobname, header, '999')  
        pf.submit_job(runjobname)

        # Then we gapfill

        #runjobname = 'jobs/gapfill-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        #header= pf.get_job_header()
        #header += 'python ../py/carbon_cycle/gapfill.py rc=%s\n'%os.path.split(rcfilename)[-1]
        #pf.write_job(runjobname, header, '999')  

        # And finally we run the gapfilled NUTS regions forward, note that we have to use a different rc-file now

        #jobrc = {'year' : year.strip(),'crop' : crop.strip() , 'dir.output' : dirname, 'optimize.type': 'gapfilled'}
        #rcfilename = 'jobs/rungap-%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
        #rc.write(rcfilename,jobrc)
        #logging.debug('An rc-file was created (%s)' % rcfilename )

        #runjobname = 'jobs/rungap-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        #header= pf.get_job_header()
        #header += 'python ../py/carbon_cycle/_05_run_forward_sim.py rc=%s\n'%os.path.split(rcfilename)[-1]
        #pf.write_job(runjobname, header, '999')  

#exit
sys.exit(0)

