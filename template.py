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
logging.info("  Optimization type : %s"% rcitems['optimize.type'])

years = rcitems['time.years'].split(',')  # the years to cover
crops = rcitems['crop.types'].split(',')  # the crop types to consider
opt_type = rcitems['optimize.type']       # opt_type = 'observed' or gap-filled' referring to the source of the yield-gap factor
projectdir = rcitems['dir.project']

if opt_type not in ['observed','gapfilled']:
    logging.error('The specified optimization type (%s) in the rc-file is not recognized' % opt_type )
    logging.error('Please use either "observed" or "gapfilled" as value')
    sys.exit(2)

rundir = os.path.join(projectdir,'exec')
if not os.path.exists(rundir): os.makedirs(rundir)
outputdir = os.path.join(projectdir,'output')
if not os.path.exists(outputdir): os.makedirs(outputdir)

# Next open platform class

import py.platforms.capegrim as platform

pf = platform.CapeGrimPlatform()

# And create a loop over years and crop types

for year in years:
    for crop in crops:

        # create directory structure for optimized output per crop and year
       
        dirname = os.path.join(outputdir,'%s'%year.strip(),crop.strip().replace(' ','_') )  
        if not os.path.exists(dirname): 
            os.makedirs(dirname)
            logging.info('Created new folder: %s'%dirname)

        jobrc = {'year' : year.strip(),'crop' : crop.strip() , 'dir.output' : dirname, 'optimize.type': opt_type}
        filename = 'jobs/step1_%s_%s.rc'%(year.strip(),crop.strip().replace(' ','_') )
        rc.write(filename,jobrc)
        logging.info('An rc-file was created (%s)' % filename )

        # We first run the ygf optimization and directly do the forward runs as well

        runjobname = 'jobs/runopt-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        header= pf.get_job_header()
        header += 'python ../py/carbon_cycle/_04_optimize_fgap.py rc=%s\n'%os.path.split(filename)[-1]
        header += 'python ../py/carbon_cycle/_05_run_forward_sim.py rc=%s\n'%os.path.split(filename)[-1]
        pf.write_job(runjobname, header, '999')  

        # Then we gapfill

        runjobname = 'jobs/gapfill-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        header= pf.get_job_header()
        header += 'python ../py/carbon_cycle/gapfill.py rc=%s\n'%os.path.split(filename)[-1]
        pf.write_job(runjobname, header, '999')  

        # And finally we run the gapfilled NUTS regions forward

        runjobname = 'jobs/rungap-%s_%s.jb'%(year.strip(),crop.strip().replace(' ','_') )
        header= pf.get_job_header()
        header += 'python ../py/carbon_cycle/_05_run_forward_sim.py rc=%s\n'%os.path.split(filename)[-1]
        pf.write_job(runjobname, header, '999')  

#exit
sys.exit(0)

