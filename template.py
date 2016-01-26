#!/usr/bin/env python
import sys
import os
import logging
import py.tools.rc as rc

# Append path so that modules can be imported
sys.path.append(os.getcwd())

# Set up the basic logging style, start logging from DEBUG and higher
logging.basicConfig(level=logging.DEBUG, 
                    format=' [%(levelname)-7s] (%(asctime)s) py-%(module)-20s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

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

years = rcitems['time.years'].split(',')
crops = rcitems['crop.types'].split(',')
print years[-1],crops[-1]

# Next open platform class

import py.platforms.capegrim as platform

pf = platform.CapeGrimPlatform()

# And create a loop over years and crop types

for year in years:
    for crop in crops:

        header= pf.get_job_header()
        pf.write_job('jobs/test_%s_%s.jb'%(year.strip(),crop.strip(),), header, '999')

#exit
sys.exit(0)

