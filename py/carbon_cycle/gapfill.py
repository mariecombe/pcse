#!/usr/bin/env python
# gapfill.py

"""
Author : peters 

Revision History:
File created on 28 Jan 2016.

"""

#
# Steps for algorithm:
#
# - For all years and for this crop
# - Find out which NUTS regions were not optimized due to a lack of observations (not cultivated area!), or which ones were previously gap-filled (code > 2)
# - For each unobserved NUTS region:
# -     find the preceeding observed year and copy its value OR 
# -     find the next observed year and copy its value OR
# -     find all available observed years and average their values
# -     assign a default value
# -     record code for method used (2,3,4,5) in output file
# - Note that successive runs should be able to UPDATE the status to a higher code, with a maximum of 2 (they can never be observed suddenly)

# import modules needed in all the methods of this script:
import sys, os, glob
sys.path.append('../')  # all py* code
sys.path.append('../../') # all pcse code
from py.tools.initexit import start_logger, parse_options
from py.carbon_cycle._01_select_crops_n_regions import select_crops_regions
import py.tools.rc as rc
import logging
import numpy as np
from datetime import datetime

def gapfill():

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
    outputdir = rcF['dir.output']
    outputdir = os.path.join(outputdir,'ygf')
    opt_type = rcF['optimize.type']
    if opt_type not in ['observed','gap-filled']:
        logging.error('The specified optimization type (%s) in the call argument is not recognized' % opt_type )
        logging.error('Please use either "observed" or "gap-filled" as value in the main rc-file')
        sys.exit(2)

#-------------------------------------------------------------------------------
# we retrieve the NUTS regions years to loop over:
    NUTS_regions,crop_dict = select_crops_regions(crops)

    for year in years:
        for crop in sorted(crop_dict.keys()):  # this will normally only give one crop, but let's leave it like this
            crop_name  = crop_dict[crop][1]
            logging.info( '\nCrop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name) )
            observed_nuts  = [f.split('_')[1] for f in os.listdir(outputdir) if "_observed" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts = [f.split('_')[1] for f in os.listdir(outputdir) if not "_observed" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code2  = [f.split('_')[1] for f in os.listdir(outputdir) if "_02" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code3  = [f.split('_')[1] for f in os.listdir(outputdir) if "_03" in f]  # code _01 refers to a ygf from observed values
            for nut in sorted(NUTS_regions)[0:10]:
                # Check if this region was not already done
                if nut in observed_nuts: 
                    logging.debug('NUTS region %s was already optimized based on observed yield, skipping' % nut)
                    continue # skippping, already done

                if nut in gapfilled_nuts_code2 or nut in gapfilled_nuts_code3: 
                    logging.debug('NUTS region %s was already gapfilled based on neighboring year, skipping' % nut)
                    continue # skipping, already done

                else: # these are all the files missing, or the code 3 and code 4 gap-filled regions

                    logging.debug('NUTS region %s will now be gapfilled ' % nut)

                    # Find if there is a previous year
                    prevyear = os.path.join(outputdir.replace('%04d'%year, '%04d'% (year-1 )),'ygf_%s_observed.pickle' % nut)
                    nextyear = os.path.join(outputdir.replace('%04d'%year, '%04d'% (year+1 )),'ygf_%s_observed.pickle' % nut)
                    availfiles = []
                    for yr in range(1995,2020):
                        searchyear =  os.path.join(outputdir.replace('%04d'%year, '%04d'% yr ),'ygf_%s_observed.pickle' % nut)
                        if os.path.exists(searchyear) :
                            availfiles.append(searchyear)
                        #else:
                            #logging.info("File not found: %s"%searchyear)
                    if not availfiles:
                        logging.info("No years found for gap filling, using default value")
                        ygf = 0.75
                    else:
                        logging.info("%d years found for gap filling" %len(availfiles) )










if __name__ == "__main__":
    gapfill()
