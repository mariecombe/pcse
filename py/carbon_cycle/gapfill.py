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
import cPickle

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

    inputdir = rcF['dir.wofost.input']
    outputdir = rcF['dir.output']
    outputdir = os.path.join(outputdir,'ygf')

    EUROSTATdir = os.path.join(inputdir,'EUROSTATobs')

#-------------------------------------------------------------------------------
# we retrieve the NUTS regions years to loop over:
    NUTS_regions,crop_dict = select_crops_regions(crops,EUROSTATdir)

    for year in years:
        for crop in sorted(crop_dict.keys()):  # this will normally only give one crop, but let's leave it like this
            crop_name  = crop_dict[crop][1]
            logging.info( '\nCrop no %i: %s / %s'%(crop_dict[crop][0],crop, crop_name) )
            observed_nuts  = [f.split('_')[1] for f in os.listdir(outputdir) if "_observed" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts = [f.split('_')[1] for f in os.listdir(outputdir) if not "_observed" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code2  = [f.split('_')[1] for f in os.listdir(outputdir) if "gapfilled02" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code3  = [f.split('_')[1] for f in os.listdir(outputdir) if "gapfilled03" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code4  = [f.split('_')[1] for f in os.listdir(outputdir) if "gapfilled04" in f]  # code _01 refers to a ygf from observed values
            gapfilled_nuts_code5  = [f.split('_')[1] for f in os.listdir(outputdir) if "gapfilled05" in f]  # code _01 refers to a ygf from observed values
            nonculti  = [f.split('_')[1] for f in os.listdir(outputdir) if "noncultivated" in f]  # code _01 refers to a ygf from observed values
            tofill  = [f.split('_')[1] for f in os.listdir(outputdir) if "togapfill" in f]  # code _01 refers to a ygf from observed values

            for nut in sorted(NUTS_regions):
                # Check if this region was not already done
                if nut not in tofill: 
                    logging.debug('NUTS region %s does not need gapfilling, skipping' % nut)
                    continue # skippping, already done

                else: # these are all the files missing, or the code 4 and code 5 gap-filled regions

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

                    logging.info("%d years found for gap filling" %len(availfiles) )

                    if prevyear in availfiles and nextyear in availfiles:  # Use average from y-1 and y+1
                        logging.info("Choosing gapfill method 02")
                        optimi_info= cPickle.load(open(prevyear,'rb'))
                        ygf_prev        = optimi_info[2]
                        optimi_info= cPickle.load(open(nextyear,'rb'))
                        ygf_next        = optimi_info[2]
                        ygf = (ygf_prev+ygf_next)/2.0  # simply average
                        opt_code='gapfilled02'
                        shortlist_cells = optimi_info[3]
                    elif prevyear in availfiles: # Use previous years value
                        logging.info("Choosing gapfill method 03a")
                        optimi_info= cPickle.load(open(prevyear,'rb'))
                        ygf = optimi_info[2]
                        opt_code='gapfilled03'
                        shortlist_cells = optimi_info[3]
                        print shortlist_cells
                    elif nextyear in availfiles:  # Use nextyears value
                        logging.info("Choosing gapfill method 03b")
                        optimi_info= cPickle.load(open(nextyear,'rb'))
                        ygf = optimi_info[2]
                        opt_code='gapfilled03'
                        shortlist_cells = optimi_info[3]
                    elif len(availfiles) > 2:  # Use climatological average from other years if nyear > 2
                        logging.info("Choosing gapfill method 04")
                        ygf=0.0
                        for filename in availfiles:
                            optimi_info= cPickle.load(open(filename,'rb'))
                            ygf += optimi_info[2]
                        ygf = ygf/len(availfiles)
                        opt_code='gapfilled04'
                        shortlist_cells = optimi_info[3]
                    else:
                        logging.info("Choosing gapfill method 05")
                        fillyear =  os.path.join(outputdir,'ygf_%s_togapfill.pickle' % nut)
                        optimi_info= cPickle.load(open(fillyear,'rb'))
                        shortlist_cells = optimi_info[3]
                        ygf = 0.666
                        opt_code='gapfilled05'

                    logging.info("Using ygf of %5.2f and code of %s"%(ygf, opt_code))
                    currentyear = os.path.join(outputdir,'ygf_%s_%s.pickle' % (nut, opt_code) )
                    cPickle.dump([nut,opt_code,ygf,shortlist_cells],open(currentyear,'wb') )
                    _ = os.remove(os.path.join(outputdir,'ygf_%s_togapfill.pickle' % nut) )










if __name__ == "__main__":
    gapfill()

