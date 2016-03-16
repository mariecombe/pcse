#!/usr/bin/env python

import sys, os
import numpy as np
from cPickle import load as pickle_load
from cPickle import dump as pickle_dump

# This script combine pickle files of the CGMS/ folder

#===============================================================================
def main():
#===============================================================================
# directory paths:
    CGMSdir = '/Users/mariecombe/mnt/promise/CO2/wofost/CGMS'
    codedir = '/Users/mariecombe/Cbalance/model_code'
#-------------------------------------------------------------------------------
# Temporarily add code directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, codedir) 
    sys.path.insert(0, os.path.join(codedir,'carbon_cycle')) 
#-------------------------------------------------------------------------------
    from pcse.db.cgms11 import TimerDataProvider, SoilDataIterator, \
                               CropDataProvider, STU_Suitability, \
                               SiteDataProvider, WeatherObsGridDataProvider
#-------------------------------------------------------------------------------
    years = np.arange(2000,2015,1)
    crop_nos  = [1,2,3,4,6,7,8,10,11,12,13] # crop ID numbers
    fourth_level = np.arange(2,20,1)
#-------------------------------------------------------------------------------
    for folder in ["gridlist_objects","soildata_objects"]:
        # only proceed if the master pickle file doesn't exist
        if os.path.exists(os.path.join(CGMSdir, 'CGMSsoil.pickle')) and \
           folder =='soildata_objects': continue
        if os.path.exists(os.path.join(CGMSdir, 'CGMSgrid.pickle')) and \
           folder =='gridlist_objects': continue

        # initialize dictionary
        pickle_dict = dict()

        # we list their content
        subdir = os.listdir(os.path.join(CGMSdir,folder))
        for name in subdir:
            if name.endswith('.pickle'):
                filepath = os.path.join(CGMSdir, folder, name)
                key = name.split('.pickle')[0]
                print filepath, key
                pickle_dict[key] = pickle_load(open(filepath,'rb'))

        # we store the dict in one master pickle file
        if (folder.startswith("soil")):
            pickle_dump(pickle_dict, open(os.path.join(CGMSdir, 'CGMSsoil.pickle'),'wb'))
        elif (folder.startswith("grid")):
            pickle_dump(pickle_dict, open(os.path.join(CGMSdir, 'CGMSgrid.pickle'),'wb'))

#-------------------------------------------------------------------------------
    for folder in ["cropdata_objects","timerdata_objects"]:
        # only proceed if the master pickle file doesn't exist
        if os.path.exists(os.path.join(CGMSdir, 'CGMScrop.pickle')) and \
           folder =='cropdata_objects': continue
        if os.path.exists(os.path.join(CGMSdir, 'CGMStimer.pickle')) and \
           folder =='timerdata_objects': continue

        # initialize dictionary
        pickle_dict = dict()

        # we list their immediate content (relevant for crop masks)
        subdir = os.listdir(os.path.join(CGMSdir,folder))
        for name in subdir:
            if name.endswith('.pickle'):
                filepath = os.path.join(CGMSdir, folder, name)
                key = name.split('.pickle')[0]
                print filepath, key
                pickle_dict[key] = pickle_load(open(filepath,'rb'))

        # we list their subfolders' content
        for year in years:
            for crop_no in crop_nos:
                subdir = os.listdir(os.path.join(CGMSdir,folder,'%i'%year,'c%i'%crop_no))
                for name in subdir:
                    if name.endswith('pickle'):
                        filepath = os.path.join(CGMSdir,folder,'%i'%year,'c%i'%crop_no,name)
                        key = name.split('.pickle')[0]
                        print filepath, key
                        pickle_dict[key] = pickle_load(open(filepath,'rb'))

        # we store the dict in one master pickle file
        if (folder.startswith("crop")):
            pickle_dump(pickle_dict, open(os.path.join(CGMSdir, 'CGMScrop.pickle'),'wb'))
        elif (folder.startswith("timer")):
            pickle_dump(pickle_dict, open(os.path.join(CGMSdir, 'CGMStimer.pickle'),'wb'))

#-------------------------------------------------------------------------------
    folder = "sitedata_objects"
    # only proceed if the master pickle file doesn't exist
    if not os.path.exists(os.path.join(CGMSdir, 'CGMSsite.pickle')):

        # initialize dictionary
        pickle_dict = dict()
 
        # we list their subfolders' content
        for year in years:
            for crop_no in crop_nos:
                for lev in fourth_level:
                    subdir = os.listdir(os.path.join(CGMSdir,folder,'%i'%year,'c%i'%crop_no, 'grid_%i'%lev))
                    for name in subdir:
                        if name.endswith('pickle'):
                            filepath = os.path.join(CGMSdir,folder,'%i'%year,'c%i'%crop_no,'grid_%i'%lev,name)
                            key = name.split('.pickle')[0]
                            print filepath, key
                            pickle_dict[key] = pickle_load(open(filepath,'rb'))
 
        # we store the dict in one master pickle file
        pickle_dump(pickle_dict, open(os.path.join(CGMSdir, 'CGMSsite.pickle'),'wb'))

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
