#!/usr/bin/env python

import sys, os
import numpy as np
from csv import reader as csv_reader
from cPickle import dump as pickle_dump
from cPickle import load as pickle_load
from string import replace as string_replace
from datetime import datetime
from matplotlib import pyplot as plt


#===============================================================================
# This script reads the observed EUROSTAT dry matter content of crops
def main():
#===============================================================================

#-------------------------------------------------------------------------------
# Define working directories

    currentdir    = os.getcwd()
	
	# directories on my local MacBook:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'
    folderpickle  = '/Users/mariecombe/Documents/Work/Research_project_3/'\
                   +'pcse/pickled_CGMS_input_data/'

    # directories on capegrim:
    #folderpickle  = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"


#-------------------------------------------------------------------------------
# We retrieve observed EUROSTAT dry matter information

    filename = 'EUROSTAT_obs_crop_humidity.pickle'

    if (os.path.exists(os.path.join(currentdir,'output_data',filename))):
        DM = pickle_load(open(os.path.join(currentdir,'output_data',filename), 
                         'rb'))
    else:
        # open the EUROSTAT file containing the dry matter info
        NUTS_data = open_csv(EUROSTATdir,['agri_prodhumid_NUTS1_1955-2015.csv'],
                             convert_to_float=True)

        # simplify the dictionaries keys:
        NUTS_data['humidity'] = NUTS_data['agri_prodhumid_NUTS1_1955-2015.csv']
        del NUTS_data['agri_prodhumid_NUTS1_1955-2015.csv']
        
        DM = pickle_DM(NUTS_data['humidity'])

#-------------------------------------------------------------------------------
# We can plot a time series of one crop x country DM combination if needed

    time_series = plot_DM_time_series(DM, list_of_years, 'Spain', 'Barley')

#-------------------------------------------------------------------------------
# We finalize the crop DM content per country, either as the observed value, or
# if no observations are available, as the standard EUROSTAT DM content

    # we build a dictionnary of standard EU humidity content
#    for i,cropid in enumerate(list_of_crop_ids_CGMS):
#        standard_EU_DM = dict()
#        if (cropid == ):
#            standard_EU_DM[cropid] = 0.
    
	# we check the observed DM dictionnary. If there are observations, we
	# calculate the mean. If there are no observations, we take the standard DM
	# content.

#===============================================================================
# Function to retrieve EUROSTAT humidity content and to store it in pickle file
def pickle_DM(obs_data):
#===============================================================================

    # EUROSTAT lists of crop names and country names
    list_of_crops_EUR     = sorted(set(obs_data['CROP_PRO']))
    list_of_countries_EUR = sorted(set(obs_data['GEO']))
    list_of_years         = sorted(set(obs_data['TIME']))
    # we overwrite the list of EUROSTAT crops for a shorter one (not all 
    # crops are needed)
    list_of_crops_EUR     = ['Barley','Beans','Common spring wheat',
					    	 'Common winter wheat',
                             'Grain maize and corn-cob-mix',
                             'Green maize',
                             'Potatoes (including early potatoes and seed potatoes)',
                             'Rye','Spring rape',
                             'Sugar beet (excluding seed)','Sunflower seed',
                             'Winter barley',
                             'Winter rape']
    # we translate the EUROSTAT list of countries into official country codes
    # European & ISO 3166 nomenclature + temporary country code for Kosovo XK
    list_of_countries_CGMS = ['AL','AT','BE','BA','BG','HR','CY','CZ','DK',
                              'EE','EU','FI','MK','FR','DE','EL','HU','IS',
                              'IE','IT','XK','LV','LI','LT','LU','MT','ME',
                              'NL','NO','PL','PT','RO','RS','SK','SI','ES',
                              'SE','CH','TR','UK']
    # we translate the EUROSTAT shortlist of crop names into the CGMS ones
    list_of_crops_CGMS     = ['Spring barley','Field beans','Spring wheat',
                              'Winter wheat','Grain maize','Fodder maize',
                              'Potato','Rye','Spring rapeseed','Sugar beets',
                              'Sunflower','Winter barley','Winter rapeseed']
	# we translate the CGMS crop names into CGMS crop numbers. NB: two crop
	# numbers are unknown: the one of spring wheat and the one of spring
	# rapeseed
    list_of_crop_ids_CGMS  = [3,8,'sw',1,2,12,7,4,'sr',6,11,13,10]

    # we print out some info for the user
    print '\nWe will loop over:\n'
    print list_of_crops_CGMS
    print list_of_countries_CGMS
    print '\nStarting the loop at timestamp:', datetime.utcnow(), '\n'

	# we build a dictionnary where we will store the dry matter content per
	# crop and country CGMS ids

    DM = dict()
    for n,crop in enumerate(list_of_crop_ids_CGMS):
        DM[crop] = dict()
        print list_of_crops_CGMS[n]
        for k, idc in enumerate(list_of_countries_CGMS):
            print '    ', idc
            DM[crop][idc] = np.zeros(len(list_of_years))
            # we look for the DM content of a specific crop-country combination
            for y, year in enumerate(list_of_years):
                for i,time in enumerate(obs_data['TIME']):
                    if ((time == year)
                    and (obs_data['GEO'][i] == list_of_countries_EUR[k])
                    and (obs_data['CROP_PRO'][i] == list_of_crops_EUR[n])):
                        DM[crop][idc][y] = obs_data['Value'][i] 

    # we pickle the generated dictionnary containing the dry matter content
    filename = 'EUROSTAT_obs_crop_humidity.pickle'
    pickle_dump(DM, open(os.path.join(currentdir,'output_data',filename), 'wb'))

    # we print out a timestamp
    print '\nRetrieval over! Finished the loop at timestamp:', datetime.utcnow()

    return DM

#===============================================================================
# Function to plot a time series of crop humidity
def plot_DM_time_series(DM_dict, list_of_years_, crop, country):
#===============================================================================
    
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
    fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
    ax.plot(list_of_years_, DM_dict[crop][country], c='k')
    ax.set_ylabel('Humidity (-)', fontsize=14)
    ax.set_xlabel('time (year)', fontsize=14)
    fig.savefig('DM_time_series_%s_%s.png'%(country,crop))
    #plt.show()

    return None

#===============================================================================
# Function to open EUROSTAT csv files
def open_csv(inpath,filelist,convert_to_float=False):
#===============================================================================

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        print "\nOpening %s......"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv_reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[0]

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]

        # two possibilities: either convert data from string to float or
        # keep it as is in string type
        if (convert_to_float == True):
            # transforming data from string to float type
            converted_data=[]
            for line in lines:
                if (line[4] != ':'): 
                    a = (line[0:4] + [float(string_replace(line[4], ' ', ''))] 
                                   + [line[5]])
                else:
                    a = line[0:4] + [float('NaN')] + [line[5]]
                converted_data.append(a)
            data = np.array(converted_data)
        else:
            # we keep the string format, we just separate the string items
            datafloat=[]
            for row in lines:
                a = row[0:2]
                datafloat.append(a)
            data=np.array(datafloat)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        print "Dictionary created!"

    return Dict

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
