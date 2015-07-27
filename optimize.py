#!/usr/bin/env python

#import pcse
from pcse.db.cgms11 import WeatherObsGridDataProvider, TimerDataProvider
from pcse.db.cgms11 import SoilDataIterator, CropDataProvider, STU_Suitability
from pcse.db.cgms11 import SiteDataProvider
from pcse.models import Wofost71_WLP_FD
from pcse.util import is_a_dekad
from pcse.base_classes import WeatherDataProvider
#import matplotlib.pyplot as plt
#import sqlalchemy as sa  #only required to connect to the oracle database
from sqlalchemy import MetaData, select, Table
#import pandas as pd

from datetime import date
#import math
#import sys, os, getopt
#import csv
#import string
#import numpy as np
#from matplotlib import pyplot
#from scipy import ma

#------------------------------------------------------------------------
def main:

    try:                                
        opts, args = getopt.getopt(sys.argv[1:], "-h")
    except getopt.GetoptError:           
        print "Error"
        sys.exit(2)      
    
    for options in opts:
        options=options[0].lower()
        if options == '-h':
            helptext = """
    This script execute the WOFOST runs for one location

                """
            
            print helptext
            
            sys.exit(2)     
 
#-------------------------------------------------------------------------------
    # Define directories

    currentdir = os.getcwd()
    #EUROSTATdir = '/Users/mariecombe/Documents/Work/Research project 3/EUROSTAT_data'
    EUROSTATdir = "/Users/mariecombe/Cbalance/EUROSTAT_data"
    #folderpickle = 'pickled_CGMS_input_data/'
    folderpickle = '/Storage/CO2/mariecombe/pickled_CGMS_input_data/'

    # we establish a connection to the remote Oracle database
    
#    user = "cgms12eu_select"
#    password = "OnlySelect"
#    tns = "EURDAS.WORLD"
#    # tns = IP_address:PORT/SID
#    dsn = "oracle+cx_oracle://{user}:{pw}@{tns}".format(user=user, pw=password, tns=tns)
#    engine = sa.create_engine(dsn)
#    print engine
    
    # we retrieve observed datasets
    
    NUTS_data = open_eurostat_csv(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'])
    NUTS_ids = open_csv_as_strings(EUROSTATdir,['NUTS_codes_2013.csv'])
    # simplifying the dictionaries keys:
    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    del NUTS_ids['NUTS_codes_2013.csv']
    
    
    

#----------------------------------------------------------------------
def open_eurostat_csv(inpath,filelist):

    Dict = {}

    for i,namefile in enumerate(filelist):
         
        print "Opening %s"%(namefile)

        # open file, read all lines
        inputpath = os.path.join(inpath,namefile)
        f=open(inputpath,'rU') 
        reader=csv.reader(f, delimiter=',', skipinitialspace=True)
        lines=[]
        for row in reader:
            lines.append(row)
        f.close()

        # storing headers in list headerow
        headerow=lines[0]
        print headerow

        # deleting rows that are not data (first and last rows of the file)
        del lines[0]
        #del lines[-1]

        # transforming data from string to float type, storing it in array 'data'
        converted_data=[]
        for line in lines:
            if (line[4] != ':'): 
                a = line[4:4] + [float(string.replace(line[4], ' ', ''))] + [line[5]]
            else:
                a = line[0:-8] + [float('NaN')] + [line[5]]
            converted_data.append(a)
        data=np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(marie___):
            dictnamelist[varname]=data[:,j]
        
        Dict[namefile] = dictnamelist
    
        print "Dictionary created!\n"

    return Dict

#----------------------------------------------------------------------

#===============================================================================
if __name__=='__main__':
  main()
#===============================================================================
