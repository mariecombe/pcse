#!/usr/bin/env python

import sys, os
import numpy as np

# This script gathers a few useful general functions used by several scripts

#===============================================================================
# Return a list of consecutive years longer than n items
def find_consecutive_years(years, nyears):
#===============================================================================

    # Split the list of years where there are gaps
    years = map(int, years) # convert years to integers
    split_years = np.split(years, np.where(np.diff(years) > 1)[0]+1)

    # Return the most recent group of years that contains at least nyears items
    consecutive_years = np.array([])
    for subset in split_years[::-1]: # [::-1] reverses the array without 
                                     # modifying it permanently
        if (len(subset) >= nyears):
            consecutive_years = np.append(consecutive_years, subset)
            break
        else:
            pass

    # return the last nyears years of the most recent group of years
    return consecutive_years[-nyears:len(consecutive_years)]

#===============================================================================
# Function that will fetch the crop name from the crop number. Returns a list
# with two items: first is the CGMS crop name, second is the EUROSTAT crop name
def get_crop_name(list_of_CGMS_crop_no):
#===============================================================================

    list_of_crops_EUR = ['Barley','Beans','Common spring wheat',
                        'Common winter wheat','Grain maize and corn-cob-mix',
                        'Green maize',
                        'Potatoes (including early potatoes and seed potatoes)',
                        'Rye','Spring rape','Sugar beet (excluding seed)',
                        'Sunflower seed','Winter barley','Winter rape']
    list_of_crops_CGMS = ['Spring barley','Field beans','Spring wheat',
                          'Winter wheat','Grain maize','Fodder maize',
                          'Potato','Rye','Spring rapeseed','Sugar beets',
                          'Sunflower','Winter barley','Winter rapeseed']
    list_of_crop_ids_CGMS = [3,8,'sw',1,2,12,7,4,'sr',6,11,13,10]

    dict_names = dict()
    for item in list_of_CGMS_crop_no:
        crop_name = list(['',''])
        for i,crop_id in enumerate(list_of_crop_ids_CGMS):
            if (crop_id == item):
                crop_name[0] = list_of_crops_CGMS[i]
                crop_name[1] = list_of_crops_EUR[i]
        dict_names[item]=crop_name

    return dict_names

#===============================================================================
#def fetch_EUROSTAT_NUTS_name(NUTS_no):
#===============================================================================
    
#    print 'Not yet coded!'
#    return None

#===============================================================================
# Function to detrend the observed EUROSTAT yields or harvests
def detrend_obs( _start_year, _end_year, _NUTS_name, _crop_name, 
                uncorrected_yields_dict, _DM_content, base_year,
                obs_type='yield', prod_fig=False):
#===============================================================================

    from matplotlib import pyplot as plt

    nb_years = int(_end_year - _start_year + 1.)
    campaign_years = np.linspace(int(_start_year), int(_end_year), nb_years)
    OBS = {}
    TREND = {}
    
    # search for the index of base_year item in the campaign_years array
    for i,val in enumerate(campaign_years): 
        if val == base_year:
            indref = i

    # select if to detrend yields or harvest:
    if (obs_type == 'yield'):
        header_to_search = 'Yields (100 kg/ha)'
        conversion_factor = 100.*_DM_content
        obs_unit = 'kgDM ha-1'
    elif (obs_type == 'harvest'):
        header_to_search = 'Harvested production (1000 t)'
        conversion_factor = _DM_content
        obs_unit = '1000 tDM'
    
    # select yields for the required region, crop and period of time
    # and convert them from kg_humid_matter/ha to kg_dry_matter/ha 
    TARGET = np.array([0.]*nb_years)
    for j,year in enumerate(campaign_years):
        for i,region in enumerate(uncorrected_yields_dict['GEO']):
            if region.startswith(_NUTS_name[0:12]):
                if uncorrected_yields_dict['CROP_PRO'][i]==_crop_name:
                    if (uncorrected_yields_dict['TIME'][i]==str(int(year))):
                        if (uncorrected_yields_dict['STRUCPRO'][i]==
                                                      header_to_search):
                            TARGET[j] = float(uncorrected_yields_dict['Value'][i])\
                                              *conversion_factor
    #print 'observed dry matter yields:', TARGET

    # fit a linear trend line in the record of observed yields
    mask = ~np.isnan(TARGET)
    z = np.polyfit(campaign_years[mask], TARGET[mask], 1)
    p = np.poly1d(z)
    OBS['ORIGINAL'] = TARGET[mask]
    TREND['ORIGINAL'] = p(campaign_years)
    
    # calculate the anomalies to the trend line
    ANOM = TARGET - (z[0]*campaign_years + z[1])
    
    # Detrend the observed yield data
    OBS['DETRENDED'] = ANOM[mask] + p(base_year)
    z2 = np.polyfit(campaign_years[mask], OBS['DETRENDED'], 1)
    p2 = np.poly1d(z2)
    TREND['DETRENDED'] = p2(campaign_years)
    
    # if needed plot a figure showing the yields before and after de-trending
    if prod_fig==True:
        plt.close('all')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
        for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
            ax.scatter(campaign_years[mask], OBS[var], c='b')
       	    ax.plot(campaign_years,TREND[var],'r-')
       	    ax.set_ylabel('%s %s (%s)'%(var,obs_type,obs_unit), fontsize=14)
            ax.set_xlabel('time (year)', fontsize=14)
        fig.savefig('observed_%ss.png'%obs_type)
        print 'the trend line is y=%.6fx+(%.6f)'%(z[0],z[1])
        plt.show()
    
    print '\nsuccesfully detrended the dry matter %ss!'%obs_type
    
    return OBS['DETRENDED'], campaign_years[mask]

#===============================================================================
# Function to open normal csv files
def open_csv(inpath,filelist,convert_to_float=False):
#===============================================================================

    from csv import reader as csv_reader

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

        # transforming data from string to float type
        converted_data=[]
        for line in lines:
            converted_data.append(map(float,line))
        data = np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        print "Dictionary created!"

    return Dict

#===============================================================================
# Function to open EUROSTAT csv files
def open_csv_EUROSTAT(inpath,filelist,convert_to_float=False):
#===============================================================================

    from csv import reader as csv_reader
    from string import replace as string_replace

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

