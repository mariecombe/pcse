#!/usr/bin/env python

import sys, os
import numpy as np

# This script gathers a few useful general functions used by several scripts

#===============================================================================
# retrieve the crop fraction from EUROSTAT data
def get_EUR_frac_crop(crop_name, NUTS_name, EUROSTATdir_, 
                                                                prod_fig=False):
#===============================================================================

    import math
    from matplotlib import pyplot as plt

    name1       = 'agri_croparea_NUTS1-2-3_1975-2014.csv'
    name2       = 'agri_landuse_NUTS1-2-3_2000-2013.csv'
    NUTS_data   =  open_csv_EUROSTAT(EUROSTATdir_, [name1, name2],
                                     convert_to_float=True)
    # retrieve the region's crop area and arable area for the years 2000-2013
    # NB: by specifying obs_type='area', we do not remove a long term trend in
    # the observations 
    crop_area   = detrend_obs(2000, 2013, NUTS_name, crop_name,
                              NUTS_data[name1], 1., 2000, obs_type='area',
                              prod_fig=False)
    arable_area = detrend_obs(2000, 2013, NUTS_name, 'Arable land', 
                              NUTS_data[name2], 1., 2000, obs_type='area',
                              prod_fig=False)
    # we calculate frac_crop for the years where we both have observations of 
    # arable land AND crop area
    frac_crop = np.array([])
    years     = np.array([])
    for year in range(2000, 2015):
        if ((year in crop_area[1]) and (year in arable_area[1])):
            indx_crop = np.array([math.pow(j,2) for j in (crop_area[1]-year)]).argmin()  
            indx_arab = np.array([math.pow(j,2) for j in (arable_area[1]-year)]).argmin()
            frac_crop = np.append(frac_crop, crop_area[0][indx_crop] / \
                                     arable_area[0][indx_arab])
            years     = np.append(years, year)
        else:
            years     = np.append(years, np.nan)
            frac_crop = np.append(frac_crop, np.nan)

    # for years without data, we fit a linear trend line in the record of 
    # observed frac
    all_years = range(2000, 2015)
    mask = ~np.isnan(np.array(years))
    z = np.polyfit(years[mask], frac_crop[mask], 1)
    p = np.poly1d(z)
    for y,year in enumerate(years):
        if np.isnan(year):
            frac_crop[y] = p(all_years[y])

    # return the yearly ratio of crop / arable land, and the list of years 
    return frac_crop, all_years

#===============================================================================
# Select the list of years over which we optimize the yldgapf
def define_opti_years(opti_year_, obs_years):
#===============================================================================

    if opti_year_ in obs_years:
        # if the year is available in the record of observed yields, we
        # use that particular year to optimize the yldgapf
        print '\nWe use the available yield observations from', opti_year_
        opti_years = [opti_year_]
    else:
        # if we don't have yield observations for that year, we use the
        # most recent 3 years of observations available to optimize the 
        # yldgapf. The generated yldgapf will be used as proxy of the 
        # opti_year yldgapf
        opti_years = find_consecutive_years(obs_years, 3)
        print '\nWe use', opti_years, 'as proxy for', opti_year

    return opti_years

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
    elif (obs_type == 'area'):
        header_to_search = 'Area (1 000 ha)'
        conversion_factor = 1.
        obs_unit = '1000 ha'
    
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
    
    if ((obs_type == 'yield') or (obs_type == 'harvest')):
        print '\nsuccesfully detrended the dry matter %ss!'%obs_type
    elif (obs_type == 'area'): 
        print '\nno %ss to detrend! returning the original obs'%obs_type
   
    if ((obs_type == 'yield') or (obs_type == 'harvest')):
        obstoreturn = OBS['DETRENDED']
    elif (obs_type == 'area'): 
        obstoreturn = OBS['ORIGINAL']

    return obstoreturn, campaign_years[mask]

#===============================================================================
# Function to retrieve the dry matter content of a given crop in a given
# country (this is based on EUROSTAT crop humidity data)
def retrieve_crop_DM_content(crop_no_, NUTS_no_):
#===============================================================================

    from cPickle import load as pickle_load

	# directories on my local MacBook:
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'
    # directories on capegrim:
    #EUROSTATdir   = "/Users/mariecombe/Cbalance/EUROSTAT_data"

    # we retrieve the dry matter of a specific crop and country, over the 
    # years 1955-2015
    DM_obs = pickle_load(open(os.path.join(EUROSTATdir, 
                         'EUROSTAT_obs_crop_humidity.pickle'), 'rb'))
    DM_content = DM_obs[crop_no_][NUTS_no_[0:2]]
    # if the retrieved array is not empty, then we use the average 
    # reported DM:
    if (np.isnan(DM_content).all() == False):
        DM_content = np.ma.mean(np.ma.masked_invalid(DM_content))
        print '\nWe use the observed DM content', DM_content
    # otherwise we use the standard EUROSTAT DM content for that crop:
    else:
        DM_standard = pickle_load(open(os.path.join(EUROSTATdir, 
                                'EUROSTAT_standard_crop_humidity.pickle'),'rb'))
        DM_content = DM_standard[crop_no_]
        print '\nWe use the standard DM content', DM_content

    return DM_content

#===============================================================================
# Function to open normal csv files
def open_pcse_csv_output(inpath,filelist):
#===============================================================================

    from csv import reader as csv_reader
    from datetime import date
    from string import split

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
        headerow=lines[18]

        # deleting rows that are not data (first and last rows of the file)
        del lines[0:19]

        # transforming data from string to float type
        converted_data=[]
        for line in lines:
            datestr = split(line[0], '-')
            a = [date(int(datestr[0]),int(datestr[1]),int(datestr[2])), \
                 float(line[1]), float(line[2]), float(line[3]), float(line[4]), \
                 float(line[5]), float(line[6]), float(line[7]), float(line[8]), \
                 float(line[9]), float(line[10]), float(line[11])]
            converted_data.append(a)
        data = np.array(converted_data)

        # creating one dictionnary and storing the float data in it
        dictnamelist= {}
        for j,varname in enumerate(headerow):
            dictnamelist[varname]=data[:,j]
        Dict[namefile] = dictnamelist
    
        print "Dictionary created!"

    return Dict

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

