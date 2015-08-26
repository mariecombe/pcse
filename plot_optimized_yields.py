#!/usr/bin/env python

import os
import numpy as np
from matplotlib import pyplot as plt
from csv import reader as csv_reader
from cPickle import load as pickle_load
from string import replace as string_replace

#===============================================================================
# This script plots optimum yields gap factors obatined with the sensitivity
# analysis
def main():
#===============================================================================
    
# Gather the results of the sensitivity analysis:

    res_yldgapf           = dict() ; nb_top_combi         = dict()
    # For region ES41 = Castilla y Leon
    res_yldgapf['ES41']   = [0.578, 0.609, 0.594, 0.625] # +/- 0.016
    nb_top_combi['ES41']  = [3, 5, 10, 165]
    # For region ES43 = Extremadura
    res_yldgapf['ES43']   = [0.828, 0.781, 0.813, 0.75]
    nb_top_combi['ES43']  = [3, 5, 10, 53]


    # Random selections
    res2_yldgapf          = dict() ; nb_rand_combi        = dict()
    # For region ES41 = Castilla y Leon
    res2_yldgapf['ES41']  = [0.609, 0.703, 0.641, 0.625] # +/- 0.016
    nb_rand_combi['ES41'] = [1, 2, 3, 165] # 1,2,3 is just to separate 
                                                   # the points
    # For region ES43 = Extremadura
    res2_yldgapf['ES43']  = [0.75, 0.766, 0.797, 0.75]
    nb_rand_combi['ES43'] = [1, 2, 3, 53]

    # retrieve all the RMSE results from the sensitivity analysis
    RMSE      = dict()
    O_YLDGAPF = dict()
    list_of_regions = ['ES41', 'ES43']
    list_of_cases   = ['all', 'topn_10gx10s', 'topn_5gx5s', 'topn_3gx3s',
                       'randomn1', 'randomn2', 'randomn3']
    list_of_labels  = ['all', 'top 10', 'top 5', 'top 3', 'rand1', 'rand2', 
                       'rand3'] 
    for region in list_of_regions:
        RMSE[region]      = dict()
        O_YLDGAPF[region] = dict() 
        for case in list_of_cases:
            filepath = [f for f in os.listdir( os.path.join(currentdir,
               'output_data') ) if ( os.path.isfile(f) and f.startswith('RMSE_')
                and (region in f) and (case in f) ) ]
            RMSE[region][case] = pickle_load(open(filepath, 'rb'))
            # we sort the list of tuples by value of RMSE, and retrieve the 
            # yldgapf corresponding to the min RMSE:
            O_YLDGAPF[region][case] = sorted(RMSE[region][case], 
                                             key = operator_itemgetter(1))[0][0]

    # Plot the RMSE = f(YLDGAPF) graph for each region x crop combination
    for region in list_of_regions:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
        fig.subplots_adjust(0.15,0.16,0.95,0.96,0.4,0.)
        for i,case in enumerate(list_of_cases):
            x,y = zip(*RMSE[region][case])
            ax.plot(x, y, label=list_of_labels[i])
            ax.set_xlabel('yldgapf (-)')
            ax.set_ylabel('RMSE')
        legend(loc='best')
        fig.savefig('RMSE_'+region+'.png')
    

#-------------------------------------------------------------------------------
# Plot the results of the sensitivity analysis

    plt.close('all')
    fig1 = plot_sensitivity_yldgapf_to_top_combi(res_yldgapf, nb_top_combi)
    fig2 = plot_sensitivity_yldgapf_to_rand_combi(res2_yldgapf, nb_rand_combi)

#-------------------------------------------------------------------------------
# Analyze the forward simulations output data to calculate the regional yields

    yldgapf = dict()
    yldgapf['ES41']=0.609
    yldgapf['ES43']=0.781
    NUTS_name = dict()
    NUTS_name['ES41'] = 'Castilla y Leon'
    NUTS_name['ES43'] = 'Extremadura'
    crop_name = 'Barley'
    DM_content    = 0.9      # EUROSTAT dry matter fraction
                             # should be read from file rather than hardcoded
    start_year    = 2000     # start_year and end_year define the period of time
    end_year      = 2014     # for which we do forward simulations
    nb_years      = int(end_year - start_year + 1.)
    campaign_years = np.linspace(int(start_year),int(end_year),nb_years)

    # We define directories
    EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
				   +'EUROSTAT_data'
    out_data_dir  = '/Users/mariecombe/Documents/Work/Research_project_3/pcse/'\
                   +'output_data'

    # Retrieve the observational dataset:
    NUTS_data   = open_csv_EUROSTAT(EUROSTATdir,['agri_yields_NUTS1-2-3_1975-2014.csv'],
                             convert_to_float=True)
    NUTS_ids    = open_csv_EUROSTAT(EUROSTATdir,['NUTS_codes_2013.csv'],
                             convert_to_float=False)
    # we simplify the dictionaries keys:
    NUTS_data['yields'] = NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    NUTS_ids['codes']   = NUTS_ids['NUTS_codes_2013.csv']
    del NUTS_data['agri_yields_NUTS1-2-3_1975-2014.csv']
    del NUTS_ids['NUTS_codes_2013.csv']
    
    # detrend the yield observations
    detrend = dict()
    detrend['ES41']   = detrend_obs_yields(start_year, end_year, NUTS_name['ES41'], crop_name,
	    						   NUTS_data['yields'], DM_content, 2000)
    detrend['ES43']   = detrend_obs_yields(start_year, end_year, NUTS_name['ES43'], crop_name,
	    						   NUTS_data['yields'], DM_content, 2000)
    
    # Retrieve the simulation datasets:
    Sim_data_O         = open_csv(out_data_dir, 
                         ['ForwardSim_Optimized_crop3_regionES43_2000-2014.dat',
                          'ForwardSim_Optimized_crop3_regionES41_2000-2014.dat'])
    Sim_data_NO        = open_csv(out_data_dir, 
                         ['ForwardSim_Non-Optimized_crop3_regionES43_2000-2014.dat',
                          'ForwardSim_Non-Optimized_crop3_regionES41_2000-2014.dat'])
    # we simplify the dictionaries keys:
    Sim_data_O['ES41'] = Sim_data_O['ForwardSim_Optimized_crop3_regionES41_2000-2014.dat']
    Sim_data_O['ES43'] = Sim_data_O['ForwardSim_Optimized_crop3_regionES43_2000-2014.dat']
    Sim_data_NO['ES41'] = Sim_data_NO['ForwardSim_Non-Optimized_crop3_regionES41_2000-2014.dat']
    Sim_data_NO['ES43'] = Sim_data_NO['ForwardSim_Non-Optimized_crop3_regionES43_2000-2014.dat']

    # aggregate into regional yields
    sim_o = dict() ; sim_no = dict()
    sim_o['ES41'] = calc_regional_yields(Sim_data_O['ES41'], campaign_years)
    sim_o['ES43'] = calc_regional_yields(Sim_data_O['ES43'], campaign_years)
    sim_no['ES41'] = calc_regional_yields(Sim_data_NO['ES41'], campaign_years)
    sim_no['ES43'] = calc_regional_yields(Sim_data_NO['ES43'], campaign_years)

    # retrieve the opt sim yields & aggregate into regional opt yields
    # plot the sim regional yields: non-opt and opt.
    # add time-frame to show years used for optimization.

    fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize=(7,6))
    fig.subplots_adjust(0.13,0.2,0.95,0.92,0.4,0.6)
    for i, region, ax in zip([0.,1.], ['ES41','ES43'], axes.flatten()):
        ax.scatter(campaign_years, sim_no[region], c='b', marker='v', s=50,
                   label='non-optimized sims')
        ax.scatter(campaign_years, sim_o[region], c='k', marker='o', s=50, 
                   label='optimized sims')
        ax.scatter(detrend[region][1],detrend[region][0], c='r', marker='+', 
                   linewidth=2, s=70, label='detrended obs')
        ax.set_ylim(0.,max(sim_no[region])+500.)
        ax.set_ylabel(r'yield (kg$_{DM}$ ha$^{-1}$)')
        ax.set_xlabel('time (year)')
        ax.set_title('NUTS region '+region+' ('+NUTS_name[region]+')')
        ax.axvspan(2003.5,2006.5,color='grey',alpha=0.3)
        ax.annotate('yldgapf=%.3f'%yldgapf[region], xy=(0.89,0.1), 
                    xycoords='axes fraction',horizontalalignment='center',
                    verticalalignment='center')
        if (i==1.): ax.legend(loc='best', ncol=3, fontsize=12, bbox_to_anchor = (1.05,-0.4))
    fig.savefig('time_series_yields.png')


    plt.show()

#===============================================================================
# Function to open normal csv files
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
# Function to detrend the observed yields
def detrend_obs_yields( _start_year, _end_year, _NUTS_name, _crop_name, 
                       uncorrected_yields_dict, _DM_content, base_year, 
                       prod_fig=False):
#===============================================================================

    nb_years = int(_end_year - _start_year + 1.)
    campaign_years = np.linspace(int(_start_year), int(_end_year), nb_years)
    OBS = {}
    TREND = {}
    
    # search for the index of base_year item in the campaign_years array
    for i,val in enumerate(campaign_years): 
        if val == base_year:
            indref = i
    
    # select yields for the required region, crop and period of time
    # and convert them from kg_humid_matter/ha to kg_dry_matter/ha 
    TARGET = np.array([0.]*nb_years)
    for j,year in enumerate(campaign_years):
        for i,region in enumerate(uncorrected_yields_dict['GEO']):
            if region.startswith(_NUTS_name[0:12]):
                if uncorrected_yields_dict['CROP_PRO'][i]==_crop_name:
                    if (uncorrected_yields_dict['TIME'][i]==str(int(year))):
                        if (uncorrected_yields_dict['STRUCPRO'][i]==
                                                      'Yields (100 kg/ha)'):
                            TARGET[j] = float(uncorrected_yields_dict['Value'][i])\
                                              *100.*_DM_content
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
        pyplot.close('all')
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10,8))
        fig.subplots_adjust(0.15,0.16,0.85,0.96,0.4,0.)
        for var, ax in zip(["ORIGINAL", "DETRENDED"], axes.flatten()):
            ax.scatter(campaign_years[mask], OBS[var], c='b')
       	    ax.plot(campaign_years,TREND[var],'r-')
       	    ax.set_ylabel('%s yield (gDM m-2)'%var, fontsize=14)
            ax.set_xlabel('time (year)', fontsize=14)
        fig.savefig('observed_yields.png')
        print 'the trend line is y=%.6fx+(%.6f)'%(z[0],z[1])
        pyplot.show()
    
    #print 'detrended dry matter yields:', OBS['DETRENDED']
    
    return OBS['DETRENDED'], campaign_years[mask]

#===============================================================================
def plot_sensitivity_yldgapf_to_rand_combi(optimum_yldgapf, nb_combi):
#===============================================================================

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(7,3))
    fig.subplots_adjust(0.13,0.17,0.95,0.9,0.4,0.05)
    d = 0.03  # size of the diagonal line in axes coordinates
    for i, var, ax in zip([0.,0.,1.,1.], ['ES41', 'ES43','ES41', 'ES43'], 
                          axes.flatten()):
    # Both top and bottom:
        ax.set_yticks([1.,2.,3.,nb_combi[var][3]])
        ax.errorbar(optimum_yldgapf[var][0], nb_combi[var][0], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][0], nb_combi[var][0], c='b', s=70,
                   marker='v')
        ax.errorbar(optimum_yldgapf[var][1], nb_combi[var][1], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][1], nb_combi[var][1], c='r', s=70,
                   marker='^')
        ax.errorbar(optimum_yldgapf[var][2], nb_combi[var][2], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][2], nb_combi[var][2], c='g', s=70, 
                   marker='p')
        ax.errorbar(optimum_yldgapf[var][3], nb_combi[var][3], xerr=0.016,
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][3], nb_combi[var][3], c='k', s=50, 
                   marker='s')
        labels = [item.get_text() for item in ax.get_yticklabels()] 
        labels[0] = 'rand10 no1'
        labels[1] = 'rand10 no2'
        labels[2] = 'rand10 no3'
        labels[3] = 'all'
        ax.set_yticklabels(labels)
    # Bottom of the graph:
        if i==1.: 
            ax.set_ylim(0.,4.)
            ax.spines['top'].set_visible(False)
            ax.xaxis.tick_bottom()
            ax.set_xlabel('Yield gap factor (-)', fontsize=14)
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) #arguments
            ax.plot((-d,+d),(1-d,1+d), **kwargs) # bottom-left diagonal
            ax.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal
    # Top of the graph:
        if i==0.: 
            ax.set_ylim(nb_combi[var][3]-5.,nb_combi[var][3]+5.)
            ax.set_title('NUTS region %s'%var, fontsize=14)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.tick_top()
            ax.tick_params(labeltop='off')
            #ax.set_ylabel('Sampling method', fontsize=14)
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) #arguments
            ax.plot((-d,+d),(-d,+d), **kwargs) # top-left diagonal
            ax.plot((1-d,1+d),(-d,+d), **kwargs) # top-right diagonal
    
    fig.savefig('sensitivity_yldgapf_to_sampled_rand_combi.png')

    return None

#===============================================================================
def plot_sensitivity_yldgapf_to_top_combi(optimum_yldgapf, nb_combi):
#===============================================================================

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize=(7,3))
    fig.subplots_adjust(0.13,0.17,0.95,0.9,0.4,0.05)
    d = 0.03  # size of the diagonal line in axes coordinates
    for i, var, ax in zip([0.,0.,1.,1.], ['ES41', 'ES43','ES41', 'ES43'], axes.flatten()):
    # Both top and bottom:
        ax.set_yticks([3.,5.,10.,nb_combi[var][3]])
apf[var][0], nb_combi[var][0], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][0], nb_combi[var][0], c='b', s=70,
                   marker='v')
        ax.errorbar(optimum_yldgapf[var][1], nb_combi[var][1], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][1], nb_combi[var][1], c='r', s=70,
                   marker='^')
        ax.errorbar(optimum_yldgapf[var][2], nb_combi[var][2], xerr=0.016, 
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][2], nb_combi[var][2], c='g', s=70, 
                   marker='p')
        ax.errorbar(optimum_yldgapf[var][3], nb_combi[var][3], xerr=0.016,
                    c='k', capsize=5)
        ax.scatter(optimum_yldgapf[var][3], nb_combi[var][3], c='k', s=50, 
                   marker='s')
        labels = [item.get_text() for item in ax.get_yticklabels()] 
        labels[0] = 'top 3'
        labels[1] = 'top 5'
        labels[2] = 'top 10'
        labels[3] = 'all'
        ax.set_yticklabels(labels)
        #ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45) #rotate x labels
    # Bottom of the graph:
        if i==1.: 
            ax.set_ylim(0.,12.)
            ax.spines['top'].set_visible(False)
            ax.xaxis.tick_bottom()
            ax.set_xlabel('Yield gap factor (-)', fontsize=14)
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) #arguments
            ax.plot((-d,+d),(1-d,1+d), **kwargs) # bottom-left diagonal
            ax.plot((1-d,1+d),(1-d,1+d), **kwargs) # bottom-right diagonal
    # Top of the graph:
        if i==0.: 
            ax.set_ylim(nb_combi[var][3]-5.,nb_combi[var][3]+5.)
            ax.set_title('NUTS region %s'%var, fontsize=14)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.tick_top()
            ax.tick_params(labeltop='off')
            #ax.set_ylabel('Sampling method', fontsize=14)
            kwargs = dict(transform=ax.transAxes, color='k', clip_on=False) #arguments
            ax.plot((-d,+d),(-d,+d), **kwargs) # top-left diagonal
            ax.plot((1-d,1+d),(-d,+d), **kwargs) # top-right diagonal
    
    fig.savefig('sensitivity_yldgapf_to_sampled_top_combi.png')

    return None


#===============================================================================
# Function to aggregate the individual yields into the yearly regional yields
def calc_regional_yields(FINAL_YLD, campaign_years_):
#===============================================================================

    TSO_regional = np.zeros(len(campaign_years_)) # empty 1D array
    for y,year in enumerate(campaign_years_):
        sum_weighted_yields  = 0.
        sum_weights          = 0.
        for l in range(0,len(FINAL_YLD['year'])):
            if (FINAL_YLD['year'][l] == year):
                # adding weighted 2D-arrays in the empty array sum_weighted_yields
                sum_weighted_yields = sum_weighted_yields + \
                        FINAL_YLD['weight(-)'][l]*FINAL_YLD['TSO(kgDM.ha-1)'][l] 
                # computing the sum of the soil type area weights
                sum_weights         = sum_weights + FINAL_YLD['weight(-)'][l]
            else:
                pass
        TSO_regional[y] = sum_weighted_yields / sum_weights # weighted average 

    return TSO_regional

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
