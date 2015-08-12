#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

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
   
 
#-------------------------------------------------------------------------------
# Plot the results of the sensitivity analysis

    plt.close('all')
    fig1 = plot_sensitivity_yldgapf_to_top_combi(res_yldgapf, nb_top_combi)
    fig2 = plot_sensitivity_yldgapf_to_rand_combi(res2_yldgapf, nb_rand_combi)
    plt.show()

#-------------------------------------------------------------------------------
# Analyze the forward simulations output data to calculate the regional yields

    # Plot the observed yields in time series
    # retrieve the non-opt sim yields & aggregate into regional non-opt yields
    # retrieve the opt sim yields & aggregate into regional opt yields
    # plot the sim regional yields: non-opt and opt.
    # add time-frame to show years used for optimization.

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
    for y,year in campaign_years_:
        sum_weighted_yields  = 0.
        sum_weights          = 0.
        for grid, year2, stu_no, weight, TSO in FINAL_YLD:
            if (year2 == year):
                # adding weighted 2D-arrays in the empty array sum_weighted_yields
                sum_weighted_yields = sum_weighted_yields + weight*TSO 
                # computing the sum of the soil type area weights
                sum_weights         = sum_weights + weight
        TSO_regional[y] = sum_weighted_yields / sum_weights # weighted average 

    return TSO_regional

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
