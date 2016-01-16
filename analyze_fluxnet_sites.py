#!/usr/bin/env python

import sys, os
import numpy as np
from cPickle import load as pickle_load

#===============================================================================
# This script does some standard analysis on FluxNet sites
def main():
#===============================================================================
    from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
    from _01_select_crops_n_regions import map_crop_id_to_crop_name
#-------------------------------------------------------------------------------
    global EUROSTATdir, FluxNetdir, inputdir, all_grids, lons, lats,\
           Eact0, Cw, wsmax, wsmin, mmC, mmCO2, mmCH2O
#-------------------------------------------------------------------------------
# constants for R_hetero
    Eact0   = 53.3e3    # activation energy [kJ kmol-1]
    Cw      = 1.6e-3    # constant water stress correction (Jacobs et al. 2007)
    wsmax   = 0.55      # upper reference value soil water [-]
    wsmin   = 0.005     # lower reference value soil water [-]
# molar masses for unit conversion of carbon fluxes
    mmC    = 12.01
    mmCO2  = 44.01
    mmCH2O = 30.03 
#-------------------------------------------------------------------------------
# flags to run parts of the script only
    obs_plot        = False
    forward_sim     = False
    sim_plot        = False
    plot_obs_vs_sim = False
    calc_integrated_fluxes_stats = False
    find_optimum_R10 = False
    find_nightime_R10 = True
#-------------------------------------------------------------------------------
# Define general working directories
    EUROSTATdir = '../observations/EUROSTAT_data/'
    FluxNetdir  = '../observations/FluxNet_data/'
    inputdir    = '../model_input_data/'
    output_Sib  = '../model_output/SiBCASA_for_FluxNet_sites/cropland_biome_runs/'
#-------------------------------------------------------------------------------
# we read the CGMS grid cells coordinates from file
    CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']
#-------------------------------------------------------------------------------
# we retrieve the crops, regions, and years to loop over:
    try:
        crop_dict    = pickle_load(open('selected_crops.pickle','rb'))
    except IOError:
        print '\nYou have not yet selected a shortlist of crops to loop over'
        print 'Run the script 01_select_crops_n_regions.py first!\n'
        sys.exit() 
#-------------------------------------------------------------------------------
# list FluxNet sites longitude and latitude data, to retrieve their
# corresponding CGMS grid cell ID

    flux_lat = [50.5522,50.8929,43.5496,48.8442,43.4965,40.5238,51.9536,51.9921]
    flux_lon = [4.7448, 13.5225, 1.1061, 1.9519, 1.2379,14.9574, 4.9029, 5.6459]
    flux_nam = ['BE-Lon','DE-Kli','FR-Aur','FR-Gri','FR-Lam','IT-BCi','NL-Lan','NL-Dij']

    cover = dict()
    cover['Winter wheat'] = ['BE-Lon','DE-Kli','FR-Aur','FR-Gri','FR-Lam']
    cover['Grain maize']  = ['DE-Kli','FR-Gri','FR-Lam','IT-BCi','NL-Dij','NL-Lan']

    years = build_years_dict()

    fgap = build_fgap_dict()

#-------------------------------------------------------------------------------
# plot the observed FluxNet carbon fluxes:

    #Store the FluxNet observed data
    Results = dict()
    for crop in cover.keys():
        Results[crop] = dict()
        for site in cover[crop]:
            Results[crop][site] = dict()
            for year in years[crop][site]:
                Results[crop][site][year] = dict()  
                # retrieve FluxNet data
                listoffiles = [f for f in os.listdir(FluxNetdir) if (site in f and 
                              'daily' in f and str(year) in f)]
                if len(listoffiles) > 0:
                    ObsData = open_csv(FluxNetdir, listoffiles)
                    Results[crop][site][year]['OBS'] = ObsData[listoffiles[0]]
                else:
                    Results[crop][site][year]['OBS'] = None

    # plot the observed data separately
    if obs_plot == True:
        for crop in cover.keys():
            plot_obs_fluxes(crop, cover[crop], years[crop], FluxNetdir)

#-------------------------------------------------------------------------------
# Store the SibCASA runs

    import netCDF4 as cdf

    for crop in cover.keys():
        for site in cover[crop]:
            for year in years[crop][site]:
				# we create an entry in the Results dictionary for the SiBCASA
				# output:
                Results[crop][site][year]['SIM2'] = dict()
                # build the pathname leading to the SiBCASA output files:
                pathname = '%s_%i_run03/'%(site,year)
                filename = 'hsib_%i*.qp2.nc'%year
                pathtofile = os.path.join(output_Sib, pathname, filename)
                # if the run doesn't exist, we go to the next year or site
                if os.path.exists(os.path.join(output_Sib, pathname))==False: 
                    Results[crop][site][year]['SIM2'] = None 
                    continue
                print pathtofile
                # we open all 12 files (1 each month) at once
                f = cdf.MFDataset(pathtofile)
                # retrieve all daily values of gpp, resp, nee of the file:
                # and convert from micromole/m2/s to gC/m2/d:
                fac = 0.000001*12. # conversion from micromole to gC
                dt  = 3600.*24.    # nb of seconds in a day
                Sib_doy = np.array(f.variables['DOY'][:])
                Sib_gpp = np.array(f.variables['gpp'][:])*fac*dt
                Sib_ter = np.array(f.variables['resp_tot'][:])*fac*dt
                #Sib_nee = np.array(f.variables['NEE_1'][:])*fac*dt
                Sib_nee = np.array(f.variables['NEE_2'][:])*fac*dt
                # only if the SiBCASA run exists, we store its output:
                Results[crop][site][year]['SIM2']['DOY'] = Sib_doy
                Results[crop][site][year]['SIM2']['GPP'] = Sib_gpp
                Results[crop][site][year]['SIM2']['TER'] = Sib_ter
                Results[crop][site][year]['SIM2']['NEE'] = Sib_nee

#-------------------------------------------------------------------------------
# find closest CGMS grid cell for all sites:

    flux_gri = dict()
    for i,site in enumerate(flux_nam):
        lon = flux_lon[i]
        lat = flux_lat[i]
        # compute the distance to site for all CGMS grid cells
        dist_list = list()
        for j,grid_no in enumerate(all_grids):
            distance = ((lons[j]-lon)**2. + (lats[j]-lat)**2.)**(1./2.)
            dist_list += [distance] 
        # select the closest grid cell
        indx = np.argmin(np.array(dist_list))
        flux_gri[site] = all_grids[indx]

        print 'FluxNet site %s with lon=%.2f, lat=%.2f: closest grid cell is %i'%(site, lon, lat, all_grids[indx])

#-------------------------------------------------------------------------------
# get the cultivated area of those grid cells:
    filepath = os.path.join(inputdir,'europe_arable_CGMS_cellids.pickle')
    europ_arable = pickle_load(open(filepath,'rb'))
    tuple_cell_area = list()
    for g,a in europ_arable:
        if g in flux_gri.values():
            tuple_cell_area += [(g,a)]

#-------------------------------------------------------------------------------
# Simulate the optimized GPP, Reco and NEE for the FluxNet sites

    # we retrieve the crop id numbers
    crop_dict = map_crop_id_to_crop_name(cover.keys())
    if (find_optimum_R10 == False): # we skip this simulation if we want to find 
                                    # the optimum R10 (see below)
        # loop over crops
        for crop in cover.keys():
            crop_no = crop_dict[crop][0]
            print '\n %s'%crop
            # loop over sites
            for site in cover[crop]:
                # get the closest grid cell id number:
                grid_no = int(flux_gri[site])
                culti_land = [a for (g,a) in tuple_cell_area if (g==grid_no)][0]
                # get the longitute and latitute of the site:
                indx = flux_nam.index(site)
                lon = flux_lon[indx]
                lat = flux_lat[indx]
                #print '\n%s: lon = %.2f, lat = %.2f, closest cell = %i'%(site, lon, 
                #                                                     lat, grid_no)
                # stuff to print to screen
                if (forward_sim == True):
                    print '\nYLDGAPF(-),  grid_no,  year,  stu_no, stu_area(ha), '\
                     +'TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                     +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)'
                for year in years[crop][site]:
                    if (grid_no == 107097):  # I forgot to retrieve input
                                             # data for the Dijkgraaf site
                        Results[crop][site][year]['SIM'] = None
                        continue
                    # OPTIMIZATION OF FGAP:
                    yldgapf = fgap[crop][site][year]
                    # FORWARD SIMULATIONS:
                    if (forward_sim == True):
                        perform_yield_sim(crop_no, grid_no, int(year), yldgapf)
                    # POST-PROCESSING OF GPP, RAUTO, RHET, NEE:
                    R10     = 0.08 # respiration at 10C [mgCO2 m-2 s-1]
                    SimData = compile_nee(crop_no, crop, site, grid_no, year, R10)
                    Results[crop][site][year]['SIM'] = SimData
                    # if required, plot the simulated fluxes separately:
                    if (sim_plot == True):
                        plot_sim_fluxes(time_series[0],time_series[1],
                        time_series[2],time_series[3], time_series[4], crop, grid_no,
                        site, year, units='perday')

#-------------------------------------------------------------------------------
# plot the observed and simulated fluxes of GPP, TER, NEE for each site
    if (plot_obs_vs_sim == True):
#-------------------------------------------------------------------------------
        from matplotlib import pyplot as plt
        # loop over crops
        for crop in cover.keys():
            crop_no = crop_dict[crop][0]
            print '\n %s'%crop
            # loop over sites
            for site in cover[crop]:
                grid_no = int(flux_gri[site])
                print '\n%s: lon = %.2f, lat = %.2f, closest cell = %i'%(site, 
                                                              lon, lat, grid_no)
                for year in years[crop][site]:
                    plot_sim_vs_obs_fluxes(Results[crop][site][year], crop,
                    site, grid_no, year)

#-------------------------------------------------------------------------------
# calculate yearly integral of the GPP, TER and NEE fluxes, and compute some 
# statistics on it like mean, std dev, RMSE
    if (calc_integrated_fluxes_stats == True):
#-------------------------------------------------------------------------------
        # loop over crops
        for crop in cover.keys():
            crop_no = crop_dict[crop][0]
            # loop over sites
            for site in cover[crop]:
                grid_no = int(flux_gri[site])
                # loop over years
                for year in years[crop][site]:
                    if (Results[crop][site][year]['OBS'] != None and
                    Results[crop][site][year]['SIM'] != None):
                        compute_stats_on_integrated_fluxes(Results[crop][site][year],
                        crop, site, grid_no, year)

#-------------------------------------------------------------------------------
# We want to find the optimum R10 per site that minimizes the NRMSE of both TER 
# and NEE
    if (find_optimum_R10 == True):
        print "\nLet's find the optimum R10!"
#-------------------------------------------------------------------------------
        # loop over crops
        for crop in cover.keys():
            crop_no = crop_dict[crop][0]
            # loop over sites
            for site in cover[crop]:
                # get the closest grid cell id number:
                grid_no = int(flux_gri[site])
                #culti_land = [a for (g,a) in tuple_cell_area if (g==grid_no)][0]
                # get the longitute and latitute of the site:
                #indx = flux_nam.index(site)
                #lon = flux_lon[indx]
                #lat = flux_lat[indx]
                # stuff to print to screen
                if (forward_sim == True):
                    print '\nYLDGAPF(-),  grid_no,  year,  stu_no, stu_area(ha), '\
                     +'TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                     +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)'
                for year in years[crop][site]:
                    if (grid_no == 107097):  # I forgot to retrieve input
                                             # data for the Dijkgraaf site
                        Results[crop][site][year]['SIM'] = None
                        continue
                    if (Results[crop][site][year]['OBS'] == None):
                        continue
                    # OPTIMIZATION OF FGAP:
                    yldgapf = fgap[crop][site][year]
                    # FORWARD SIMULATIONS:
                    if (forward_sim == True):
                        perform_yield_sim(crop_no, grid_no, int(year), yldgapf)
                    # POST-PROCESSING OF GPP, RAUTO, RHET, NEE:
                    R10_list  = np.arange(0.00,0.24,0.01)
                    rmse_ter = np.array([0.]*len(R10_list))
                    rmse_nee = np.array([0.]*len(R10_list))
                    print '\n---------------------------------------------'
                    print 'Site %s, %s, year %i'%(site, crop, year)
                    print '---------------------------------------------'
                    for i,R10 in enumerate(R10_list):
                        print 'R10 = %.2f'%R10
                        SimData = compile_nee(crop_no, crop, site, grid_no, year, R10)
                        Results[crop][site][year]['SIM'] = SimData
                        # COMPUTING RMSE STATISTICS:
                        dum = compute_stats_on_integrated_fluxes(
                        Results[crop][site][year], crop, site, grid_no, year,
                        print_screen=False)
                        rmse_ter[i] = dum[1]['TER']
                        rmse_nee[i] = dum[1]['NEE']
                    # plot the RMSE and NRMSE for that site:
                    plot_optimum_R10(R10_list, rmse_ter, rmse_nee, site, crop, year)

#-------------------------------------------------------------------------------
# We analyze the R10 of the nighttime respiration
    if (find_nightime_R10 == True):
        print "\nLet's find the nightime R10!"
#-------------------------------------------------------------------------------
        # loop over crops
        for crop in cover.keys():
            crop_no = crop_dict[crop][0]
            # loop over sites
            for site in cover[crop]:
                # get the closest grid cell id number:
                grid_no = int(flux_gri[site])
                # loop over years
                for year in years[crop][site]:
                    # we open the hourly data files
                    #
                    if (Results[crop][site][year]['OBS'] == None):
                        continue
                    # we select the night-time respiration from the observations
                    obs_ter = Results[crop][site][year]['OBS']['Reco']
                    # we select the night-time temperature from the observations
                    obs_ts  = Results[crop][site][year]['OBS']['Ts1_f']
                    print len(obs_ter), len(obs_ts)
                    print Results[crop][site][year]['OBS']['SWin']
                    sys.exit()
                    # we filter the missing data out
                    ma_ter  = np.ma.masked_equal(obs_ter,-9999.) 
                    ma_ts   = np.ma.masked_equal(obs_ts ,-9999.) 
                    # we filter the daytime hours: i.e. when SWin>0.
#                    ma_day  = 
#        ma_sim_doy    = np.ma.masked_outside(Results_dict['SIM'][0], 
#                                           Results_dict['OBS']['DoY'][0],
#                                           Results_dict['OBS']['DoY'][-1])
#        obs_mask      = np.ma.getmask(ma_sim_doy)
#        # apply the daytime mask on both ter and ts:
#        ma_sim_data   = np.ma.masked_where(obs_mask, sim_data)
#        # we compute statistics on the filtered data
#        sum_sim_var   = np.ma.MaskedArray.sum(ma_sim_data)
#        ave_sim_var   = np.ma.MaskedArray.mean(ma_sim_data)
#        stdev_sim_var = np.ma.MaskedArray.std(ma_sim_data)
#
#        # we compute the root mean squared error of the model:
#        RMSE[name]  = np.sqrt(((ma_sim_data - ma_obs_data) ** 2).mean())
#        NRMSE[name] = RMSE[name] / (np.ma.max(ma_obs_data) - np.ma.min(ma_obs_data)) *100.

                    # we loop over R10 to simulate the night-time respiration 
                    # with temperature
                    # we choose the R10 that minimizes RMSE

#===============================================================================
def plot_optimum_R10(R10_list, rmse_ter, rmse_nee, site_name, crop_name, year):
#===============================================================================
    from matplotlib import pyplot as plt
    plt.close('all')
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    fs = 16
    fig.subplots_adjust(0.15,0.15,0.97,0.97,0.2,0.2)
    ax.plot(R10_list, rmse_ter, label='TER', c='k', ls='-', lw=2)
    ax.plot(R10_list, rmse_nee, label='NEE', c='k', ls='--', lw=2)
    indx_1    = np.argmin(rmse_ter)
    optimum_1 = R10_list[indx_1]
    indx_2    = np.argmin(rmse_nee)
    optimum_2 = R10_list[indx_2]
    ax.axvline(x=optimum_1, ymin=0., ymax=100., c='r', lw=1)
    ax.axvline(x=optimum_2, ymin=0., ymax=100., c='r', ls='--', lw=1)
    ax.set_xlim([0.,0.23])
    ax.set_ylim([0.,100.])
    ax.set_xlabel('R10', fontsize=fs)
    ax.set_ylabel('NRMSE (%)', fontsize=fs)
    ysize = [tick.label.set_fontsize(fs) for tick in ax.yaxis.get_major_ticks()]
    xsize = [tick.label.set_fontsize(fs) for tick in ax.xaxis.get_major_ticks()]
    plt.legend(loc='upper left', ncol=2, fontsize=14)
    #plt.show()
    fig.savefig('../figures/R10opti_%s_%i.png'%(site_name,year))

#===============================================================================
def compute_stats_on_integrated_fluxes(Results_dict, crop_name, site_name, 
grid_no, year, print_screen=True):
#===============================================================================
    RMSE  = dict()
    NRMSE = dict()

    if (print_screen == True): 
        print '\n---------------------------------------------'
        print 'Site %s, %s, year %i'%(site_name, crop_name, year)
        print 'there are %i observed days and %i simulated days'%(
        np.ma.count(np.ma.masked_equal(Results_dict['OBS']['GPP_f'],-9999.)),
        len(Results_dict['SIM'][0]))
        print '---------------------------------------------'
    obs_varz = ['GPP_f','Reco','NEE_f']
    factorz  = [-1.,1.,1.]
    for var,i,name,fac in zip(obs_varz,[1,3,4], ['GPP','TER','NEE'], factorz):
        # STATS ON OBSERVATIONS
        # we filter out the -9999. values:
        ma_obs_data   = fac*np.ma.masked_equal(Results_dict['OBS'][var], -9999.)
        # we compute statistics on the filtered data
        sum_obs_var   = np.ma.MaskedArray.sum(ma_obs_data)
        ave_obs_var   = np.ma.MaskedArray.mean(ma_obs_data)
        stdev_obs_var = np.ma.MaskedArray.std(ma_obs_data)

        # STATS ON SIMULATIONS
        # we filter out the days on which we didn't have observations
        ma_sim_doy    = np.ma.masked_outside(Results_dict['SIM'][0], 
                                           Results_dict['OBS']['DoY'][0],
                                           Results_dict['OBS']['DoY'][-1])
        obs_mask      = np.ma.getmask(ma_sim_doy)
        if (i != 3): # For GPP or NEE
            sim_data  = Results_dict['SIM'][i]
        else: # for TER, we use the max between WOFOST Rauto and AGs TER:
            sim_data  = np.maximum.reduce([Results_dict['SIM'][3], 
                                           Results_dict['SIM'][2]]) 
        #ma_sim_data   = np.ma.masked_where(obs_mask, sim_data)
        ma_sim_data   = np.ma.masked_equal(sim_data, -9999.)
        # we compute statistics on the filtered data
        sum_sim_var   = np.ma.MaskedArray.sum(ma_sim_data)
        ave_sim_var   = np.ma.MaskedArray.mean(ma_sim_data)
        stdev_sim_var = np.ma.MaskedArray.std(ma_sim_data)

        # we compute the root mean squared error of the model:
        RMSE[name]  = np.sqrt(((ma_sim_data - ma_obs_data) ** 2).mean())
        NRMSE[name] = RMSE[name] / (np.ma.max(ma_obs_data) - np.ma.min(ma_obs_data)) *100.

        # We print out the results on screen
        if (print_screen == True):
            print '\n%s'%name
            print '         %10s %10s'%('SIM', 'OBS')
            print 'Sum:     %10.2f %10.2f gC m-2 y-1'%(sum_sim_var, sum_obs_var)
            print 'Mean:    %10.2f %10.2f gC m-2 d-1'%(ave_sim_var, ave_obs_var)
            print 'Std dev: %10.2f %10.2f gC m-2 d-1'%(stdev_sim_var, stdev_obs_var)
            print 'RMSE of %s: %5.2f gC m-2 d-1'%(name, RMSE[name])
            print 'NRMSE of %s: %5.2f'%(name, NRMSE[name])+' %'

    return RMSE, NRMSE

#===============================================================================
def plot_sim_vs_obs_fluxes(Results_dict, crop_name, site_name, grid_no, year):
#===============================================================================

    from matplotlib import pyplot as plt
    from matplotlib import rcParams as rc
    plt.close('all')

    obs_varz = ["GPP_f","Reco","NEE_f"]
    labz     = ['GPP','Reco','NEE']
    sib_labz = ['GPP','TER','NEE']
    colorz   = ['k','k','k']
    limz     = [[-20.,0.],[0.,10.],[-15.,0.]]
    factorz  = [-1.,1.,1.] 
    if (Results_dict['OBS'] != None): obs_time = Results_dict['OBS']['DoY']
    if (Results_dict['SIM'] != None): sim_time = Results_dict['SIM'][0]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
    fig.subplots_adjust(0.1,0.1,0.98,0.98,0.2,0.2)

    for ax, it, var, fac, sib_lab, lab, col, lims in zip(axes.flatten(), [1,2,4], obs_varz, 
    factorz, sib_labz, labz, colorz, limz):
        ax.axhline(y=0., ls='-', lw=0.5, c='k')
        # plot observed fluxes as scattered dots:
        if (Results_dict['OBS'] != None): 
            ax.scatter(obs_time, fac*np.ma.masked_equal(Results_dict['OBS'][var],
                       -9999.), marker='+', s=50, label='obs', c=col)
        # plot the GPP and NEE simulated fluxes:
        if (lab != 'Reco' and Results_dict['SIM'] != None): 
            ax.plot(sim_time, Results_dict['SIM'][it], label='WOFOST+', c=col, 
                    ls='-', lw=2)
        # plot the TER simulated fluxes:
        if (lab == 'Reco' and Results_dict['SIM'] != None): 
            ax.plot(sim_time, np.maximum.reduce([Results_dict['SIM'][3], 
                    Results_dict['SIM'][2]]), c=col, ls='-', lw=2)
            ax.fill_between(sim_time, 0., Results_dict['SIM'][2], facecolor=col, 
                            alpha=0.2, lw=0)
            #ax.plot(sim_time, Results_dict['SIM'][3], c=col, ls=':', lw=2)
        # plot the SiBCASA results:
        if (Results_dict['SIM2'] != None): 
            ax.plot(Results_dict['SIM2']['DOY'], Results_dict['SIM2'][sib_lab]*fac,
                    c=col, ls='--', lw=2, label='SiBCASA')
        ax.set_xlim([0.,366.])
        #ax.set_ylim(lims)
        ax.set_ylabel(lab + r' (g$_{C}$ m$^{-2}$ d$^{-1}$)', fontsize=18)
        size = [tick.label.set_fontsize(18) for tick in ax.yaxis.get_major_ticks()]
        size = [tick.label.set_fontsize(18) for tick in ax.xaxis.get_major_ticks()]
        if (lab=='GPP'): 
            ax.legend(loc='lower left', ncol=1, fontsize=14)
    plt.xlabel('time (DOY)', fontsize=18)
    figname = '%s_%i_%s_g%i.png'%(crop_name,year,\
                                                        site_name,grid_no)
    fig.savefig('../figures/FluxNet_R10=08_'+figname)
    return None

#===============================================================================
# Function to do forward simulations of crop yield for a given YLDGAPF and for a
# selection of grid cells x soil types within a NUTS region
def perform_yield_sim(crop_no, grid_no, year, fgap):
#===============================================================================
    from pcse.fileinput.cabo_weather import CABOWeatherDataProvider
    from maries_toolbox import select_soils
    from pcse.models import Wofost71_WLP_FD
#-------------------------------------------------------------------------------
    # fixed settings for these point simulations:
    weather = 'ECMWF'   
    selec_method = 'all' 
    nsoils = 1
    pickledir = '../model_input_data/CGMS/'
    caboecmwfdir = '../model_input_data/CABO_weather_ECMWF/'
    outputdir = '../model_output/FluxNet_sites_output/'
#-------------------------------------------------------------------------------
    # Retrieve the weather data of one grid cell
    if (weather == 'CGMS'):
        filename = pickledir+'weatherobject_g%d.pickle'%grid_no
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
    if (weather == 'ECMWF'):
        weatherdata = CABOWeatherDataProvider('%i'%(grid_no), 
                                                         fpath=caboecmwfdir)
    #print weatherdata(datetime.date(datetime(2006,4,1)))

    # Retrieve the soil data of one grid cell 
    filename =pickledir+'soilobject_g%d.pickle'%grid_no
    soil_iterator = pickle_load(open(filename,'rb'))

    # Retrieve calendar data of one year for one grid cell
    filename = pickledir+'timerobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    timerdata = pickle_load(open(filename,'rb'))
                    
    # Retrieve crop data of one year for one grid cell
    filename = pickledir+'cropobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    cropdata = pickle_load(open(filename,'rb'))

    # retrieve the fgap data of one year and one grid cell
    cropdata['YLDGAPF'] = fgap

    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no], pickledir, 
                                       method=selec_method, n=nsoils)

    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        
        # Retrieve the site data of one year, one grid cell, one soil type
        filename = pickledir+'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid_no,crop_no,
                                                                      year,stu_no)
        sitedata = pickle_load(open(filename,'rb'))

        # run WOFOST
        wofost_object = Wofost71_WLP_FD(sitedata, timerdata, soildata, cropdata, 
                                                                    weatherdata)
        wofost_object.run_till_terminate() #will stop the run when DVS=2

        # get time series of the output and take the selected variables
        wofost_object.store_to_file( outputdir +\
                                    "pcse_timeseries_c%i_y%i_g%i_s%i.csv"\
                                    %(crop_no,year,grid_no,stu_no))

        # get major summary output variables for each run
        # total dry weight of - dead and alive - storage organs (kg/ha)
        TSO       = wofost_object.get_variable('TWSO')
        # total dry weight of - dead and alive - leaves (kg/ha) 
        TLV       = wofost_object.get_variable('TWLV')
        # total dry weight of - dead and alive - stems (kg/ha) 
        TST       = wofost_object.get_variable('TWST')
        # total dry weight of - dead and alive - roots (kg/ha) 
        TRT       = wofost_object.get_variable('TWRT')
        # maximum LAI
        MLAI      = wofost_object.get_variable('LAIMAX')
        # rooting depth (cm)
        RD        = wofost_object.get_variable('RD')
        # Total above ground dry matter (kg/ha)
        TAGP      = wofost_object.get_variable('TAGP')

        #output_string = '%10.3f, %8i, %5i, %7i, %15.2f, %12.5f, %14.2f, '
                        #%(yldgapf, grid_no, year, stu_no, arable_area/10000.,stu_area/10000.,TSO) 
        output_string = '%10.3f, %8i, %5i, %7i, %12.5f, %14.2f, '%(fgap,
                         grid_no, year, stu_no, stu_area/10000., TSO) + \
                        '%14.2f, %14.2f, %14.2f, %14.2f, %13.2f, %15.2f'%(TLV,
                         TST, TRT, MLAI, RD, TAGP)
        print output_string

    return None

#===============================================================================
def compile_nee(crop_no, crop_name, site_name, grid_no, year, R10):
#===============================================================================
# fixed settings for these point simulations:
    prod_figure = True
    #weather = 'ECMWF'   
    #CGMSdir  = '/Users/mariecombe/mnt/promise/CO2/marie/pickled_CGMS_input_data/'
    CGMSdir = '../model_input_data/CGMS/'
    pcse_outputdir = '../model_output/FluxNet_sites_output/'
    #caboecmwfdir = '../model_input_data/CABO_weather_ECMWF/'
    #outputdir = '../model_output/FluxNet_sites_output/'
#-------------------------------------------------------------------------------
    from cPickle import dump as pickle_dump
    from maries_toolbox import select_soils, open_pcse_csv_output
    import datetime as dt
    import math
#-------------------------------------------------------------------------------
# We retrieve the longitude and latitude of the CGMS grid cell
    i   = np.argmin(np.absolute(all_grids - grid_no))
    lon = lons[i]
    lat = lats[i]
    #print '- grid cell no %i: lon = %.2f , lat = %.2f'%(grid_no,lon,lat)

#-------------------------------------------------------------------------------
# We open the incoming surface shortwave radiation [W.m-2] 

    filename_rad = 'rad_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(year,lon,lat)
    path_rad     = os.path.join(inputdir,filename_rad)
    if os.path.exists(path_rad):
        rad = pickle_load(open(path_rad, 'rb'))
    else:
        rad = retrieve_ecmwf_ssrd(year, lon, lat)
        pickle_dump(rad,open(path_rad, 'wb'))

#-------------------------------------------------------------------------------
# We open the surface temperature record

    filename_ts = 'ts_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(year,lon,lat)
    path_ts     = os.path.join(inputdir,filename_ts)
    if os.path.exists(path_ts):
        ts = pickle_load(open(path_ts, 'rb'))
    else:
        ts = retrieve_ecmwf_tsurf(year, lon, lat)
        pickle_dump(ts,open(path_ts, 'wb'))

#-------------------------------------------------------------------------------
# we initialize the timeseries of gpp and Resp for the grid cell

    time_cell_persec_timeseries = rad[0]
    time_cell_perday_timeseries = rad[0][::8]/(3600.*24.)

    len_cell_persec_timeseries  = len(rad[0])
    len_cell_perday_timeseries  = len(rad[0][::8])

    gpp_cell_persec_timeseries  = np.array([0.]*len_cell_persec_timeseries)
    gpp_cell_perday_timeseries  = np.array([0.]*len_cell_perday_timeseries)

    raut_cell_persec_timeseries = np.array([0.]*len_cell_persec_timeseries)
    raut_cell_perday_timeseries = np.array([0.]*len_cell_perday_timeseries)

    rhet_cell_persec_timeseries = np.array([0.]*len_cell_persec_timeseries)
    rhet_cell_perday_timeseries = np.array([0.]*len_cell_perday_timeseries)
    sum_stu_areas        = 0.
    # number of seconds in 3 hours (the radiation and temperature are 3-hourly)
    delta = 3600. * 3.

    if (prod_figure == True):
        from matplotlib import pyplot as plt
        plt.close('all')
        fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
        fig1.subplots_adjust(0.1,0.1,0.98,0.9,0.2,0.2)

#-------------------------------------------------------------------------------
    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no],
                           CGMSdir, method='all', n=1)
#-------------------------------------------------------------------------------
#       WE NEED TO LOOP OVER THE SOIL TYPE
    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        #print grid_no, stu_no
#-------------------------------------------------------------------------------
# We open the WOFOST results file

        filename    = 'pcse_timeseries_c%i_y%i_g%i_s%i.csv'\
                       %(crop_no, year, grid_no, stu_no) 
        results_set = open_pcse_csv_output(pcse_outputdir, [filename])
        wofost_data = results_set[0]
        #print wofost_data[filename].keys()

#-------------------------------------------------------------------------------
# We apply the short wave radiation diurnal cycle on the GPP and R_auto

        # we create empty time series for this specific stu
        gpp_cycle_timeseries   = np.array([])
        raut_cycle_timeseries  = np.array([])
        gpp_perday_timeseries  = np.array([])
        raut_perday_timeseries = np.array([])

        # we compile the sum of the stu areas to do a weighted average of
        # GPP and Rauto later on
        sum_stu_areas += stu_area 

        for DOY, timeinsec in enumerate(time_cell_persec_timeseries[::8]):
            # conversion of current time in seconds into date
            time = dt.date(year,1,1) + dt.timedelta(DOY)
            #print 'date:', time

            # we test to see if we are within the growing season
            test_sow = (time - wofost_data[filename]['day'][0]).total_seconds()
            test_rip = (time - wofost_data[filename]['day'][-1]).total_seconds() 
            #print 'tests:', test_sow, test_rip

            # if the day of the time series is before sowing date: plant 
            # fluxes are set to zero
            if test_sow < 0.: 
                gpp_day  = 0.
                raut_day = 0.
            # or if the day of the time series is after the harvest date: 
            # plant fluxes are set to zero
            elif test_rip > 0.: 
                gpp_day  = 0.
                raut_day = 0.
            # else we get the daily total GPP and Raut in kgCH2O/ha/day
            # from wofost, and we weigh it with the stu area to later on 
            # calculate the weighted average GPP and Raut in the grid cell
            else: 
                # index of the sowing date in the time_cell_timeseries:
                if (test_sow == 0.): DOY_sowing  = DOY
                if (test_rip == 0.): DOY_harvest = DOY
                #print 'DOY sowing:', DOY_sowing
                # translation of cell to stu timeseries index
                index_day_w  = DOY - DOY_sowing
                #print 'index of day in wofost record:', index_day_w

                # unit conversion: from kgCH2O/ha/day to gC/m2/day
                gpp_day    = - wofost_data[filename]['GASS'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                maint_resp = wofost_data[filename]['MRES'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                try: # if there are any available assimilates for growth
                    growth_fac = (wofost_data[filename]['DMI'][index_day_w]) / \
                             (wofost_data[filename]['GASS'][index_day_w] - 
                              wofost_data[filename]['MRES'][index_day_w])
                    growth_resp = (1.-growth_fac)*(-gpp_day-maint_resp) 
                except ZeroDivisionError: # otherwise there is no crop growth
                    growth_resp = 0.
                raut_day   = growth_resp + maint_resp

            # we select the radiation diurnal cycle for that date
            # NB: the last index is ignored in the selection, so we DO have
            # 8 time steps selected only (it's a 3-hourly dataset)
            rad_cycle      = rad[1][DOY*8:DOY*8+8] 

            # we apply the radiation cycle on the GPP and Rauto
            # and we transform the daily integral into a rate
            weights        = rad_cycle / sum(rad_cycle)
            # the sum of the 8 rates is equal to total/delta:
            sum_gpp_rates  = gpp_day   / delta
            sum_raut_rates = raut_day  / delta
            # the day's 8 values of actual gpp and raut rates per second:
            gpp_cycle      = weights * sum_gpp_rates
            raut_cycle     = weights * sum_raut_rates
            # NB: we check if the applied diurnal cycle is correct
            assert (sum(weights)-1. < 0.000001), "wrong radiation kernel"
            assert (len(gpp_cycle)*int(delta) == 86400), "wrong dt in diurnal cycle"
            assert ((sum(gpp_cycle)*delta-gpp_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on GPP: residual=%.2f "%(sum(gpp_cycle)*delta-gpp_day) +\
                "on DOY %i"%DOY
            assert ((sum(raut_cycle)*delta-raut_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on Rauto: residual=%.2f "%(sum(raut_cycle)*delta-raut_day) +\
                "on DOY %i"%DOY

            # if the applied diurnal cycle is ok, we append that day's cycle
            # to the yearly record of the stu
            gpp_cycle_timeseries  = np.concatenate((gpp_cycle_timeseries, 
                                                   gpp_cycle), axis=0)
            raut_cycle_timeseries = np.concatenate((raut_cycle_timeseries,
                                                   raut_cycle), axis=0)
            # we also store the carbon fluxes per day, for comparison with fluxnet
            gpp_perday_timeseries = np.concatenate((gpp_perday_timeseries,
                                                   [gpp_day]), axis=0) 
            raut_perday_timeseries = np.concatenate((raut_perday_timeseries,
                                                   [raut_day]), axis=0)

        if (prod_figure == True):
            for ax, var, name, lims in zip(axes.flatten(), 
            [gpp_perday_timeseries, raut_perday_timeseries, 
            gpp_perday_timeseries + raut_perday_timeseries],
            ['GPP', 'Rauto', 'NPP'], [[-20.,0.],[0.,10.],[-15.,0.]]):
                ax.plot(time_cell_perday_timeseries, var, label='stu %i'%stu_no)
                #ax.set_xlim([40.,170.])
                #ax.set_ylim(lims)
                ax.set_ylabel(name + r' (g$_{C}$ m$^{-2}$ d$^{-1}$)', fontsize=14)

#-------------------------------------------------------------------------------
# We add the gpp of all soil types in the grid cell. NB: different calendars
# are applied depending on the site!! so the sowing and maturity dates might
# differ from stu to stu

		# TWO OPTIONS: we can compile time series of carbon fluxes in units per
		# day or per second
        # a- sum the PER SECOND timeseries of GPP and Rauto for all soil types
        gpp_cell_persec_timeseries  = gpp_cell_persec_timeseries + \
                                      gpp_cycle_timeseries*stu_area
        raut_cell_persec_timeseries = raut_cell_persec_timeseries + \
                                      raut_cycle_timeseries*stu_area

        # b- sum the PER DAY timeseries of GPP and Rauto for all soil types
        gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries + \
                                      gpp_perday_timeseries*stu_area
        raut_cell_perday_timeseries = raut_cell_perday_timeseries + \
                                      raut_perday_timeseries*stu_area

    # finish the figure of multiple stu carbon fluxes
    if (prod_figure == True):
        plt.xlabel('time (DOY)', fontsize=14)
        plt.legend(loc='upper left', ncol=2, fontsize=10)
        fig1.suptitle('Daily carbon fluxes of %s for all '%crop_name+\
                     'soil types of grid cell %i in %i'%(grid_no,
                                              year), fontsize=18)
        figname = 'GPP_allsoils_%s_%i_%s_g%i.png'%(crop_name,year,\
                                                            site_name,grid_no)
        #plt.show()
        fig1.savefig('../figures/FluxNet_'+figname)

    # TWO OPTIONS: we can compile time series of carbon fluxes in units per day
    # or per second
    # a- for each grid cell, weighted average GPP and Rauto PER SECOND
    gpp_cell_persec_timeseries  = gpp_cell_persec_timeseries  / sum_stu_areas
    raut_cell_persec_timeseries = raut_cell_persec_timeseries / sum_stu_areas

    # b- for each grid cell, weighted average GPP and Rauto PER DAY
    gpp_cell_perday_timeseries  = gpp_cell_perday_timeseries  / sum_stu_areas
    raut_cell_perday_timeseries = raut_cell_perday_timeseries / sum_stu_areas

#-------------------------------------------------------------------------------
# We calculate the heterotrophic respiration with the surface temperature

    # from the A-gs model:
    # pb: we need to simulate wg with that approach...
    #fw = Cw * wsmax / (wg + wsmin)
    tsurf_inter = Eact0 / (283.15 * 8.314) * (1 - 283.15 / ts[1])
    rhet_cell_persec_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 
    # compute rhet per day:
    for i in range(len(rhet_cell_perday_timeseries)):
        rhet_cell_perday_timeseries[i] = rhet_cell_persec_timeseries[i*8] * delta +\
                                       rhet_cell_persec_timeseries[i*8+1] * delta +\
                                       rhet_cell_persec_timeseries[i*8+2] * delta +\
                                       rhet_cell_persec_timeseries[i*8+3] * delta +\
                                       rhet_cell_persec_timeseries[i*8+4] * delta +\
                                       rhet_cell_persec_timeseries[i*8+5] * delta +\
                                       rhet_cell_persec_timeseries[i*8+6] * delta +\
                                       rhet_cell_persec_timeseries[i*8+7] * delta 

    # conversion from mgCO2 to gC
    conversion_fac = (mmC / mmCO2) * 0.001
    rhet_cell_persec_timeseries = rhet_cell_persec_timeseries * conversion_fac
    rhet_cell_perday_timeseries = rhet_cell_perday_timeseries * conversion_fac

#-------------------------------------------------------------------------------
# We calculate NEE as the net flux

    nee_cell_persec_timeseries = gpp_cell_persec_timeseries  + \
                                 raut_cell_persec_timeseries + \
                                 rhet_cell_persec_timeseries
    nee_cell_perday_timeseries = gpp_cell_perday_timeseries  + \
                                 raut_cell_perday_timeseries + \
                                 rhet_cell_perday_timeseries

#-------------------------------------------------------------------------------
# here we choose to return the carbon fluxes PER DAY
    return time_cell_perday_timeseries, gpp_cell_perday_timeseries, \
           raut_cell_perday_timeseries, rhet_cell_perday_timeseries, \
           nee_cell_perday_timeseries

#===============================================================================
def plot_obs_fluxes(crop, site_names, years_dict, FLUXNETdir):
#===============================================================================

    from maries_toolbox import open_csv
    from matplotlib import pyplot as plt
    varz   = ["GPP_f","Reco","NEE_f"]
    labz   = ['GPP','Reco','NEE']
    colorz = ['g','r','k']

    for site in site_names:
        print site, years_dict[site]
        # retrieve FluxNet data
        listoffiles = [f for f in os.listdir(FLUXNETdir) if (site in f and 
                       'daily' in f)]
        print listoffiles
        if len(listoffiles) > 0:
            FNdata = open_csv(FLUXNETdir, listoffiles)
        else:
            continue
        # create a new figure
        nbj = 0. # nb of days in a year
        first_year = 0.
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(14,7))
        for y,year in enumerate(years_dict[site]):
            shortlist = [f for f in listoffiles if (str(year) in f)]
            print shortlist
            if len(shortlist)>0:
                if first_year == 0.: first_year = year
                filename = shortlist[0]
                for v,var in enumerate(varz):
                    plt.plot(FNdata[filename]['DoY']+y*nbj,
                    np.ma.masked_equal(FNdata[filename][var],-9999.), 
                    c=colorz[v], label=labz[v])
                nbj = FNdata[filename]['DoY'][-1]
            if year==first_year: plt.legend(loc='best')
        plt.axhline(y=0., ls='-', lw=0.5, c='k')
        plt.xlabel('Time (DOY)')
        plt.ylabel('Carbon fluxes (gC m-2 d-1)')
        plt.title('%s: %s - first year = %i'%(crop,site,first_year))
        #plt.show()
        fig.savefig('../figures/FluxNet_%s_%s.png'%(crop,site))

    return None

#===============================================================================
def plot_sim_fluxes(time_cell_timeseries, gpp_cell_timeseries, 
    raut_cell_timeseries, rhet_cell_timeseries, nee_cell_timeseries, crop_name,
    grid_no, site_name, year, units='persec'):
#===============================================================================

    from matplotlib import pyplot as plt
    fig2 = plt.figure(figsize=(14,6))
    fig2.subplots_adjust(0.1,0.2,0.98,0.85,0.4,0.6)

    if (units == 'persec'):
        plt.plot(time_cell_timeseries/(3600.*24.),gpp_cell_timeseries*1000., 
                                                label=r'$\mathrm{GPP}$', c='g')
        plt.plot(time_cell_timeseries/(3600.*24.),raut_cell_timeseries*1000.,
                                                label=r'$R_{aut}$', c='b')
        plt.plot(time_cell_timeseries/(3600.*24.),rhet_cell_timeseries*1000.,
                                                label=r'$R_{het}$', c='r')
        plt.plot(time_cell_timeseries/(3600.*24.),nee_cell_timeseries*1000., 
                                                label=r'$\mathrm{NEE}$', c='k')
        plt.ylabel(r'carbon flux (mg$_{C}$ m$^{-2}$ s$^{-1}$)')
    if (units == 'perday'):
        plt.plot(time_cell_timeseries,gpp_cell_timeseries, 
                                                label=r'$\mathrm{GPP}$', c='g')
        plt.plot(time_cell_timeseries,raut_cell_timeseries,
                                                label=r'$R_{aut}$', c='b')
        plt.plot(time_cell_timeseries,rhet_cell_timeseries,
                                                label=r'$R_{het}$', c='r')
        plt.plot(time_cell_timeseries,nee_cell_timeseries, 
                                                label=r'$\mathrm{NEE}$', c='k')
        plt.ylabel(r'carbon flux (g$_{C}$ m$^{-2}$ d$^{-1}$)')
    plt.xlim([0.,365.])
    plt.xlabel('time (DOY)')
    plt.legend(loc='best', ncol=2, fontsize=10)
    plt.title('Average carbon fluxes of %s over the  '%crop_name+\
                 'cultivated area of grid cell %i in %i'%(grid_no,
                                                    year))
    filename = 'FluxNet_NEE_cell%i_year%i_%s_%s.png'%(grid_no,year,crop_name,site_name)
    fig2.savefig('../figures/'+filename)
    return None

#===============================================================================
def build_years_dict():
#===============================================================================
    years = dict()
    years['Winter wheat'] = dict()
    years['Grain maize']  = dict()
    # list of years for winter wheat sites:
    years['Winter wheat']['BE-Lon'] = [2005,2007] #winter wheat rotation
    years['Winter wheat']['DE-Kli'] = [2006]
    years['Winter wheat']['FR-Aur'] = [2006] #not sure between 2005 and 2006
    years['Winter wheat']['FR-Gri'] = [2006]
    years['Winter wheat']['FR-Lam'] = [2007]
    # list of years for grain maize sites:
    years['Grain maize']['DE-Kli']  = [2007]
    years['Grain maize']['FR-Gri']  = [2005]
    years['Grain maize']['FR-Lam']  = [2006]
    years['Grain maize']['IT-BCi']  = [2004]
    years['Grain maize']['NL-Lan']  = [2005]
    years['Grain maize']['NL-Dij']  = [2007]

    return years

#===============================================================================
def build_fgap_dict():
#===============================================================================
    #list of yield gap factors
    fgap = dict()
    fgap['Winter wheat'] = dict()
    fgap['Grain maize']  = dict()
    # winter wheat fgap
    fgap['Winter wheat']['BE-Lon'] = dict()
    fgap['Winter wheat']['BE-Lon'][2004] = 0.8
    fgap['Winter wheat']['BE-Lon'][2005] = 0.8
    fgap['Winter wheat']['BE-Lon'][2006] = 0.8
    fgap['Winter wheat']['BE-Lon'][2007] = 0.8
    fgap['Winter wheat']['DE-Kli'] = dict()
    fgap['Winter wheat']['DE-Kli'][2005] = 0.8
    fgap['Winter wheat']['DE-Kli'][2006] = 0.8
    fgap['Winter wheat']['DE-Kli'][2007] = 0.8
    fgap['Winter wheat']['FR-Aur'] = dict()
    fgap['Winter wheat']['FR-Aur'][2005] = 0.8
    fgap['Winter wheat']['FR-Aur'][2006] = 0.8
    fgap['Winter wheat']['FR-Gri'] = dict()
    fgap['Winter wheat']['FR-Gri'][2005] = 0.8
    fgap['Winter wheat']['FR-Gri'][2006] = 0.8
    fgap['Winter wheat']['FR-Lam'] = dict()
    fgap['Winter wheat']['FR-Lam'][2006] = 0.8
    fgap['Winter wheat']['FR-Lam'][2007] = 0.8
    # grain maize fgap
    fgap['Grain maize']['DE-Kli'] = dict()
    fgap['Grain maize']['DE-Kli'][2007] = 0.8 
    fgap['Grain maize']['FR-Gri'] = dict()
    fgap['Grain maize']['FR-Gri'][2005] = 0.8
    fgap['Grain maize']['FR-Lam'] = dict()
    fgap['Grain maize']['FR-Lam'][2006] = 0.8
    fgap['Grain maize']['IT-BCi'] = dict()
    fgap['Grain maize']['IT-BCi'][2004] = 0.8
    fgap['Grain maize']['NL-Dij'] = dict()
    fgap['Grain maize']['NL-Dij'][2007] = 0.8
    fgap['Grain maize']['NL-Lan'] = dict()
    fgap['Grain maize']['NL-Lan'][2005] = 0.8

    return fgap

#===============================================================================
# function that will retrieve the surface temperature from the ECMWF data
# (ERA-interim). It will return two arrays: one of the time in seconds since
# 1st of Jan, and one with the tsurf variable in K.
def retrieve_ecmwf_tsurf(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    ecmwfdir_tsurf = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/'+\
                     'eur100x100/'

    tsurf = np.array([])
    time  = np.array([])

    for month in range (1,13):
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 't_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_tsurf,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir_tsurf,'%i/%02d'%(year,month),namefile)
            f = cdf.Dataset(pathfile)

            # retrieve closest latitude and longitude index of desired location
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the temperature at the highest pressure level, at that 
            # lon,lat location 
            #print f.variables['ssrd'] # to get the dimensions of the variable
            tsurf = np.append(tsurf, f.variables['T'][0:8, 0, latindx, lonindx])
            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]
            # NB: the file has 8 time steps (3-hourly)
            time = np.append(time, f.variables['time'][:] - convtime)  
            f.close()

    if (len(time) < 2920):
        print '!!!WARNING!!!'
        print 'there are less than 365 days of data that we could retrieve'
        print 'check the folder %s for year %i'%(ecmwfdir_tsurf, year)
 
    return time, tsurf

#===============================================================================
# function that will retrieve the incoming surface shortwave radiation from the
# ECMWF data (ERA-interim). It will return two arrays: one of the time in
# seconds since 1st of Jan, and one with the ssrd variable in W.m-2.
def retrieve_ecmwf_ssrd(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    ecmwfdir_ssrd = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/sfc/glb100x100/'

    ssrd = np.array([])
    time = np.array([])

    for month in range (1,13):
        print year, month
        for day in range(1,32):

            # open file if it exists
            namefile = 'ssrd_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(ecmwfdir_ssrd,'%i/%02d'%(year,month),
                                                             namefile))==False):
                print 'cannot find %s'%namefile
                continue
            pathfile = os.path.join(ecmwfdir_ssrd,'%i/%02d'%(year,month),namefile)
            f = cdf.Dataset(pathfile)

            # retrieve closest latitude and longitude index of desired location
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the shortwave downward surface radiation at that location 
            #print f.variables['ssrd'] # to get the dimensions of the variable
            ssrd = np.append(ssrd, f.variables['ssrd'][0:8, latindx, lonindx])
            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]
            # NB: the file has 8 time steps (3-hourly)
            time = np.append(time, f.variables['time'][:] - convtime)  
            f.close()

    if (len(time) < 2920):
        print '!!!WARNING!!!'
        print 'there are less than 365 days of data that we could retrieve'
        print 'check the folder %s for year %i'%(ecmwfdir_ssrd, year)
 
    return time, ssrd

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
