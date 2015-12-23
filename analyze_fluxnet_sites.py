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
    global EUROSTATdir, FluxNetdir, inputdir, all_grids, lons, lats
#-------------------------------------------------------------------------------
# flags to run parts of the script only
    obs_plot        = False
    forward_sim     = False
    plot_obs_vs_sim = False
#-------------------------------------------------------------------------------
# Define general working directories
    EUROSTATdir = '../observations/EUROSTAT_data/'
    FluxNetdir  = '../observations/FluxNet_data/'
    inputdir    = '../model_input_data/'
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
    if obs_plot == True:
        for crop in cover.keys():
            plot_fluxnet_daily_fluxes(crop, cover[crop], years[crop], FluxNetdir)

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

        print 'FluxNet site %s: closest grid cell is %i'%(site, all_grids[indx])

#-------------------------------------------------------------------------------
# Simulate the optimize GPP, Reco and NEE for the FluxNet sites

    # we retrieve the crop id numbers
    crop_dict = map_crop_id_to_crop_name(cover.keys())
    # loop over crops
    for crop in cover.keys():
        crop_no = crop_dict[crop][0]
        print '\n %s'%crop
        # loop over sites
        for site in cover[crop]:
            print '\n%s: lon = %.2f, lat = %.2f, closest cell = %i'%(site, lon, 
                                                                 lat, grid_no)
            # get the closest grid cell id number:
            grid_no = int(flux_gri[site])
            # get the longitute and latitute of the site:
            indx = flux_nam.index(site)
            lon = flux_lon[indx]
            lat = flux_lat[indx]
            # stuff to print to screen
            print '\nYLDGAPF(-),  grid_no,  year,  stu_no, stu_area(ha), '\
                 +'TSO(kgDM.ha-1), TLV(kgDM.ha-1), TST(kgDM.ha-1), '\
                 +'TRT(kgDM.ha-1), maxLAI(m2.m-2), rootdepth(cm), TAGP(kgDM.ha-1)'
            for year in years[crop][site]:
                if (grid_no == 107097): continue # I forgot to retrieve input
                                                 # data for the Dijkgraaf site
                # OPTIMIZATION OF FGAP:
                yldgapf = fgap[crop][site][year]
                # FORWARD SIMULATIONS:
                if (forward_sim == True):
                    perform_yield_sim(crop_no, grid_no, int(year), yldgapf)
                # POST-PROCESSING OF GPP, RAUTO:
                time_series = compile_nee(grid_no, culti_land, year)
            
#-------------------------------------------------------------------------------
# plot the fluxes of GPP, TER, NEE for each site
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
                # get the closest grid cell id number:
                for year in years[crop][site]:
                    # plot one time series of observed fluxes obs + simulated    
                    plot_nee_time_series()
                    plt.show()

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
    inputdir = '../model_input_data/CGMS/'
    caboecmwfdir = '../model_input_data/CABO_weather_ECMWF/'
    outputdir = '../model_output/FluxNet_sites_output/'
#-------------------------------------------------------------------------------
    # Retrieve the weather data of one grid cell
    if (weather == 'CGMS'):
        filename = inputdir+'weatherobject_g%d.pickle'%grid_no
        weatherdata = WeatherDataProvider()
        weatherdata._load(filename)
    if (weather == 'ECMWF'):
        weatherdata = CABOWeatherDataProvider('%i'%(grid_no), 
                                                         fpath=caboecmwfdir)
    #print weatherdata(datetime.date(datetime(2006,4,1)))

    # Retrieve the soil data of one grid cell 
    filename = inputdir+'soilobject_g%d.pickle'%grid_no
    soil_iterator = pickle_load(open(filename,'rb'))

    # Retrieve calendar data of one year for one grid cell
    filename = inputdir+'timerobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    timerdata = pickle_load(open(filename,'rb'))
                    
    # Retrieve crop data of one year for one grid cell
    filename = inputdir+'cropobject_g%d_c%d_y%d.pickle'%(grid_no,crop_no,year)
    cropdata = pickle_load(open(filename,'rb'))

    # retrieve the fgap data of one year and one grid cell
    cropdata['YLDGAPF'] = fgap

    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no], inputdir, 
                                       method=selec_method, n=nsoils)

    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        
        # Retrieve the site data of one year, one grid cell, one soil type
        filename = inputdir+'siteobject_g%d_c%d_y%d_s%d.pickle'%(grid_no,crop_no,
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
def compile_nee(grid_no, culti_land, year):
#===============================================================================
# We retrieve the longitude and latitude of the CGMS grid cell
        i   = np.argmin(np.absolute(all_grids - grid_no))
        lon = lons[i]
        lat = lats[i]
        print '- grid cell no %i: lon = %.2f , lat = %.2f'%(grid_no,lon,lat)

#-------------------------------------------------------------------------------
# We open the incoming surface shortwave radiation [W.m-2] 

    filename_rad = 'rad_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(opti_year,lon,lat)
    path_rad     = os.path.join(inputdir,filename_rad)
    if os.path.exists(path_rad):
        rad = pickle_load(open(path_rad, 'rb'))
    else:
        from 06_complete_c_cycle import retrieve_ecmwf_ssrd
        rad = retrieve_ecmwf_ssrd(opti_year, lon, lat)
        pickle_dump(rad,open(path_rad, 'wb'))

#-------------------------------------------------------------------------------
# We open the surface temperature record

    filename_ts = 'ts_ecmwf_%i_lon%.2f_lat%.2f.pickle'%(opti_year,lon,lat)
    path_ts     = os.path.join(inputdir,filename_ts)
    if os.path.exists(path_ts):
        ts = pickle_load(open(path_ts, 'rb'))
    else:
        from 06_complete_c_cycle import retrieve_ecmwf_tsurf
        ts = retrieve_ecmwf_tsurf(opti_year, lon, lat)
        pickle_dump(ts,open(path_ts, 'wb'))

#-------------------------------------------------------------------------------
# we initialize the timeseries of gpp and Resp for the grid cell

    time_cell_timeseries = rad[0]
    len_cell_timeseries  = len(rad[0])
    gpp_cell_timeseries  = np.array([0.]*len_cell_timeseries)
    raut_cell_timeseries = np.array([0.]*len_cell_timeseries)
    rhet_cell_timeseries = np.array([0.]*len_cell_timeseries)
    sum_stu_areas        = 0.

    if (prod_figure == True):
        from matplotlib import pyplot as plt
        plt.close('all')
        fig1, axes = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
        fig1.subplots_adjust(0.1,0.1,0.98,0.9,0.2,0.2)

#-------------------------------------------------------------------------------
    # Select soil types to loop over for the forward runs
    selected_soil_types = select_soils(crop_no, [grid_no],
                           pickledir, method=selec_method, n=nsoils)
#-------------------------------------------------------------------------------
#       WE NEED TO LOOP OVER THE SOIL TYPE
    for smu, stu_no, stu_area, soildata in selected_soil_types[grid_no]:
        print grid_no, stu_no
#-------------------------------------------------------------------------------
# We open the WOFOST results file

        filelist    = 'pcse_output_c%i_g%i_s%i_y%i.csv'\
                       %(crop_no, grid_no, stu_no, opti_year) 
        wofost_data = open_pcse_csv_output(pcse_outputdir, [filelist])

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

        for DOY, timeinsec in enumerate(time_cell_timeseries[::8]):
            #print 'doy, timeinsec:', DOY, timeinsec

            # conversion of current time in seconds into date
            time = datetime.date(opti_year,1,1) + datetime.timedelta(DOY)
            #print 'date:', time

            # we test to see if we are within the growing season
            test_sow = (time - wofost_data[filelist]['day'][0]).total_seconds()
            test_rip = (time - wofost_data[filelist]['day'][-1]).total_seconds() 
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
                gpp_day    = - wofost_data[filelist]['GASS'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                maint_resp = wofost_data[filelist]['MRES'][index_day_w] * \
                                                        (mmC / mmCH2O) * 0.1
                try: # if there are any available assimilates for growth
                    growth_fac = (wofost_data[filelist]['DMI'][index_day_w]) / \
                             (wofost_data[filelist]['GASS'][index_day_w] - 
                              wofost_data[filelist]['MRES'][index_day_w])
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
            # the sum of the 8 rates is equal to total/dt:
            dt             = 3600. * 3.
            sum_gpp_rates  = gpp_day   / dt
            sum_raut_rates = raut_day  / dt 
            # the day's 8 values of actual gpp and raut rates per second:
            gpp_cycle      = weights * sum_gpp_rates
            raut_cycle     = weights * sum_raut_rates
            # NB: we check if the applied diurnal cycle is correct
            assert (sum(weights)-1. < 0.000001), "wrong radiation kernel"
            assert (len(gpp_cycle)*int(dt) == 86400), "wrong dt in diurnal cycle"
            assert ((sum(gpp_cycle)*dt-gpp_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on GPP: residual=%.2f "%(sum(gpp_cycle)*dt-gpp_day) +\
                "on DOY %i"%DOY
            assert ((sum(raut_cycle)*dt-raut_day) < 0.00001), "wrong diurnal cycle "+\
                "applied on Rauto: residual=%.2f "%(sum(raut_cycle)*dt-raut_day) +\
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
                ax.plot(time_cell_timeseries[::8]/(3600.*24.), var, 
                                                      label='stu %i'%stu_no)
                ax.set_xlim([40.,170.])
                ax.set_ylim(lims)
                ax.set_ylabel(name + r' (g$_{C}$ m$^{-2}$ d$^{-1}$)', fontsize=14)

#-------------------------------------------------------------------------------
# We add the gpp of all soil types in the grid cell. NB: different calendars
# are applied depending on the site!! so the sowing and maturity dates might
# differ from stu to stu

        # sum the timeseries of GPP and Rauto for all soil types
        gpp_cell_timeseries  = gpp_cell_timeseries  + gpp_cycle_timeseries*stu_area
        raut_cell_timeseries = raut_cell_timeseries + raut_cycle_timeseries*stu_area

    # finish the figure of multiple stu carbon fluxes
    if (prod_figure == True):
        plt.xlabel('time (DOY)', fontsize=14)
        plt.legend(loc='upper left', ncol=2, fontsize=10)
        fig1.suptitle('Daily carbon fluxes of %s for all '%crop_name+\
                     'soil types of grid cell %i (%s) in %i'%(grid_no,
                                             NUTS_no, opti_year), fontsize=18)
        figname = 'GPP_perday_c%s_%s_y%i_g%i.png'%(crop_no,NUTS_no,opti_year,\
                                                                    grid_no)
        fig1.savefig(figname)

    # for each grid cell, calculate the weighted average GPP and Rauto
    gpp_cell_timeseries  = gpp_cell_timeseries  / sum_stu_areas
    raut_cell_timeseries = raut_cell_timeseries / sum_stu_areas

#-------------------------------------------------------------------------------
# We calculate the heterotrophic respiration with the surface temperature

    # from the A-gs model:
    # pb: we need to simulate wg with that approach...
    #fw = Cw * wsmax / (wg + wsmin)
    tsurf_inter = Eact0 / (283.15 * 8.314) * (1 - 283.15 / ts[1])
    rhet_cell_timeseries = R10 * np.array([ math.exp(t) for t in tsurf_inter ]) 
    # conversion from mgCO2/m2/s to gC/m2/s
    rhet_cell_timeseries = rhet_cell_timeseries * (mmC / mmCO2) * 0.001

#-------------------------------------------------------------------------------
# We calculate NEE as the net flux

    nee_cell_timeseries = gpp_cell_timeseries + raut_cell_timeseries +\
                          rhet_cell_timeseries

#-------------------------------------------------------------------------------
    return gpp_cell_timeseries, raut_cell_timeseries, rhet_cell_timeseries,\
           nee_cell_timeseries

#===============================================================================
def plot_fluxnet_daily_fluxes(crop, site_names, years_dict, FLUXNETdir):
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
def plot_nee_time_series():
#===============================================================================

    from matplotlib import pyplot as plt
    fig2 = plt.figure(figsize=(14,6))
    fig2.subplots_adjust(0.1,0.2,0.98,0.85,0.4,0.6)
    plt.plot(time_cell_timeseries/(3600.*24.),gpp_cell_timeseries*1000., 
                                                 label=r'$\mathrm{GPP}$', c='g')
    plt.plot(time_cell_timeseries/(3600.*24.),raut_cell_timeseries*1000.,
                                              label=r'$R_{aut}$', c='b')
    plt.plot(time_cell_timeseries/(3600.*24.),rhet_cell_timeseries*1000.,
                                              label=r'$R_{het}$', c='r')
    plt.plot(time_cell_timeseries/(3600.*24.),nee_cell_timeseries*1000., 
                                                 label=r'$\mathrm{NEE}$', c='k')
    plt.xlim([0.,365.])
    plt.ylim([-1.,1.])
    plt.xlabel('time (DOY)')
    plt.ylabel(r'carbon flux (mg$_{C}$ m$^{-2}$ s$^{-1}$)')
    plt.legend(loc='best', ncol=2, fontsize=10)
    plt.title('Average carbon fluxes of %s over the  '%crop_name+\
                 'cultivated area of grid cell %i (%s) in %i'%(grid_no,
                                                    NUTS_no, opti_year))
    return None

#===============================================================================
def build_years_dict():
#===============================================================================
    years = dict()
    years['Winter wheat'] = dict()
    years['Grain maize']  = dict()
    # list of years for winter wheat sites:
    years['Winter wheat']['BE-Lon'] = [2004,2005,2006,2007]
    years['Winter wheat']['DE-Kli'] = [2005,2006,2007]
    years['Winter wheat']['FR-Aur'] = [2005,2006]
    years['Winter wheat']['FR-Gri'] = [2005,2006]
    years['Winter wheat']['FR-Lam'] = [2006,2007]
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
if __name__=='__main__':
    main()
#===============================================================================
