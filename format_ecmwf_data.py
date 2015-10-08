#!/usr/bin/env python

import sys, os
import numpy as np

#===============================================================================
# This script formats ECMWF weather data stored on capegrim 
def main():
#===============================================================================
    """
    This script uses the ECMWF weather data stored on capegrim to create weather
    files on the finer CGMS grid scale. We format the weather files with the CABO
    format, i.e. the standard Wageningen crop model weather file format.
    --> the generated weather files can be used with most Wageningen crop models
        like SUCROS, WOFOST, GECROS...

    NB: only European CGMS grid cells containing arable land will be looped over
    to create the weather files. Like this it reduces the number of files being
    created.

    User input is the list of years for which we want to create weather files.
    You can also switch on parallelization (recommended).
    Created files will be stored in folder:
    /Users/mariecombe/mnt/promise/CO2/marie/CABO_weather_ECMWF/
    """
#-------------------------------------------------------------------------------
    from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
    from datetime import datetime
    from cPickle import load as pickle_load
#-------------------------------------------------------------------------------
    global dir_sfc, dir_tropo, year, all_grids, lons, lats, weatherdir
#-------------------------------------------------------------------------------
# USER INPUT - only this part should be modified by the user!!

    years    = [2006]      # list of years we want to retrieve data for
    process  = 'parallel'  # multiprocessing option: can be 'serial' or 'parallel'
    nb_cores = 10          # number of cores used in case of a parallelization

#-------------------------------------------------------------------------------
# Calculate key variables from the user input:

    campaign_years = np.linspace(int(years[0]),int(years[-1]),len(years))

#-------------------------------------------------------------------------------
# Define working directories

    # capegrim directories where the ECMWF weather data is stored
    dir_sfc     = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/sfc/glb100x100/'
    dir_tropo   = '/Storage/TM5/METEO/tm5-nc/ec/ei/fc012up2tr3/tropo25/'+\
                  'eur100x100/'

    # capegrim directory where the list of CGMS grid cell coordinates is stored
    EUROSTATdir = "/Users/mariecombe/Cbalance/EUROSTAT_data"

    # capegrim directory where the formated weather file will be written
    weatherdir  = '/Users/mariecombe/mnt/promise/CO2/marie/CABO_weather_ECMWF/'

    # capegrim directory where the formated weather file will be written
    pickledir   = '/Users/mariecombe/mnt/promise/CO2/marie/pickled_CGMS_input_data/'

#-------------------------------------------------------------------------------
# we read the CGMS grid cells coordinates from file

    CGMS_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
    all_grids  = CGMS_cells['CGMS_grid_list.csv']['GRID_NO']
    lons       = CGMS_cells['CGMS_grid_list.csv']['LONGITUDE']
    lats       = CGMS_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# From this list, we select the subset of grid cells located in Europe that
# contain arable land (no need to create weather data where there are no crops!)

    filename = pickledir + 'europe_arable_CGMS_cellids.pickle'
    europ_arable = pickle_load(open(filename,'rb'))    
    europ_arable = sorted(europ_arable)

#-------------------------------------------------------------------------------
# We add a timestamp at start of the script
    start_timestamp = datetime.utcnow()

#-------------------------------------------------------------------------------
#   WE LOOP OVER ALL YEARS:
    for y, year in enumerate(campaign_years): 
        print '######################## Year %i ########################\n'%year
        europ_cultivated = np.array([])

#-------------------------------------------------------------------------------
# START OF PARALLELIZATION HERE !!!!!!!
#-------------------------------------------------------------------------------
# if we do a serial iteration, we loop over the grid cells that contain arable 
# land
        if (process == 'serial'):
            for grid in europ_arable:
                format_weather(grid)

#-------------------------------------------------------------------------------
# if we do a parallelization, we use the multiprocessor module to provide series
# of cells to the function

        if (process == 'parallel'):
            import multiprocessing
            p = multiprocessing.Pool(nb_cores)
            parallel = p.map(format_weather, europ_arable)
            p.close()

#-------------------------------------------------------------------------------
# We add a timestamp at end of the retrieval, to time the process
    end_timestamp = datetime.utcnow()
    print '\nDuration of the weather formatting:', end_timestamp-start_timestamp


### END OF THE MAIN CODE ###

#===============================================================================
# function that will format ECMWF weather data into CABO format
def format_weather(grid_id):
#===============================================================================
# if the file already exists, we do nothing
    filename = '%i.%s'%(grid_id,str(year)[1:4])
    if os.path.exists(os.path.join(weatherdir,filename)):
        print 'File already exists! %s\n'%filename
        pass
#-------------------------------------------------------------------------------
    else:
# we get the lon and lat of the grid cell:
        i   = np.argmin(np.absolute(all_grids - grid_id))
        lon = lons[i]
        lat = lats[i]
        print '- grid cell no %i: lon = %.2f , lat = %.2f'%(grid_id,lon,lat)

#-------------------------------------------------------------------------------
# We open the weather data and compile daily information about the ECMWF grid
# point closest to the provided coordinates

        # incoming shortwave radiation from W.m-2 to kJ.m-2.d-1
        rad = retrieve_ecmwf_sfc_data('ssrd', year, lon, lat, ope='integral')
        rad = (rad[0], [x/1000. for x in rad[1]])
 
        # maximum and minimum 2-m temperature from K to degree C
        tmin = retrieve_ecmwf_sfc_data('t2m', year, lon, lat, ope='min')
        tmax = retrieve_ecmwf_sfc_data('t2m', year, lon, lat, ope='max')
        tmin = (tmin[0], [x - 273.15 for x in tmin[1]])
        tmax = (tmax[0], [x - 273.15 for x in tmax[1]])
 
        # daily precipitation from m.s-1 to mm.d-1
        lsp  = retrieve_ecmwf_sfc_data('lsp', year, lon, lat, ope='integral')
        cp   = retrieve_ecmwf_sfc_data('cp' , year, lon, lat, ope='integral')
        precip =  cp[1] + lsp[1]
        precip = (lsp[0], [x*1000. for x in precip])
 
        # diurnal mean wind speed in m.s-1
        m10m = retrieve_U_V_calculate_m10m(year, lon, lat)
 
        # 2-m vapor pressure
        VP = retrieve_q_calculate_VP(year, lon, lat)

#-------------------------------------------------------------------------------
# We write a formatted output file
        w = write_CABO_weather_file(grid_id,lon,lat,year,rad[1],tmin[1],
                                    tmax[1],VP[1],m10m[1],precip[1],weatherdir)

    return None

#===============================================================================
def write_CABO_weather_file(grid_no,lon,lat,year,rad,tmin,tmax,vp,ws,prec,dir_):
#===============================================================================
    '''
    this function writes a WOFOST formated weather file for one grid point.

    Arguments:
    ----------
    grid_no   integer, it is the CGMS grid cell number
    lon       float, longitude of the grid cell center point
    lat       float, latitude of the grid cell center point
    year      integer, year of the weather record
    rad       array of floats, containing the record of shortwave incoming 
              radiation in kJ m-2 d-1
    tmin      array of floats, containing the record of minimum temperature
              in degrees Celsius
    tmax      array of floats, containing the record of maximum temperature
              in degrees Celsius
    vp        array of floats, contains the daily average vapor pressure in kPa
    ws        array of floats, contains the daily average wind speed in m.s-1
    prec      array of floats, contains the daily precipitation in mm.d-1
    dir_      string, is the directory where the output file should be saved

    '''

    # check your assumptions on the weather record being provided...
    assert len(rad)==len(tmin)==len(tmax)==len(vp)==len(ws)==len(prec),\
           'the weather variables have different dimensions! check your arrays'
    assert 365<=len(rad)<367, 'the weather variables do not contain 365 days'

    # NB: I need to find a way of getting the elevation of the site
    elev = 0.

    # the formula to calculate the Angstrom parameters taken from WOFOST 6.0
    # user guide ()
    AngA = 0.4885 - 0.0052*lat #0.18 = indicative value for cold temperate region
    AngB = 0.1563 + 0.0074*lat #0.55 = indicative value for cold temperate region

    filename = '%i.%s'%(grid_no,str(year)[1:4])
    filepath = os.path.join(dir_,filename)
    if os.path.exists(filepath):
        print '\nRemoving old file %s'%filepath
        os.remove(filepath)
    print 'Creating new file %s'%filepath
    output   = open(filepath,'w')

    output.write('*---------------------------------------------------------*\n')
    output.write('*  Longitude : %5.2f E                     \n'%lon)
    output.write('*  Latitude  : %5.2f N                     \n'%lat)
    output.write('*  Elevation : unknown                     \n')
    output.write('*  Year      : %i                          \n'%year)
    output.write('*  Source    : ECMWF Era-Interim           \n')
    output.write('*  Author    : Marie Combe                 \n')
    output.write('*                                          \n')
    output.write('*  First line: Lon Lat Elev AngA AngB      \n')
    output.write('*  AngA and AngB are empirical Angstrom    \n')
    output.write('*  coefficients for making global radiation\n')
    output.write('*  estimates                               \n')
    output.write('*                                          \n')
    output.write('*  Columns:                                \n')
    output.write('*  ========                                \n')
    output.write('*  CGMS grid cell ID number                \n')
    output.write('*  year                                    \n')
    output.write('*  DOY                                     \n')
    output.write('*  irradiation (kJ m-2 d-1)                \n')
    output.write('*  minimum temperature (degrees Celsius)   \n')
    output.write('*  maximum temperature (degrees Celsius)   \n')
    output.write('*  vapour pressure (kPa)                   \n')
    output.write('*  mean wind speed (m s-1)                 \n')
    output.write('*  precipitation (mm d-1)                  \n')
    output.write('*---------------------------------------------------------*\n')
    output.write('%8.2f %8.2f %8.0f %8.2f %8.2f\n'%(lon, lat, elev, AngA, AngB))

    for i in range(len(rad)): # loop over the number of days
        output.write(' %6d  %5i  %4d  %6.0f'%(grid_no,year,i+1,rad[i]))
        output.write(' %6.2f  %6.2f  %5.2f'%(tmin[i],tmax[i],vp[i]))
        output.write(' %5.2f  %6.2f\n'%(ws[i],prec[i]))

    output.close()
    print 'Success! Saved file %s in folder %s\n'%(filename, dir_)
    return None

#===============================================================================
# function that will retrieve a weather variable from the ECMWF re-analysis
# dataset (ERA-interim). It will return two arrays: one of the time in seconds
# since 1st of Jan, and one with the variable.
def retrieve_q_calculate_VP(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf

    vp  = np.array([])
    time = np.array([])

    for month in range (1,13):
        for day in range(1,32):

            # open file if it exists
            namefile1 = 'q_%i%02d%02d_00p03.nc'%(year,month,day)
            namefile2 = 'sp_%i%02d%02d_00p03.nc'%(year,month,day)
            if (os.path.exists(os.path.join(dir_tropo,'%i/%02d'%(year,month),
                                                            namefile1))==False):
                continue
            if (os.path.exists(os.path.join(dir_tropo,'%i/%02d'%(year,month),
                                                            namefile2))==False):
                continue
            pathfile1 = os.path.join(dir_tropo,'%i/%02d'%(year,month),namefile1)
            pathfile2 = os.path.join(dir_tropo,'%i/%02d'%(year,month),namefile2)
            f1 = cdf.Dataset(pathfile1)
            f2 = cdf.Dataset(pathfile2)

            # retrieve closest latitude and longitude index of desired location
            lats = f1.variables['lat'][:]
            lons = f1.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f1.variables['time'][0]

            # retrieve q at that location, convert to vapor pressure in kPa, do 
            # a diurnal mean
            e = f1.variables['Q'][0:8, 0, latindx, lonindx] * \
                f2.variables['sp'][0:8, latindx, lonindx]/1000. / 0.622
            vp = np.append(vp, np.mean(e))
            time = np.append(time, f1.variables['time'][4] - convtime)  

            f1.close()
            f2.close()

    assert (365<=len(time)<367), 'variable q has less than 365 days we could'+\
        'retrieve\ncheck the folder %s/%i'%(dir_tropo, year)

    print 'Successfully retrieved the mean VP record for year %i'%year 
    return time, vp

#===============================================================================
# function that will retrieve a weather variable from the ECMWF re-analysis
# dataset (ERA-interim). It will return two arrays: one of the time in seconds
# since 1st of Jan, and one with the variable.
def retrieve_U_V_calculate_m10m(year, lon, lat):
#===============================================================================

    import netCDF4 as cdf
    import math

    var  = np.array([])
    time = np.array([])

    for month in range (1,13):
        for day in range(1,32):

            # open file if it exists
            namefile1 = 'u10m_%i%02d%02d_00p03.nc'%(year,month,day)
            namefile2 = 'v10m_%i%02d%02d_00p03.nc'%(year,month,day)
            if ((os.path.exists(os.path.join(dir_sfc,'%i/%02d'%(year,month),
            namefile1))==False) or (os.path.exists(os.path.join(dir_sfc,
            '%i/%02d'%(year,month), namefile2))==False)):
                continue
            pathfile1 = os.path.join(dir_sfc,'%i/%02d'%(year,month),namefile1)
            pathfile2 = os.path.join(dir_sfc,'%i/%02d'%(year,month),namefile2)
            f1 = cdf.Dataset(pathfile1)
            f2 = cdf.Dataset(pathfile2)

            # retrieve closest latitude and longitude index of desired location
            lats = f1.variables['lat'][:]
            lons = f1.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f1.variables['time'][0]

            # retrieve the variable at that location, do a diurnal mean, min, max,
            # sum if necessary 
            # NB: if ope != 'None', we write one time per day: 12 hUTC,
            # otherwise we write the 8 time steps (3-hourly data)
            usquare = [math.pow(u,2) for u in 
                       f1.variables['u10m'][0:8, latindx, lonindx]]
            vsquare = [math.pow(v,2) for v in 
                       f2.variables['v10m'][0:8, latindx, lonindx]]
            mw   = usquare + vsquare
            mw   = [math.pow(m,0.5) for m in mw] # mean wind vector
            var  = np.append(var, np.mean(mw))
            time = np.append(time, f1.variables['time'][4] - convtime)  

            f1.close()
            f2.close()

    assert (365<=len(time)<367), 'variable u10m or v10m have less than 365 '+\
        'days we could retrieve\ncheck the folder %s/%i'%(dir_sfc, year)
 
    print 'Successfully retrieved the mean M record for year %i'%year 
    return time, var

#===============================================================================
# function that will retrieve a weather variable from the ECMWF re-analysis
# dataset (ERA-interim). It will return two arrays: one of the time in seconds
# since 1st of Jan, and one with the variable.
def retrieve_ecmwf_sfc_data(varname, year, lon, lat, ope='None'):
#===============================================================================

    import netCDF4 as cdf

    var  = np.array([])
    time = np.array([])

    for month in range (1,13):
        for day in range(1,32):

            # open file if it exists
            namefile = '%s_%i%02d%02d_00p03.nc'%(varname,year,month,day)
            if (os.path.exists(os.path.join(dir_sfc,'%i/%02d'%(year,month),
                                                             namefile))==False):
                continue
            pathfile = os.path.join(dir_sfc,'%i/%02d'%(year,month),namefile)
            f = cdf.Dataset(pathfile)

            # retrieve closest latitude and longitude index of desired location
            lats = f.variables['lat'][:]
            lons = f.variables['lon'][:]
            latindx = np.argmin( np.abs(lats - lat) )
            lonindx = np.argmin( np.abs(lons - lon) )

            # retrieve the nb of seconds on day 1 of that year
            if (month ==1 and day ==1):
                convtime = f.variables['time'][0]

            # retrieve the variable at that location, do a diurnal mean, min, max,
            # sum if necessary 
            # NB: if ope != 'None', we write one time per day: 12 hUTC,
            # otherwise we write the 8 time steps (3-hourly data)
            if (ope == 'mean'):
                var = np.append(var, np.mean(f.variables[varname][0:8, latindx,
                                                                       lonindx]))
                time = np.append(time, f.variables['time'][4] - convtime)  
            elif (ope == 'min'):
                var = np.append(var, np.min(f.variables[varname][0:8, latindx,
                                                                       lonindx]))
                time = np.append(time, f.variables['time'][4] - convtime)  
            elif (ope == 'max'):
                var = np.append(var, np.max(f.variables[varname][0:8, latindx,
                                                                       lonindx]))
                time = np.append(time, f.variables['time'][4] - convtime)  
            elif (ope == 'integral'):
                dt  = f.variables['time'][1]-f.variables['time'][0]
                var = np.append(var, np.sum(f.variables[varname][0:8, latindx,
                                                                    lonindx])*dt)
                time = np.append(time, f.variables['time'][4] - convtime)  
            elif (ope == 'None'):
                var = np.append(var, f.variables[varname][0:8, latindx, lonindx])
                time = np.append(time, f.variables['time'][:] - convtime)  
            else:
                print "you need to specify the argument 'ope' in function retrieve"+\
                      "_ecmwf_data"
                sys.exit(2)

            f.close()

    if ope == 'None':
        assert (2920<=len(time)<2936), 'variable % has less than 365 '%varname+\
            'days we could retrieve\ncheck the folder %s/%i'%(dir_sfc, year)
    else:
        assert (365<=len(time)<367), 'variable % has less than 365 '%varname+\
            'days we could retrieve\ncheck the folder %s/%i'%(dir_sfc, year)
 
    print 'Successfully retrieved the %s %s record for year %i'%(ope,varname,year) 
    return time, var

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
