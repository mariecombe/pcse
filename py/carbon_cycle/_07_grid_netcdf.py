#!/usr/bin/env python

import sys
from os import path
sys.path.append( path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
import os
import numpy as np
from numpy.core import multiarray

import logging as mylogger
from py.tools.initexit import start_logger, parse_options
from py.carbon_cycle.maries_toolbox import get_crop_names
import py.tools.rc as rc

import cPickle
import datetime as dt

#===============================================================================
def main():
#===============================================================================
    """
    This method postprocesses WOFOST output (daily carbon pools increment) into 
    3-hourly crop surface CO2 exchange: we use radiation data to create a
    diurnal cycle for photosynthesis (GPP) and autotrophic respiration (Rauto),
    and we add an heterotrophic respiration (Rhet) component.,sorted(Tsfiles)[0:100]

    The result of this post-processing is saved as pandas time series object in
    pickle files, for easy ploting purposes.

    """
#-------------------------------------------------------------------------------
    from maries_toolbox import open_csv
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# declare as global variables: folder names, main() method arguments, and a few
# variables/constants passed between functions
    global analysisdir, CGMSdir
#-------------------------------------------------------------------------------
# constant molar masses for unit conversion of carbon fluxes
    mmC    = 12.01
    mmCO2  = 44.01
    mmCH2O = 30.03 
    sec_per_day = 86400.
#-------------------------------------------------------------------------------
# Temporarily add the parent directory to python path, to be able to import pcse
# modules
    sys.path.insert(0, "..") 
#-------------------------------------------------------------------------------
# ================================= USER INPUT =================================
    _ = start_logger(level=mylogger.INFO)

    opts, args = parse_options()


    # First message from logger
    mylogger.info('Python Code has started')
    mylogger.info('Passed arguments:')
    for arg,val in args.iteritems():
        mylogger.info('         %s : %s'%(arg,val) )

    rcfilename = args['rc']
    rcF = rc.read(rcfilename)
    years = [int(rcF['year'])]
    outputdir = os.path.join(rcF['dir.project'],'output',str(years[0]))
    inputdir = rcF['dir.wofost.input']
    potential_sim = (rcF['fwd.wofost.potential'] in ['True','TRUE','true','T'])

    crops = ['Winter wheat','Spring wheat','Winter barley','Spring barley','Potato','Grain maize','Fodder maize','Rye','Sugar beet']  #WP We loop over all crops in this script

    # input data directory paths

# ==============================================================================
#-------------------------------------------------------------------------------
# Define general working directories
    cwdir       = os.getcwd()
    CGMSdir     = os.path.join(inputdir, 'CGMS')
#-------------------------------------------------------------------------------
# we read the coordinates of all possible CGMS grid cells from file

    gridX = cPickle.load(open(os.path.join(CGMSdir,'gridX.pickle'),'rb'))
    gridY = cPickle.load(open(os.path.join(CGMSdir,'gridY.pickle'),'rb'))
    gridlons = cPickle.load(open(os.path.join(CGMSdir,'gridlon.pickle'),'rb'))
    gridlats = cPickle.load(open(os.path.join(CGMSdir,'gridlat.pickle'),'rb'))
    gid   = cPickle.load(open(os.path.join(CGMSdir,'gridno_togridindex.pickle'),'rb'))
    cropweights = cPickle.load(open(os.path.join(CGMSdir,'fractions_cultivation.pickle'),'rb'))
    CGMSgrid = cPickle.load(open(os.path.join(CGMSdir,'CGMSgrid.pickle'),'rb'))

#-------------------------------------------------------------------------------
# PERFORM THE POST-PROCESSING:
#-------------------------------------------------------------------------------

    # build folder name from year and crop
    if potential_sim:
        subfolder = 'potential' 
    else:
        subfolder = 'optimized' 

    # create post-processing folder if needed
    analysisdir = os.path.join(outputdir,'gridded-nc', subfolder)

    if not os.path.exists(analysisdir):
        os.makedirs(analysisdir)
        mylogger.info('Created new folder: %s' %analysisdir )

    cropdict=get_crop_names([],method='all')

    saveas = os.path.join(analysisdir, 'griddedfluxes.nc')
    if os.path.exists(saveas):
        os.remove(saveas)

    import netCDF4 as cdf

    # Create global attributes

    rootgrp = cdf.Dataset(saveas, 'w', format='NETCDF4')
    mylogger.info("New file created: %s"%saveas)
    rootgrp.disclaimer = "This data belongs to the CarbonTracker project"
    rootgrp.email = "wouter.peters@wur.nl"
    rootgrp.institution = "Wageningen University"
    rootgrp.source 	= "Results calculated with the WOFOST crop model and EUROSTAT reported crop yield data"
    rootgrp.conventions = "CF-1.1"
    rootgrp.historytext	= 'created on '+dt.datetime.now().strftime('%B %d, %Y')+' by %s'%os.environ['USER']

    rootgrp.proj4string = "+proj=laea +lat_0=52 +lon_0=10 +x_0=4321000 +y_0=3210000 +ellps=GRS80 +units=m +no_defs"
    rootgrp.projepsg = '3035'

    # Create dimensions

    time = rootgrp.createDimension('time', None)
    xdim = rootgrp.createDimension('x', gridX.shape[1])
    ydim = rootgrp.createDimension('y', gridX.shape[0])
    longitudes = rootgrp.createDimension('lon', gridX.shape[1])
    latitudes = rootgrp.createDimension('lat', gridX.shape[0])
    ncrops = rootgrp.createDimension('crops', 16)

    # Create global variables + attributes

    times = rootgrp.createVariable('time','f8',('time',))
    x = rootgrp.createVariable('x','f4',('y','x'))
    y = rootgrp.createVariable('y','f4',('y','x',))
    lons = rootgrp.createVariable('lon','f4',('lat','lon'))
    lats = rootgrp.createVariable('lat','f4',('lat','lon',))

    # Projection details
    crs = rootgrp.createVariable('crs','i')
    crs.grid_mapping_name = "lambert_azimuthal_equal_area"
    crs.latitude_of_projection_origin = 52.0
    crs.longitude_of_projection_origin = 10.0
    crs.false_easting = 4321000.0 
    crs.false_northing = 3210000.0 

    y.units = 'meters'
    y.standard_name = "projection_y_coordinate"
    y.actual_range = "%f,%f"%(gridY.min(),gridY.max())
    y.comment = "lower-left corner of gridbox"
    y[:,:] = gridY

    lons.units = 'degrees_east'
    lons.comment = "lower-left corner of gridbox"
    lons[:,:] = gridlons

    x.units = 'meters'
    x.standard_name = "projection_x_coordinate"
    x.actual_range = "%f,%f"%(gridX.min(),gridX.max())
    x.comment = "lower-left corner of gridbox"
    x[:,:] = gridX

    lats.units = 'degrees_north'
    lats.comment = "lower-left corner of gridbox"
    lats[:,:] = gridlats

    times.units = 'days since 2000-01-01 00:00:00.0'
    times.calendar = 'gregorian'

    #
    # Create global variables that contain data summed/averaged over all crops
    #
    # GPP
    totgpp = rootgrp.createVariable('GPP','f4',('time','y','x'))
    totgpp.long_name = "Gross Primary Production"
    totgpp.units = 'gC box s-1'
    # TER
    totter = rootgrp.createVariable('TER','f4',('time','y','x'))
    totter.long_name = "Terrestrial Ecosystem Respiration"
    totter.units = 'gC box-1 s-1'
    # NEE
    totnee = rootgrp.createVariable('NEE','f4',('time','y','x'))
    totnee.long_name = "Net Ecosystem Respiration"
    totnee.units = 'gC box-1 s-1'
    # Cultivated area, can be used to go from fluxes/gridbox to fluxes/m2
    sumcultarea = rootgrp.createVariable('cultivated_area','f4',('y','x'))
    sumcultarea.long_name = "Total cultivated area of each gridbox"
    sumcultarea.comment = "Can be used to convert between fluxes in m-2 and fluxes in box -1"
    sumcultarea.units = 'm2'

    # Cultivated area, can be used to go from fluxes/gridbox to fluxes/m2
    arablearea = rootgrp.createVariable('arable_area','f4',('y','x'))
    arablearea.long_name = "Total arable area of each gridbox"
    arablearea.comment = "Can be used to convert between fluxes in m-2 and fluxes in box -1"
    arablearea.units = 'm2'
    
    
    # Aggregation arrays for loop over crops
    sumgpp=np.zeros((12,)+gridX.shape)
    sumter=np.zeros((12,)+gridX.shape)
    sumnee=np.zeros((12,)+gridX.shape)
    sumarea=np.zeros(gridX.shape)

# Now loop over the crops and process gridded fields per crop, as well as the weighted total

    for crop in crops:

        # Check for existence of files, otherwise skip crop
        pandaresultsdir = os.path.join(outputdir,crop.replace(' ','_'),'analysis',subfolder)
        yieldgapresultsdir = os.path.join(outputdir,crop.replace(' ','_'),'ygf')

        if not os.path.exists(pandaresultsdir):
            mylogger.info("Skipping crop without analysis folder (%s)"%crop)
            continue

        # Based on the name of the crop, find its CGMS ID number and its long name

        cropno , croplongname = cropdict[crop]

        # Create NetCDF group for results, and add variables

        cropname = crop.replace(' ','_').lower()  # derive the name with an underscore to prevent spaces in variable names
        cropgrp = rootgrp.createGroup(cropname)

        cropgrp.CGMS_ID = cropno
        cropgrp.long_name = croplongname

        cropyieldsum = cropgrp.createVariable('yield_sum','f4',('y','x'))
        cropyieldsum.source = pandaresultsdir
        cropyieldsum.units = 'gC m-2'

        cropmasssum = cropgrp.createVariable('biomass_sum','f4',('y','x'))
        cropmasssum.source = pandaresultsdir
        cropmasssum.units = 'gC'

        crophi = cropgrp.createVariable('harvest_index','f4',('y','x'))
        crophi.source = pandaresultsdir
        crophi.units = '-'

        cropyieldgap = cropgrp.createVariable('yieldgapfactor','f4',('y','x'))
        cropyieldgap.source = yieldgapresultsdir
        cropyieldgap.units = '-'
        cropyieldgap.comment = 'yield gap factor applied to WOFOST runs'

        cropnuts = cropgrp.createVariable('NUTSlevel','f4',('y','x'))
        cropnuts.source = pandaresultsdir
        cropnuts.units = '-'
        cropnuts.comment = 'Represents the NUTS level sampled for each box (0,1,2,3)'

        cropcultfrac = cropgrp.createVariable('cultivated_frac','f4',('y','x'))
        cropcultfrac.units = '-'
        cropcultfrac.comment = 'Fraction of arable land cultivated by this crop in each gridbox'

        cropgpp = cropgrp.createVariable('GPP','f4',('time','y','x'))
        cropgpp.source = pandaresultsdir
        cropgpp.long_name = "Gross Primary Production"
        cropgpp.units = 'gC m-2 s-1'

        cropter = cropgrp.createVariable('TER','f4',('time','y','x'))
        cropter.source = pandaresultsdir
        cropter.long_name = "Terrestrial Ecosystem Respiration"
        cropter.units = 'gC m-2 s-1'

        cropnee = cropgrp.createVariable('NEE','f4',('time','y','x'))
        cropnee.source = pandaresultsdir
        cropnee.long_name = "Net Ecosystem Respiration"
        cropnee.units = 'gC m-2 s-1'

        cropt2m = cropgrp.createVariable('sum_T2M','f4',('time','y','x'))
        cropt2m.source = pandaresultsdir
        cropt2m.long_name = "sum of t2m from ECMWF"
        cropt2m.units = 'K'

        croptsum = cropgrp.createVariable('sum_expT2M','f4',('time','y','x'))
        croptsum.source = pandaresultsdir
        croptsum.long_name = "exponential function of t2m, summed, to be multiplied with Eact0"
        croptsum.units = '-'

        cropssr = cropgrp.createVariable('sum_ssr','f4',('time','y','x'))
        cropssr.source = pandaresultsdir
        cropssr.long_name = "sum of ssr from ECMWF"
        cropssr.units = 'W m-2'

        cropagp = cropgrp.createVariable('abovegroundbiomass','f4',('time','y','x'))
        cropagp.source = pandaresultsdir
        cropagp.long_name = "Total Biomass Aboveground"
        cropagp.units = 'gC'

        croptwrt= cropgrp.createVariable('rootbiomass','f4',('time','y','x'))
        croptwrt.source = pandaresultsdir
        croptwrt.long_name = "Total Biomass in roots"
        croptwrt.units = 'gC'

        croptwso= cropgrp.createVariable('yield','f4',('time','y','x'))
        croptwso.source = pandaresultsdir
        croptwso.long_name = "yield"
        croptwso.units = 'gC m-2'

        cropstress= cropgrp.createVariable('fstress','f4',('time','y','x'))
        cropstress.source = pandaresultsdir
        cropstress.long_name = "stress factor for soil moisture, tra/tramx"
        cropstress.units = '-'

        # Make arrays needed to fill arrays specific to this crop

        masssumfield=np.zeros(gridX.shape)+np.NaN
        hifield=np.zeros(gridX.shape)+np.NaN
        yieldsumfield=np.zeros(gridX.shape)+np.NaN
        yieldgapfield=np.zeros(gridX.shape)+np.NaN
        nutsfield=np.zeros(gridX.shape)-1.0  # all start at NUTS=-1 level
        cultfracfield=np.zeros(gridX.shape)  # The area of this crop cultivated per gridbox
        arableareafield=np.zeros(gridX.shape)  # The area of this crop cultivated per gridbox

        gppfield=np.zeros((12,)+gridX.shape)
        terfield=np.zeros((12,)+gridX.shape)
        neefield=np.zeros((12,)+gridX.shape)
        ssrfield=np.zeros((12,)+gridX.shape)
        t2mfield=np.zeros((12,)+gridX.shape)
        tsumfield=np.zeros((12,)+gridX.shape)
        tagpfield=np.zeros((12,)+gridX.shape)
        twrtfield=np.zeros((12,)+gridX.shape)
        twsofield=np.zeros((12,)+gridX.shape)
        stressfield=np.zeros((12,)+gridX.shape)

        #
        # Now proceed to process each file of output
        #

        Tsfiles=[os.path.join(pandaresultsdir,f) for f in os.listdir(pandaresultsdir) if f.endswith('pickle')]

        mylogger.info("Processing %d gridpoints for crop %s"%(len(Tsfiles),crop))

        for filename in Tsfiles:

            # Get the NUTS_no and grid_no from the filename

            _ , NUTS_no, grid_no = os.path.basename(filename).split('_')
            if len(NUTS_no) == 2:
                NUTS_level=0
            elif len(NUTS_no) == 3:
                NUTS_level=1
            elif len(NUTS_no) == 4:
                NUTS_level=2
            else:
                mylogger.info("Weird NUTS code encountered (%s)"%NUTS_no)

            # Now open the file and see if we want to use its data, or whether a higher NUTS level was already processed

            data = cPickle.load(open(filename,'rb'))
            lon,lat = data['coords']
            grid_no = data['grid_no']
            lo,la,ii,jj, _, _ = gid[str(grid_no)]

            if not NUTS_level > nutsfield[jj,ii]: # WP If we already have better output, continue
                continue

            #
            # WP First order of business is to get the yieldgapfactor for this NUTS region
            #

            filelist = [ f for f in os.listdir(yieldgapresultsdir) if ('observed' in f or 'gapfilled' in f)]  #WP Selection for only observed, or only gap-filled NUTS
            filelist2 = [ f for f in filelist if f.split('_')[1] == NUTS_no]
            if not filelist2:
                mylogger.info("No optimized yield-gap factor file found for NUTS region %s in folder %s"%(NUTS_no, yieldgapresultsdir))
                print filelist2
                continue

            ygffile = os.path.join(yieldgapresultsdir,filelist2[0])
            optimi_info = cPickle.load(open(ygffile,'rb')) 
            fgap        = optimi_info[2]

            #
            #WP With the crop code, crop name, and NUTS name known we can find out the fraction of this crop cultivated in this country

            try:
                cultfrac = cropweights['%s'%NUTS_no]  # WP Get NUTS0 level code to estimate weights for each crop, these are 16 numbers
            except KeyError:

                mylogger.info("Non-existing NUTS code encountered (%s), cannot determine cultivated_fraction, using NaN instead"%NUTS_no)
                cultfrac = np.zeros((16,))+np.NaN  # WP insert NaN if unknown
                cropweights['%s'%NUTS_no] = cultfrac  # Add to crop weights for the next iteration

#WP Out of these 16 numbers, select the one with the number that corresponds to our crop number.
#WP Note that the crop order in the weights dictionary is stored under the key "crop_order", while our cropno was derived from the
#WP cropdict with names/numbers/decription of each crop in CGMS

            try:
                cropindex = cropweights['crop_order'].index(cropno) #WP Find index of this crop in the list of weights
            except ValueError,msg:
                mylogger.info("Crop number not found in list of weights for nee (%d), exiting..."%cropno)
                mylogger.info(msg)
                sys.exit(2)
            #
            # WP Now get the arable land area for this grid ID, and use it to calculate the cultivated area
            #

            NUTS_arable = CGMSgrid['nutstogrids_filled'][NUTS_no]  # all ids and areas of this NUTS region
            arable_area = [d for c,d in NUTS_arable if c == grid_no]  # for the current grid_id, get arable area
            cult_area = arable_area[0] * cultfrac[cropindex]  # total area cultivated by this crop in this grid

            # Okay, we can now start to fill all arrays with the data

            nutsfield[jj,ii] = NUTS_level # Only higher NUTS levels can replace
            masssumfield[jj,ii] = data['crop_mass']
            hifield[jj,ii] = data['crop_hi']
            yieldsumfield[jj,ii] = data['crop_yield']
            yieldgapfield[jj,ii] = fgap
            cultfracfield[jj,ii] = cultfrac[cropindex]
            arableareafield[jj,ii] = arable_area[0]

            gppfield[:,jj,ii] = (data['daily']['GPP']).resample('M').mean()[:]/sec_per_day
            terfield[:,jj,ii] = (data['daily']['TER']).resample('M').mean()[:]/sec_per_day
            neefield[:,jj,ii] = (data['daily']['NEE']).resample('M').mean()[:]/sec_per_day
            t2mfield[:,jj,ii] = (data['daily']['T2M']).resample('M').sum()[:]
            tsumfield[:,jj,ii] = (data['daily']['TSUM']).resample('M').sum()[:]
            ssrfield[:,jj,ii] = (data['daily']['SSR']).resample('M').sum()[:]
            tagpfield[:,jj,ii] = (data['daily']['TAGP']).resample('M').sum()[:]
            twrtfield[:,jj,ii] = (data['daily']['TWRT']).resample('M').sum()[:]
            twsofield[:,jj,ii] = (data['daily']['TWSO']).resample('M').sum()[:]
            stressfield[:,jj,ii] = (data['daily']['TRA']).resample('M').mean()[:]/(data['daily']['TRAMX']).resample('M').mean()[:]

        # When all files processes, we can write the gridded results to the NetCDF file variables

        cropyieldsum[:,:] = yieldsumfield
        cropmasssum[:,:] = masssumfield
        crophi[:,:] = hifield
        cropyieldgap[:,:] = yieldgapfield
        cropnuts[:,:] = nutsfield
        cropcultfrac[:,:] = cultfracfield

        cropgpp[:,:,:] = gppfield 
        cropter[:,:,:] = terfield 
        cropnee[:,:,:] = neefield 
        cropagp[:,:,:] = tagpfield 
        croptwso[:,:,:] = twsofield 
        cropstress[:,:,:] = stressfield 

    # And now, let's write the final accumulated fields to the NetCDF file, and close it

    for i in range(12):
        times[i] = (dt.datetime(years[0],i+1,1)-dt.datetime(2000,1,1)).days

    rootgrp.close()    

    mylogger.info('Successfully finished the script, returning...')
    sys.exit(0)


# END OF THE MAIN CODE


#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
