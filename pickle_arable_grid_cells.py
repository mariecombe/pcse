
from cPickle import dump as pickle_dump
from maries_toolbox import open_csv, get_list_CGMS_cells_in_Europe_arable
#-------------------------------------------------------------------------------
# folder on my local macbook where the CGMS_grid_list.csv file is located:
EUROSTATdir   = '/Users/mariecombe/Documents/Work/Research_project_3/'\
			   +'EUROSTAT_data'
#-------------------------------------------------------------------------------
# we read the list of CGMS grid cells from file

all_CGMS_grid_cells = open_csv(EUROSTATdir, ['CGMS_grid_list.csv'])
all_grids           = all_CGMS_grid_cells['CGMS_grid_list.csv']['GRID_NO']
lons                 = all_CGMS_grid_cells['CGMS_grid_list.csv']['LONGITUDE']
lats                 = all_CGMS_grid_cells['CGMS_grid_list.csv']['LATITUDE']

#-------------------------------------------------------------------------------
# From this list, we select the subset of grid cells located in Europe that
# contain arable land (no need to create weather data where there are no crops!)

europ_arable = get_list_CGMS_cells_in_Europe_arable(all_grids, lons, lats)
print europ_arable[0:10]

#-------------------------------------------------------------------------------
filename = 'europe_arable_CGMS_cellids.pickle'
#-------------------------------------------------------------------------------
pickle_dump(europ_arable,open(filename,'wb'))    
