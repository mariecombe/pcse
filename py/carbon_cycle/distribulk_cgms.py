#!/usr/bin/env python

import sys, os, shutil
import numpy as np

# This script will move any lost files from the main CGMS/ folder to the
# correct subfolders where they belong

#===============================================================================
def main():
#===============================================================================
# open the list of files
    f = open('test.txt', 'r')
    list_of_files = f.readlines()

#-------------------------------------------------------------------------------
# move files one by one

    # define root directory
    origin = 'CGMS/'

    for filename in list_of_files:
        # remove the return sign from the filename:
        filename = filename.split('\n')[0]

        # we skip any files that are not pickle files
        if not filename.endswith('.pickle'): continue

        # initialize the path name
        subfolder = ''

        # get the first level of partition the file should be moved to
        args = filename.split('_')
        if ((args[0]=='cropmask') or (args[0]=='cropobject')):
            subfolder += 'cropdata_objects/'
        elif args[0].startswith('soilobject') or args[0].startswith('suitablesoils'):
            subfolder += 'soildata_objects/'
        elif args[0].startswith('timerobject'):
            subfolder += 'timerdata_objects/'
        elif args[0].startswith('siteobject'):
            subfolder += 'sitedata_objects/'
        elif args[0].startswith('gridlistobject_all'):
            subfolder += 'gridlist_objects/'
        else:
            continue

        # get the second level of partition for cropobjects, timerobjects, 
        # siteobjects:
        if len(args)>2:
            # concatenate the year to the path name:
            subfolder += args[3][1:5] + '/'
            # concatenate the crop number to the path name:
            subfolder += args[2] + '/'

			# get the third level of partition for siteobjects only:
            if len(args)>4:
                grid_no = args[1]
                if grid_no.startswith('g1'):
                    subfolder += 'grid_' + grid_no[1:3] + '/'
                else: 
                    subfolder += 'grid_' + grid_no[1] + '/'

        sourcedir   = os.path.join(origin, filename)
        destination = os.path.join(origin, subfolder, filename)
        try:
            shutil.move(sourcedir,destination) 
            print 'successfully moved %s to %s'%(filename, subfolder)
        except IOError:
            print 'skipping %s'%filename

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
