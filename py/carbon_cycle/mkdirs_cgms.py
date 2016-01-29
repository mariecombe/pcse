#!/usr/bin/env python

import sys, os, shutil
import numpy as np

# This script will move any lost files from the main CGMS/ folder to the
# correct subfolders where they belong

#===============================================================================
def main():
#===============================================================================
# move files one by one

    # root directory
    origin = os.path.join(os.getcwd(),'CGMS_test/')

    first_level  = ["cropdata_objects/","soildata_objects/","timerdata_objects/",
                    "sitedata_objects/","gridlist_objects/"]
    second_level = np.arange(2000,2011,1)
    third_level  = [1,2,3,4,6,7,8,10,11,12,13] # crop ID numbers
    fourth_level = np.arange(2,20,1)
    print first_level
    print second_level
    print third_level, '\n'

    # create the first level of folders:
    for folder1 in first_level:
        if not os.path.exists(os.path.join(origin,folder1)):
            os.makedirs(os.path.join(origin,folder1))
            print 'successfully created %s'%(folder1)
        else:
            print '%s already exists!'%(folder1)
        # for soil and grid data, create the first level of partition only:
        if (folder1.startswith("soil") or folder1.startswith("grid")): continue

        # create the second level of folders:
        for year in second_level:
            folder2 = '%i/'%year
            if not os.path.exists(os.path.join(origin, folder1, folder2)):
                os.makedirs(os.path.join(origin, folder1, folder2))
                print 'successfully created %s'%(folder1+folder2)
            else:
                print '%s already exists!'%(folder1+folder2)

            # create the third level of folders:
            for crop_no in third_level:
                folder3 = 'c%i/'%crop_no
                if not os.path.exists(os.path.join(origin, folder1, folder2, folder3)):
                    os.makedirs(os.path.join(origin, folder1, folder2, folder3))
                    print 'successfully created %s'%(folder1+folder2+folder3)
                else:
                    print '%s already exists!'%(folder1+folder2+folder3)
                # for site date, we create the third level of partition:
                if (not folder1.startswith("site")): continue

                # create the fourth level of folders:
                for grid_no in fourth_level:
                    folder4 = 'grid_%i/'%grid_no
                    if not os.path.exists(os.path.join(origin, folder1, folder2, folder3, folder4)):
                        os.makedirs(os.path.join(origin, folder1, folder2, folder3, folder4))
                        print 'successfully created %s'%(folder1+folder2+folder3+folder4)
                    else:
                        print '%s already exists!'%(folder1+folder2+folder3+folder4)
        

#===============================================================================
if __name__=='__main__':
    main()
#===============================================================================
