#!/bin/bash
# this script will create a folder structure for the CGMS input data files to
# be stored in. Then it will move any lost files from the main CGMS/ folder to
# the correct subfolders where they belong

# we first go to the directory containing the CGMS input data
#CGMSDIR="/Users/mariecombe/mnt/promise/CO2/marie/CGMS/"
CGMSDIR="/Users/mariecombe/Documents/Work/Research_project_3/model_input_data/CGMS"
cd ${CGMSDIR}

########################## FIRST PART OF THE SCRIPT ############################
# we create our tree of folders to contain the CGMS input data
################################################################################

# we directly create the directories that don't require any subfolder structure
# (these are the ones that wil contain files that do not depend on year and
# crop type)
GRIDDIR="gridlist_objects/" # will contain 1 file per NUTS regions
SOILDIR="soildata_objects/" # will contain 1 file per grid cell (~15 000 files)
if [ -d "$GRIDDIR" ]; then 
    echo "directory $GRIDDIR already exists!"
else
    mkdir $GRIDDIR
    echo "directory $GRIDDIR has been created"
fi
if [ -d "$SOILDIR" ]; then 
    echo "directory $SOILDIR already exists!"
else
    mkdir $SOILDIR
    echo "directory $SOILDIR has been created"
fi


# we list the directories that require a year/crop_nb/ subfolder structure, and
# the list of years and crop numbers to create subfolders for:
nbdirs=3
targetdir[1]="sitedata_objects/"
targetdir[2]="cropdata_objects/"
targetdir[3]="timerdata_objects/"

nbyears=11
year[1]=2000
year[2]=2001
year[3]=2002
year[4]=2003
year[5]=2004
year[6]=2005
year[7]=2006
year[8]=2007
year[9]=2008
year[10]=2009
year[11]=2010

nbcrops=11
crop[1]="c1" # winter wheat
crop[2]="c2" # grain maize
crop[3]="c3" # spring barley
crop[4]="c4" # rye
crop[5]="c6" # sugar beet
crop[6]="c7" # potato
crop[7]="c8" # field beans
crop[8]="c10" # winter rapeseed
crop[9]="c11" # sunflower
crop[10]="c12" # fodder maize
crop[11]="c13" # winter barley

# loop over the directories that need year/cropnb/ subfolders
for k in $(jot $nbdirs 1); do
    dirdata=${targetdir[$k]}

    # create the data type directory only if it doesn't already exist
    if [ -d "$dirdata" ]; then 
        echo "directory $dirdata already exists!"
    else
        mkdir $dirdata
        echo "directory $dirdata has been created"
    fi

    # move inside the data type directory
    cd ${dirdata}
    pwd

    # loop over the years folder
    for i in $(jot $nbyears 1); do
        diryear=${year[$i]}

        # create the year directory only if it doesn't already exist
        if [ -d "$diryear" ]; then 
            echo "directory $diryear already exists!"
        else
            mkdir $diryear
            echo "directory $diryear has been created"
        fi

        # move inside the year directory
        cd ${diryear}
        pwd
 
        # loop over the crop folder
        for j in $(jot $nbcrops 1); do
            dircrop=${crop[$j]}
 
            # create the crop directory only if it doesn't already exist
            if [ -d "$dircrop" ]; then 
                echo "directory $dircrop already exists!"
            else
                mkdir $dircrop
                echo "directory $dircrop has been created"
            fi
           
            # for site data objects, we need one more level of partition
            # we will do it based on the grid cell number, in 9 groups:
            # group 1: grid cell nb starts with 1
            # group 2: grid cell nb starts with 2
            # ...
            # ...
            # group 9: grid cell nb starts with 9
            if [ $dirdata = "sitedata_objects/" ]; then

                # move inside the crop directory
                cd ${dircrop}
                pwd
 
                for g in $(jot 18 2); do
                    dirgrid="grid_$g"

                    # create the partition on grid cell nb
                    if [ -d "$dirgrid" ]; then 
                        echo "directory $dirgrid already exists!"
                    else
                        mkdir $dirgrid
                        echo "directory $dirgrid has been created"
                    fi
                done

                # move outside the crop directory
                cd ..
            fi
        done

        # move outside the year directory
        cd ..
    done

    # move outside the data type directory
    cd ..
done



######################### SECOND PART OF THE SCRIPT ############################
# we move files contained in the main CGMS/ folder to the created sub-folders
################################################################################

# see my separate script called distri_cgms.sh 


########################## THIRD PART OF THE SCRIPT ############################
# we sunc the whole tree structure on capegrim
################################################################################

# see my separate script called sync_cgms.sh
