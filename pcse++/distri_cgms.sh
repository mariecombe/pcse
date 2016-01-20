#!/bin/bash
# This script will move any lost files from the main CGMS/ folder to the
# correct subfolders where they belong

# we first go to the directory containing the CGMS input data
#CGMSDIR="/Users/mariecombe/mnt/promise/CO2/marie/CGMS/"
CGMSDIR="/Users/mariecombe/Documents/Work/Research_project_3/model_input_data/CGMS"
cd ${CGMSDIR}

######################### SECOND PART OF THE SCRIPT ############################
# we move files contained in the main CGMS/ folder to the created sub-folders
################################################################################

# move up one step
cd ..

# we list the directories that require a year/crop_nb/ subfolder structure, and
# the list of years and crop numbers to create subfolders for:

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

nbdirs=3
targetdir[1]="sitedata_objects"
targetdir[2]="cropdata_objects"
targetdir[3]="timerdata_objects"
objectname[1]="siteobject"
objectname[2]="cropobject"
objectname[3]="timerobject"

SEARCHDIR="CGMS/"
COUNTER=1

# find the files, then move them to the correct subfolder
for k in $(jot $nbdirs 1); do
    dirdata=${targetdir[$k]}
    objdata=${objectname[$k]}

    for i in $(jot $nbyears 1); do
        diryear=${year[$i]}
 
        for j in $(jot $nbcrops 1); do
            dircrop=${crop[$j]}
 
            if [ $dirdata = "sitedata_objects" ]; then
                for g in $(jot 18 2); do
                    dirgrid="grid_$g"
                    TARGETDIR="CGMS/$dirdata/$diryear/$dircrop/$dirgrid/"
                    PATTERN="${objdata}_g${g}*_${dircrop}_y${diryear}*pickle"
                    echo "######## $COUNTER ########"
                    date
                    echo "Currently finding files with pattern $PATTERN in folder $SEARCHDIR"
                    echo "and moving them to $TARGETDIR"
                    echo "..."
                    find $SEARCHDIR -maxdepth 1 -name $PATTERN | xargs -I '{}' mv {} $TARGETDIR
                    COUNTER=$((COUNTER+1))
                done

            # if it is NOT site data, no partition based on grid cell number
            else
                TARGETDIR="CGMS/$dirdata/$diryear/$dircrop/"
                PATTERN="${objdata}*_${dircrop}_y${diryear}.pickle"
                echo "######## $COUNTER ########"
                date
                echo "Currently finding files with pattern $PATTERN in folder $SEARCHDIR"
                echo "and moving them to $TARGETDIR"
                echo "..."
                find $SEARCHDIR -maxdepth 1 -name $PATTERN | xargs -I '{}' mv {} $TARGETDIR
                COUNTER=$((COUNTER+1))
            fi
        done
    done
done

# grid list objects
dirdata="gridlist_objects"
TARGETDIR="CGMS/$dirdata/"
PATTERN="gridlistobject_all*.pickle"
echo "######## $COUNTER ########"
date
echo "Currently finding files with pattern $PATTERN in folder $SEARCHDIR"
echo "and moving them to $TARGETDIR"
echo "..."
find $SEARCHDIR -maxdepth 1 -name $PATTERN | xargs -I '{}' mv {} $TARGETDIR
COUNTER=$((COUNTER+1))

# soil data objects
dirdata="soildata_objects"
TARGETDIR="CGMS/$dirdata/"
PATTERN1="soilobject*.pickle"
PATTERN2="suitablesoilsobject*.pickle"
echo "######## $COUNTER ########"
date
echo "Currently finding files with pattern $PATTERN1 in folder $SEARCHDIR"
echo "and moving them to $TARGETDIR"
echo "..."
find $SEARCHDIR -maxdepth 1 -name $PATTERN1 | xargs -I '{}' mv {} $TARGETDIR
COUNTER=$((COUNTER+1))

echo "######## $COUNTER ########"
date
echo "Currently finding files with pattern $PATTERN2 in folder $SEARCHDIR"
echo "and moving them to $TARGETDIR"
echo "..."
find $SEARCHDIR -maxdepth 1 -name $PATTERN2 | xargs -I '{}' mv {} $TARGETDIR
COUNTER=$((COUNTER+1))

# crop masks
dirdata="cropdata_objects"
TARGETDIR="CGMS/$dirdata/"
PATTERN="cropmask*.pickle"
echo "######## $COUNTER ########"
date
echo "Currently finding files with pattern $PATTERN in folder $SEARCHDIR"
echo "and moving them to $TARGETDIR"
echo "..."
find $SEARCHDIR -maxdepth 1 -name $PATTERN | xargs -I '{}' mv {} $TARGETDIR
