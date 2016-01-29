#!/bin/bash
# This script will move any lost files from the main CGMS/ folder to the
# correct subfolders where they belong

# we first go to the directory containing the CGMS input data
REMDIR="/Users/mariecombe/mnt/promise/CO2/marie/CGMS"
LOCDIR="/Users/mariecombe/Documents/Work/Research_project_3/model_input_data/CGMS"
cd ${LOCDIR}

########################## THIRD PART OF THE SCRIPT ############################
# we sync the whole local tree structure to capegrim
################################################################################


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

OPTS="--delete"
SERVER=mariecombe@capegrim.wur.nl
KEY="ssh -l mariecombe -i /Users/mariecombe/.ssh/id_dsa"

COUNTER=1

# find the files, then move them to the correct subfolder
for k in $(jot $nbdirs 1); do
    dirdata=${targetdir[$k]}

    for i in $(jot $nbyears 1); do
        diryear=${year[$i]}
 
        for j in $(jot $nbcrops 1); do
            dircrop=${crop[$j]}
 
            if [ $dirdata = "sitedata_objects" ]; then
                for g in $(jot 18 2); do
                    dirgrid="grid_$g"
                    SRCDIR="$LOCDIR/$dirdata/$diryear/$dircrop/$dirgrid/"
                    TARGETDIR="$REMDIR/$dirdata/$diryear/$dircrop/$dirgrid/"
                    echo "######## $COUNTER ########"
                    date
                    echo "Currently syncing files from local folder $SRCDIR"
                    echo "to folder $TARGETDIR on $SERVER"
                    echo "..."
                    #rsync -auEv -e $KEY $OPTS $SRCDIR $SERVER:$TARGETDIR
                    COUNTER=$((COUNTER+1))
                done

            # if it is NOT site data, no partition based on grid cell number
            else
                SRCDIR="$LOCDIR/$dirdata/$diryear/$dircrop/"
                TARGETDIR="$REMDIR/$dirdata/$diryear/$dircrop/"
                echo "######## $COUNTER ########"
                date
                echo "Currently syncing files from local folder $SRCDIR"
                echo "to folder $TARGETDIR on $SERVER"
                echo "..."
                #rsync -auEv -e $KEY $OPTS $SRCDIR $SERVER:$TARGETDIR
                COUNTER=$((COUNTER+1))
            fi
        done
    done
done

# sync the last remaining folders that do not have a subfolder structure
lasttargetdirs[1]="gridlist_objects"
lasttargetdirs[2]="soildata_objects"
lasttargetdirs[3]="cropdata_objects"

for i in $(jot 3 1); do
    dirdata=${lasttargetdirs[$i]}
    SRCDIR="$LOCDIR/$dirdata/"
    TARGETDIR="$REMDIR/$dirdata/"
    echo "######## $COUNTER ########"
    date
    echo "Currently syncing files from local folder $SRCDIR"
    echo "to folder $TARGETDIR on $SERVER"
    echo "..."
    #rsync -auEv --no-r -e $KEY $OPTS $SRCDIR $SERVER:$TARGETDIR
    COUNTER=$((COUNTER+1))
done


