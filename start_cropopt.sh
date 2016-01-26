#!/bin/bash
set -e

EXPECTED_ARGS=2
E_BADARGS=666

if [ $# -ne $EXPECTED_ARGS ]
then
  echo ""
  echo "Usage: `basename $0` projectdir projectname"
  exit $E_BADARGS
fi

echo "New project to be started in folder $1"
echo "               ...........with name $2"

rootdir=$1/$2
rundir=$1/$2/exec
jobdir=$rundir/jobs
sedrundir=$1/$2/exec

if [ -d "$rootdir" ]; then
    echo "Directory already exists, please remove before running $0"
    exit 1
fi

mkdir -p ${rundir}
mkdir -p ${jobdir}
rsync -au --cvs-exclude * ${rundir}/
cd ${rundir}

echo "Creating jb file, py file, and rc-file"
sed -e "s/template/$2/g" template.jb > $2.jb
sed -e "s/template/$2/g" template.py > $2.py
sed -e "s,template,${rootdir},g" template.rc > $2.rc
rm -f template.py
rm -f template.jb
rm -f template.rc
rm -f start_cropopt.sh

chmod u+x $2.jb

echo ""
echo "************* NOW USE ****************"
ls -lrta $2.*
echo "**************************************"
echo ""
cd ${rundir}
pwd

