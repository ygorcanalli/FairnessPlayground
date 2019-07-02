#!/bin/bash
# -------------------------------------------------------
#  find last created directory: format NNNN
# -------------------------------------------------------
originaldir=$PWD
pythonpath="~/anaconda3/bin/python"
basedir="results"
mkdir -p $basedir
cd $basedir
lastdir=$(ls -d [0-9][0-9][0-9][0-9] | tail -1)

# -------------------------------------------------------
#  create new sequential directory
# -------------------------------------------------------
next=$((++lastdir))
newdir=$(printf "%04u" $next)
mkdir $newdir
cd $newdir

destinationdir=$PWD

infofile="run.info"
logfile="nohup.out"

echo "Saving output to $destinationdir/$logfile"

echo "Command: ${0} ${@}" >> $infofile
echo "Directory: $originaldir" >> $infofile
echo "Date: $(date)" >> $infofile
echo "Current commit: $(git rev-parse HEAD)" >> $infofile

eval "nohup $pythonpath $originaldir/${@} >> $logfile 1>> $logfile 2>> $logfile &"