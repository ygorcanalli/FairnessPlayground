#!/bin/bash
# -------------------------------------------------------
#  find last created directory: format NNNN
# -------------------------------------------------------
basedir=results
cd $basedir
lastdir=$(ls -d [0-9][0-9][0-9][0-9] | tail -1)
cd ..
# -------------------------------------------------------
#  create new sequential directory
# -------------------------------------------------------
next=$((++lastdir))
newdir=$(printf "%04u" $next)
mkdir $basedir/$newdir

infofile=$basedir/$newdir/run.info
logfile=$basedir/$newdir/nohup.out

echo "Saving output to $logfile"

echo "Command: ${0} ${@}" >> $infofile
echo "Date: $(date)" >> $infofile
echo "Current commit: $(git rev-parse HEAD)" >> $infofile

eval "nohup ${@} >> $logfile 1>> $logfile 2>> $logfile &"