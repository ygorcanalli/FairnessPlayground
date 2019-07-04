#!/bin/bash

nohup ssh -f $1.cos.ufrj.br -p 2222 "cd $2; ~/anaconda3/bin/jupyter-notebook --no-browser --port=$3" >> froward.log 1>> forward.log 2>> forward.log
nohup ssh -N -f -L localhost:$3:localhost:$3 rio.cos.ufrj.br -p 2222 >> forward.log 1>> forward.log 2>> forward.log
tail forward.log