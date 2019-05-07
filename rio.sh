#!/bin/bash

nohup ssh -f rio.cos.ufrj.br -p 2222 "cd $1; ~/anaconda3/bin/jupyter-notebook --no-browser --port=$2"
nohup ssh -N -f -L localhost:$2:localhost:$2 rio.cos.ufrj.br -p 2222

exit 0