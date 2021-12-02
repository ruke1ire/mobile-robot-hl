#!/bin/bash

#eval "$(conda shell.bash hook)"
#conda activate mobile_robot_hl

source src/utils/init.sh
cd src/mobile_robot_hl
export PYTHONPATH=.
python3 -i mobile_robot_hl/trainer/console.py
