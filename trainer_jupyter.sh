#!/bin/bash

source src/utils/init.sh
cd src/mobile_robot_hl
export PYTHONPATH=.
jupyter lab
