#!/usr/bin/bash

model="rsync -avz dlbox3:~/mobile-robot-hl/data/model/ $MOBILE_ROBOT_HL_ROOT/data/model"

if [ -z "$1" ]
  then
    eval $model
fi

while getopts "mh" o; do
    case "${o}" in
        m)
			eval $model
            ;;
		h)
			echo "Example usage: ./recv.sh -m"
			echo "options [model = m]"
			;;
    esac
done