#!/usr/bin/bash

source src/utils/init.sh
demo="rsync -avz $MOBILE_ROBOT_HL_ROOT/data/demo/ dlbox3:~/mobile-robot-hl/data/demo --delete"
task="rsync -avz $MOBILE_ROBOT_HL_ROOT/data/task/ dlbox3:~/mobile-robot-hl/data/task --delete"
#run_setup="rsync -avz $MOBILE_ROBOT_HL_ROOT/data/run_setup/ dlbox3:~/mobile-robot-hl/data/run_setup --delete"
#src="rsync -avz $MOBILE_ROBOT_HL_ROOT/src/ dlbox3:~/mobile-robot-hl/src"

if [ -z "$1" ]
  then
    eval $demo
    eval $task
    #eval $run_setup
    #eval $src
fi

while getopts "dth" o; do
    case "${o}" in
        d)
			eval $demo
            ;;
        t)
			eval $task
            ;;
        h)
                        echo "Example usage: ./send.sh -dt"
                        echo "options [demo = d, task = t]"
            ;;
    esac
done
