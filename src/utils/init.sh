export MOBILE_ROBOT_HL_ROOT=$PWD/
export MOBILE_ROBOT_HL_DEMO_PATH=$PWD/data/demo
export MOBILE_ROBOT_HL_TASK_PATH=$PWD/data/task
export MOBILE_ROBOT_HL_MODEL_PATH=$PWD/data/model
export MOBILE_ROBOT_HL_RUN_SETUP_PATH=$PWD/data/run_setup
export MOBILE_ROBOT_HL_RUN_CHECKPOINT_PATH=$PWD/data/run_checkpoint

export MOBILE_ROBOT_HL_DESIRED_VELOCITY_TOPIC=/diffbot_base_controller/cmd_vel_unstamped
export MOBILE_ROBOT_HL_IMAGE_RAW_TOPIC=image_raw/uncompressed

export MOBILE_ROBOT_HL_MAX_LINEAR_VEL=0.1
export MOBILE_ROBOT_HL_MAX_ANGULAR_VEL=0.4

export MOBILE_ROBOT_HL_ACTOR_MODEL_TYPE=SSActor
export MOBILE_ROBOT_HL_CRITIC_MODEL_TYPE=SSCritic