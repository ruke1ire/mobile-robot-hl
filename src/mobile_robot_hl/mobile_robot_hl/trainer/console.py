from mobile_robot_hl.trainer.trainer import *
from mobile_robot_hl.model import *
from mobile_robot_hl.episode_data import *

import traceback
import os
import torch
import yaml
import time

demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
model_path = os.environ['MOBILE_ROBOT_HL_MODEL_PATH']
run_setup_path = os.environ['MOBILE_ROBOT_HL_RUN_SETUP_PATH']

dh = DemoHandler(demo_path)
th = TaskHandler(task_path, dh)
mh = ModelHandler(model_path)

dummy_observations = torch.zeros((40, 3, 30, 40))
dummy_input_latent = torch.zeros((4, 40))
dummy_actions = torch.zeros((3, 40))
frame_no = torch.tensor(list(range(1,21))+list(range(1,21)))

trainer = Trainer(mh, dh, th)

def execute(run_name, run_file):
	with open(f'{run_setup_path}/{run_name}/{run_file}.yaml', 'r') as stream:
		f = yaml.safe_load(stream)
	for key, value in f.items():
		key_split = key.split('__')
		command = key_split[0]
		exec(f"trainer.{command}(**value)")
	return f

#def main():
#
#	while True:
#		user_input = input("Trainer> ")
#		if(user_input == '-h'):
#			print("You can execute commands through yaml files with the command \n\texecute(run_name, run_file)\n")
#			continue
#		try:
#			exec(f"{user_input}")
#		except Exception:
#			print(traceback.format_exc())
#
#if __name__ == "__main__":
#	main()