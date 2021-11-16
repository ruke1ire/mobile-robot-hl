from .trainer import *
from mobile_robot_hl.utils import *
from mobile_robot_hl.model.model_handler import ModelHandler

import traceback
import os
import torch
import yaml

demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']
model_path = os.environ['MOBILE_ROBOT_HL_MODEL_PATH']
run_setup_path = os.environ['MOBILE_ROBOT_HL_RUN_SETUP_PATH']

def load_runs(run_name):
	with open(f'{run_setup_path}/{run_name}.yaml', 'r') as stream:
		f = yaml.safe_load(stream)
	return f

dh = DemoHandler(demo_path)
th = TaskHandler(task_path, dh)
mh = ModelHandler(model_path)

dummy_observations = torch.zeros((100, 3, 240, 320))
dummy_input_latent = torch.zeros((4, 100))
dummy_actions = torch.zeros((3, 100))

trainer = Trainer(mh, dh, th, None)

while True:
	user_input = input("Trainer> ")
	try:
		exec(f"{user_input}")
	except Exception:
		print(traceback.format_exc())
