import torch
from torch.utils.data import DataLoader
import os

from mobile_robot_hl.utils import *
from mobile_robot_hl.trainer.utils import *

demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']
task_path = os.environ['MOBILE_ROBOT_HL_TASK_PATH']

dh = DemoHandler(demo_path)
th = TaskHandler(task_path, dh)

td = TaskDataset(th, 0.9)

idx = 1

print("image shape:", td[idx][0].shape)
print("linear_vel shape:", td[idx][1].shape)
print("angular_vel shape:", td[idx][2].shape)
print("termination_flag shape:", td[idx][3].shape)
print("demonstration_flag shape:", td[idx][4].shape)
print("values shape:", td[idx][5].shape)

dl = DataLoader(td, None, True)

print(next(iter(dl))[5].shape)
print(next(iter(dl))[5].shape)
