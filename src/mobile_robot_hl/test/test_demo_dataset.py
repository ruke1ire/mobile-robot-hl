import torch
from torch.utils.data import DataLoader
import os

from mobile_robot_hl.utils import *
from mobile_robot_hl.episode_data import *
from mobile_robot_hl.trainer.utils import *
from mobile_robot_hl.trainer.dataset import *

demo_path = os.environ['MOBILE_ROBOT_HL_DEMO_PATH']

dh = DemoHandler(demo_path)

dd = DemoDataset(dh)

idx = 1

print("image shape:", dd[idx][0].shape)

dl = DataLoader(dd, None, True)

print(next(iter(dl))[1].shape)
print(next(iter(dl))[1].shape)
print(next(iter(dl))[1].shape)
print(next(iter(dl))[1].shape)
