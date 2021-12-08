#!/usr/bin/python3

from mobile_robot_hl.trainer.utils import compute_rewards
import torch

demo_flag = torch.tensor([1,1,1,1,0,0,1,0])
user_term = torch.tensor([0,0,0,1,0,0,0,1])
agent_term = torch.tensor([1,1,1,1,1,1,1,1])
user_lin = torch.tensor([1,2,3,4,5,6,7,8])
agent_lin = torch.tensor([1,1,1,1,1,1,1,1])
user_ang = torch.tensor([1,2,3,4,5,6,7,8])
agent_ang = torch.tensor([1,1,1,1,1,1,1,1])

print("reward:",compute_rewards(demo_flag, user_term, agent_term, user_lin, agent_lin, user_ang, agent_ang))

