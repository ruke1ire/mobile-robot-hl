from mobile_robot_hl.model.module import *
from mobile_robot_hl.model import *

import torch

input_tensor = torch.rand(5, 6)
frame_no = torch.tensor([1,2,3,1,2,3])

tc = TCBlock(2, 5, 2)

print("============TEST TC")

print(tc((input_tensor, frame_no, InferenceMode.NONE))[0].T)
print(tc((input_tensor, frame_no, InferenceMode.STORE))[0].T)
tc.reset()
print(tc((input_tensor[:,:2], frame_no[:2], InferenceMode.STORE))[0].T)
print(tc((input_tensor[:, 2:5], frame_no[2:5], InferenceMode.STORE))[0].T)
print(tc((input_tensor[:, 5:], frame_no[5:], InferenceMode.STORE))[0].T)

print("============TEST AB")

ab = AttentionBlock(5, key_size = 1, value_size = 1)

print(ab((input_tensor, frame_no, InferenceMode.NONE))[0].T)
print(ab((input_tensor, frame_no, InferenceMode.STORE))[0].T)
ab.reset()
print(ab((input_tensor[:,:2], frame_no[:2], InferenceMode.STORE))[0].T)
print(ab((input_tensor[:,2:5], frame_no[2:5], InferenceMode.STORE))[0].T)
print(ab((input_tensor[:,5:], frame_no[5:], InferenceMode.STORE))[0].T)
