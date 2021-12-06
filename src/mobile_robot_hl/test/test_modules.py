from mobile_robot_hl.model.module import *
from mobile_robot_hl.model import *

import torch

input_tensor = torch.rand(5, 4)

tc = TCBlock(2, 5, 2)

print("============TEST TC")

print(tc((input_tensor, InferenceMode.NONE))[0].T)
print(tc((input_tensor, InferenceMode.STORE))[0].T)
tc.reset()
print(tc((input_tensor[:,:2], InferenceMode.STORE))[0].T)
print(tc((input_tensor[:, 2:], InferenceMode.STORE))[0].T)

print("============TEST AB")

ab = AttentionBlock(5, key_size = 1, value_size = 1)

print(ab((input_tensor, InferenceMode.NONE))[0].T)
print(ab((input_tensor, InferenceMode.STORE))[0].T)
ab.reset()
print(ab((input_tensor[:,:2], InferenceMode.STORE))[0].T)
print(ab((input_tensor[:, 2:], InferenceMode.STORE))[0].T)



