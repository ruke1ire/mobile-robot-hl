from mobile_robot_hl.model import *

import torch

dummy_observations = torch.randn((40, 3, 30, 40))
dummy_input_latent = torch.randn((4, 40))
dummy_actions = torch.randn((3, 40))
frame_no = torch.tensor(list(range(1,21))+list(range(1,21)))

actor_kwargs = dict(
        base_architecture = 
            [
                dict(module_type = 'Conv2d', 
                    module_kwargs = dict(
                        in_channels = 3,
                        out_channels = 16,
                        kernel_size = 4,
                        stride = 2,
                        padding = 0)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'BatchNorm2d',
                    module_kwargs = dict(num_features = 16)),
                dict(module_type = 'Dropout', 
                    module_kwargs = dict(p = 0.2)),
                dict(module_type = 'Conv2d', 
                    module_kwargs = dict(
                        in_channels = 16,
                        out_channels = 32,
                        kernel_size = 4,
                        stride = 2,
                        padding = 0)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'BatchNorm2d',
                    module_kwargs = dict(num_features = 32)),
                dict(module_type = 'Dropout', 
                    module_kwargs = dict(p = 0.2)),
                dict(module_type = 'Flatten',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 1536,
                        out_features = 500)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 500,
                        out_features = 100)),
            ],
        out_architecture = 
            [
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 104,
                        out_features = 50)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 50,
                        out_features = 3)),
            ]
        )

critic_kwargs = dict(
        base_architecture = 
            [
                dict(module_type = 'Conv2d', 
                    module_kwargs = dict(
                        in_channels = 3,
                        out_channels = 16,
                        kernel_size = 4,
                        stride = 2,
                        padding = 0)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'BatchNorm2d',
                    module_kwargs = dict(num_features = 16)),
                dict(module_type = 'Dropout', 
                    module_kwargs = dict(p = 0.2)),
                dict(module_type = 'Conv2d', 
                    module_kwargs = dict(
                        in_channels = 16,
                        out_channels = 32,
                        kernel_size = 4,
                        stride = 2,
                        padding = 0)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'BatchNorm2d',
                    module_kwargs = dict(num_features = 32)),
                dict(module_type = 'Dropout', 
                    module_kwargs = dict(p = 0.2)),
                dict(module_type = 'Flatten',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 1536,
                        out_features = 500)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 500,
                        out_features = 100)),
            ],
        out_architecture = 
            [
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 107,
                        out_features = 50)),
                dict(module_type = 'ReLU',
                    module_kwargs = dict()),
                dict(module_type = 'Linear',
                    module_kwargs = dict(
                        in_features = 50,
                        out_features = 1)),
            ]
        )
actor = SSActor(**actor_kwargs)
critic = SSCritic(**critic_kwargs)

print("============TEST ACTOR")

output_actor = actor(dummy_observations, dummy_input_latent, None, frame_no, noise = 0.3)
print("Output =", output_actor)
print("Size =", output_actor.shape)

print("============TEST CRITIC")

output_critic = critic(dummy_observations, dummy_input_latent, dummy_actions, frame_no)
print("Output =", output_critic)
print("Size =", output_critic.shape)
