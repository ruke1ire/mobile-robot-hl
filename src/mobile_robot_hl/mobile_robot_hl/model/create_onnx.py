from .module import *
from .utils import *
from .model import *

import torch

architecture = [
        dict(
            module_type = ModuleType.TC.name,
            module_kwargs = dict(
                filter_size = 30
            )
        ),
        dict(
            module_type = ModuleType.ATTENTION.name,
            module_kwargs = dict(
                key_size = 16,
                value_size = 16
            )
        ),
        dict(
            module_type = ModuleType.TC.name,
            module_kwargs = dict(
                filter_size = 30
            )
        ),
    ]

snail_kwargs = dict(input_size = 100, seq_length= 100, architecture=architecture)

actor_architecture = [
    dict(
        module_type = "Linear",
        module_kwargs = dict(
            in_features = 536,
            out_features = 3
        )
    )
]

msnail = MimeticSNAIL(
    base_net_name='efficientnet_b0', 
    latent_vector_size=100, 
    snail_kwargs=snail_kwargs, 
    out_net_architecture = actor_architecture)

dummy_input = torch.ones((10, 3, 320, 460))
torch.onnx.export(msnail, dummy_input, "MimeticSNAIL.onnx", verbose=True)
