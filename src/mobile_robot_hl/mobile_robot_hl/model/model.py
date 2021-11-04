import torch
import torch.nn as nn
import math

from .module import *
from .utils import *

class Snail(nn.Module):
    def __init__(self, input_size, seq_length, architecture: list):
        '''
        SNAIL neural network model

        input_size: the flattened size of each vector in a frame
        seq_length: receptive field length of the convolution mechanism in the SNAIL model
        architecture: list(dict(module_type, module_kwargs))
        '''
        super().__init__()

        self.input_size = input_size
        self.seq_length = seq_length

        modules = []

        size = input_size
        for module_information in architecture:
            if module_information['module_type'] == ModuleType.TC:
                modules.append(TCBlock(seq_len=self.seq_length, input_size = size, **module_information['module_kwargs']))
            elif module_information['module_type'] == ModuleType.ATTENTION:
                modules.append(AttentionBlock(input_size = size, **module_information['module_kwargs']))
            size = modules[-1].output_size
        
        self.model = nn.Sequential(*modules)

    def forward(self, input_, inference_mode = InferenceMode.WHOLE_BATCH):
        '''
        Forward propagation of the neural network. It also temporarily stores the computed values as self.output

        input: ?
        inference_mode = mode of inference
        '''
        x = input_
        if(inference_mode == InferenceMode.WHOLE_BATCH):
            x = self.model(x)
            return x
        else:
            x = self.model(x)
            raise NotImplementedError()
    
    def reset(self):
        '''
        Reset all temporarily saved values
        '''
        raise NotImplementedError()

if __name__ == "__main__":
    architecture = [
            dict(
                module_type = ModuleType.TC,
                module_kwargs = dict(
                    filter_size = 30
                )
            ),
            dict(
                module_type = ModuleType.ATTENTION,
                module_kwargs = dict(
                    key_size = 30,
                    value_size = 30
                )
            ),
        ]
    s = Snail(input_size = 10, seq_length= 10, architecture=architecture)
    print(s)
    input_vec = torch.ones(100,10)
    print(input_vec.shape)
    print(s(input_vec).shape)