import torch
import torch.nn as nn
import math

from .module import *
from .utils import *

class Snail(nn.Module):
    def __init__(self, input_size, seq_length, architecture: list):
        '''
        SNAIL neural network model

        input_size: size of latent vector
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

    def forward(self, input, inference_mode = InferenceMode.WHOLE_BATCH):
        '''
        Forward propagation of the neural network. It also temporarily stores the computed values as self.output

        input: arrary of latent vectors of each time frame
        inference_mode = mode of inference
        '''
        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.ONLY_LAST_FRAME):
            assert ((input.shape[0] == 1) and (input.shape[2] == 1)), "Input batch size should == 1 and input time length should == 1 for inference_mode == ONLY_LAST_FRAME"

        output, inference_mode = self.model((input, inference_mode))

        if(shape_len == 2):
            output = output.squeeze(0)

        return output, inference_mode
    
    def reset(self):
        '''
        Reset all temporarily saved values
        '''
        for model in self.model:
            model.reset()

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
                    key_size = 16,
                    value_size = 16
                )
            ),
            dict(
                module_type = ModuleType.TC,
                module_kwargs = dict(
                    filter_size = 30
                )
            ),
        ]

    s = Snail(input_size = 100, seq_length= 100, architecture=architecture)
    print(s)
    batch_input = torch.ones(100,500)
    print("Batch input size:", batch_input.shape)
    print("Batch output size:", s(batch_input)[0].shape)

    single_input = torch.ones(100, 1)
    print("Single input size:", single_input.shape)
    print("Single output size:", s(single_input)[0].shape)






