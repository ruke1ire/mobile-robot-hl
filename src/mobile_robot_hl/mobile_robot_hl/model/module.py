import torch
import torch.nn as nn
import math

from .utils import *

class DenseBlock(nn.Module):
    def __init__(self, dilation: int, input_size: int, filter_size: int):
        super().__init__()
        self.dilatation = dilation
        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = input_size + filter_size
        self.input = None

        self.causal_conv1 = nn.Sequential(
            nn.ConstantPad1d((dilation, 0), 0),
            nn.Conv1d(input_size, filter_size, kernel_size=2,
                                   dilation=dilation)
        )
        self.causal_conv2 = nn.Sequential(
            nn.ConstantPad1d((dilation, 0), 0),
            nn.Conv1d(input_size, filter_size, kernel_size=2, dilation=dilation)
        )        
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, inference_mode = InferenceMode.WHOLE_BATCH):
        '''
        input, output = (time x filters)
        '''
        if(inference_mode == InferenceMode.WHOLE_BATCH):
            input = input.unsqueeze(2)
            xf, xg = self.causal_conv1(input), self.causal_conv2(input)
            activations = self.tanh(xf) * self.sigmoid(xg)
            return torch.cat([input, activations], dim=1).squeeze(2)
        else:
            raise NotImplementedError()
            if(self.input is not None):
                self.input = torch.cat((self.input, input), dim=0)
            else:
                self.input = input
            input_ = self.input[-max((self.dilation+1), self.input.shape[0]),:]
            xf, xg = self.causal_conv1(input_), self.causal_conv2(input_)
            activations = self.tanh(xf) * self.sigmoid(xg)
            return torch.cat([input, activations], dim = 1).squeeze(2)
    
    def reset(self):
        self.input = None

class TCBlock(nn.Module):
    def __init__(self, seq_len: int, input_size: int, filter_size: int):
        super().__init__()
        self.input_size = input_size
        self.no_layer = int(math.ceil(math.log2(seq_len)))
        modules = []

        for i in range(self.no_layer):
            modules.append(DenseBlock(2 ** i, (input_size + i * filter_size), filter_size))

        self.model = nn.Sequential(*modules)
        self.output_size = input_size + filter_size*(i+1)
        self.input = None

    def forward(self, input_, inference_mode = InferenceMode.WHOLE_BATCH):
        x = input_

        if(inference_mode == InferenceMode.WHOLE_BATCH):
            self.reset()
            x = self.model(x)
            return x
        else:
            raise NotImplementedError()
    
    def reset(self):
        self.input = None

class AttentionBlock(nn.Module):
    def __init__(self, input_size, key_size, value_size):
        super().__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = self.input_size + value_size
        self.input = None

        self.key_layer = nn.Linear(input_size, key_size)
        self.query_layer = nn.Linear(input_size, key_size)
        self.value_layer = nn.Linear(input_size, value_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, inference_mode = InferenceMode.WHOLE_BATCH):
        '''
        input = (time x input_size)
        output = (time x (input_size + value_size)
        '''
        if(inference_mode == InferenceMode.WHOLE_BATCH):
            seq_length = input.shape[0]
            keys = self.key_layer(input)  # bs x t x ks
            query = self.query_layer(input)  # bs x t x ks
            dot_product = query@keys.T
            mask = torch.ones(seq_length, seq_length, dtype=torch.bool) 
            for i in range(seq_length):
                mask[i, i:] = False
            dot_product[mask] = - float('inf')
            probs = self.softmax(dot_product / math.sqrt(self.key_size))
            values = self.value_layer(input)
            read = probs.matmul(values)
            output = torch.cat([input, read], dim=-1)
            return output
        else:
            raise NotImplementedError()
    
    def reset(self):
        self.input = None