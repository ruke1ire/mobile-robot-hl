import torch
import torch.nn as nn
import math
import numpy as np

from .utils import *

class DenseBlock(nn.Module):
    def __init__(self, dilation: int, input_size: int, filter_size: int):
        super().__init__()
        self.dilation = dilation
        self.input_size = input_size
        self.filter_size = filter_size
        self.output_size = input_size + filter_size
        self.input = None

        self.causal_conv1 = nn.Conv1d(input_size, filter_size, kernel_size=2,
                                    dilation=dilation)
        self.causal_conv2 = nn.Conv1d(input_size, filter_size, kernel_size=2, 
                                    dilation=dilation)
        self.initial_padding = nn.ConstantPad1d((dilation, 0), 0)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tuple = (None, InferenceMode.WHOLE_BATCH)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.WHOLE_BATCH):
            self.reset()
            input_ = self.initial_padding(input)
        else:
            assert ((input.shape[0] == 1) and (input.shape[2] == 1)), "Input batch size should == 1 and input time length should == 1 for inference_mode == ONLY_LAST_FRAME "

            if(self.input is not None):
                self.input = torch.cat((self.input, input), dim=2)
            else:
                self.input = self.initial_padding(input)
            input_ = self.input[:,:,-(self.dilation+1):]

        xf, xg = self.causal_conv1(input_), self.causal_conv2(input_)
        activations = self.tanh(xf) * self.sigmoid(xg)
        output = torch.cat((input, activations), dim = 1)

        if(shape_len == 2):
            output = output.squeeze(0)

        return output, inference_mode
    
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
        self.output_size = input_size + filter_size*self.no_layer

    def forward(self, input_tuple = (None, InferenceMode.WHOLE_BATCH)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.WHOLE_BATCH):
            self.reset()
        else:
            assert ((input.shape[0] == 1) and (input.shape[2] == 1)), "Input batch size should == 1 and input time length should == 1 for inference_mode == ONLY_LAST_FRAME"

        input_ = input

        output, inference_mode = self.model((input_, inference_mode))

        if(shape_len == 2):
            output = output.squeeze(0)

        return output, inference_mode
    
    def reset(self):
        for model in self.model:
            model.reset()

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

    def forward(self, input_tuple = (None, InferenceMode.WHOLE_BATCH)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.WHOLE_BATCH):
            self.reset()
            seq_length = input.shape[2]
            input_ = input.transpose(1, 2) # Batch x Time x Channels
        else:
            assert ((input.shape[0] == 1) and (input.shape[2] == 1)), "Input batch size should == 1 and input time length should == 1 for inference_mode == ONLY_LAST_FRAME "

            if(self.input is not None):
                self.input = torch.cat((self.input, input), dim=2)
            else:
                self.input = input
            seq_length = self.input.shape[2]

            input_ = self.input.transpose(1,2)

        values = self.value_layer(input_) # Batch x Time x ValueSize
        keys = self.key_layer(input_)  # Batch x Time x KeySize
        query = self.query_layer(input_)  # Batch x Time x QuerySize

        dot_product = query@keys.transpose(1,2)
        scores = dot_product / math.sqrt(self.key_size)

        mask = subsequent_mask(seq_length)
        scores = scores.masked_fill(mask == 0, -float('inf'))

        probs = self.softmax(scores)
        activation = probs.matmul(values).transpose(1,2)[:,:,-input.shape[2]:] # Batch x Time x ValueSize

        output = torch.cat((input, activation), dim=1)

        if(shape_len == 2):
            output = output.squeeze(0)

        return output, inference_mode
    
    def reset(self):
        self.input = None

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
    
if __name__ == "__main__":
    dense_block = TCBlock(10, 5, 30)

    batch_input = torch.ones(10, 5, 100)
    print("Batch input shape = ", batch_input.shape)
    print("Batch output shape = ",dense_block((batch_input, InferenceMode.WHOLE_BATCH))[0].shape)
    single_input = torch.ones(1, 5, 1)
    print("Single input shape = ", single_input.shape)
    dense_block((single_input, InferenceMode.ONLY_LAST_FRAME))
    print("Single output shape = ",dense_block((single_input, InferenceMode.ONLY_LAST_FRAME))[0].shape)

