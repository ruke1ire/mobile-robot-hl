import torch
import torch.nn as nn
import math
import numpy as np

from mobile_robot_hl.model.utils import *

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

    def forward(self, input_tuple = (None, InferenceMode.NONE)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)
        
        if(inference_mode == InferenceMode.NONE):
            self.reset()
            input_ = self.initial_padding(input)
        else:
            assert (input.shape[0] == 1), "Input batch size should == 1"

            if(self.input is not None):
                self.input = torch.cat((self.input, input), dim=2)
            else:
                self.input = self.initial_padding(input)
            input_ = self.input[:,:,-(self.dilation+input.shape[2]):]

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

    def forward(self, input_tuple = (None, InferenceMode.NONE)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.NONE):
            self.reset()
        else:
            assert (input.shape[0] == 1), "Input batch size should == 1"

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
        self.keys = None
        self.values = None

        self.key_layer = nn.Linear(input_size, key_size)
        self.query_layer = nn.Linear(input_size, key_size)
        self.value_layer = nn.Linear(input_size, value_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_tuple = (None, InferenceMode.NONE)):
        input = input_tuple[0]
        inference_mode = input_tuple[1]
        assert input is not None, "Input is None"

        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.NONE):
            self.reset()
            seq_length = input.shape[2]
            input_ = input.transpose(1, 2) # Batch x Time x Channels
            values = self.value_layer(input_) # Batch x Time x ValueSize
            keys = self.key_layer(input_)  # Batch x Time x KeySize
            query = self.query_layer(input_)  # Batch x Time x QuerySize
        else:
            assert (input.shape[0] == 1), "Input batch size should == 1"

            input_ = input.transpose(1,2)
            value = self.value_layer(input_) # 1 x Time x ValueSize
            key = self.key_layer(input_)  # 1 x Time x KeySize
            query = self.query_layer(input_)  # 1 x Time x QuerySize

            if(self.values is not None):
                self.values = torch.cat((self.values, value), dim=1) # 1 x Time x ValueSize
                self.keys = torch.cat((self.keys, key), dim = 1) # 1 x Time x KeySize
            else:
                self.values = value
                self.keys = key

            seq_length = self.values.shape[1]

            values = self.values
            keys = self.keys

        dot_product = query@keys.transpose(1,2)
        scores = dot_product / math.sqrt(self.key_size)

        mask = subsequent_mask(seq_length, input.device)
        scores = scores.masked_fill(mask[:,-scores.shape[1]:,:] == 0, -float('inf'))

        probs = self.softmax(scores)
        activation = probs.matmul(values).transpose(1,2) # Batch x ValueSize x Time

        output = torch.cat((input, activation), dim=1)

        if(shape_len == 2):
            output = output.squeeze(0)

        return output, inference_mode
    
    def reset(self):
        self.values = None
        self.keys = None

def subsequent_mask(size, device):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    return subsequent_mask.to(device)
    
if __name__ == "__main__":
    dense_block = AttentionBlock(5, 30, 30)

    batch_input = torch.ones(10, 5, 100)
    print("Batch input shape = ", batch_input.shape)
    print("Batch output shape = ",dense_block((batch_input, InferenceMode.NONE))[0].shape)
    single_input = torch.ones(1, 5, 1)
    print("Single input shape = ", single_input.shape)
    output, inference_mode = dense_block((single_input, InferenceMode.NONE))
    print("Single output shape = ", output.shape)
    output, inference_mode = dense_block((single_input, InferenceMode.STORE))
    print(dense_block.values.shape)
    print("Single output shape = ",output.shape)
    output, inference_mode = dense_block((single_input, InferenceMode.STORE))
    print(dense_block.values.shape)
    print("Single output shape = ",output.shape)
    output, inference_mode = dense_block((single_input, InferenceMode.STORE))
    print(dense_block.values.shape)
    print("Single output shape = ",output.shape)
    multi_input = torch.ones(1, 5, 10)
    output, inference_mode = dense_block((multi_input, InferenceMode.STORE))
    print(dense_block.values.shape)
    print("Multi output shape = ",output.shape)

