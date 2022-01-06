import torch
import torch.nn as nn
import torchvision.models as models
import math
import os

from .module import *
from .utils import *

MAX_LINEAR_VELOCITY = float(os.environ['MOBILE_ROBOT_HL_MAX_LINEAR_VEL'])
MAX_ANGULAR_VELOCITY = float(os.environ['MOBILE_ROBOT_HL_MAX_ANGULAR_VEL'])

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
            if ModuleType[module_information['module_type']] == ModuleType.TC:
                modules.append(TCBlock(seq_len=self.seq_length, input_size = size, **module_information['module_kwargs']))
            elif ModuleType[module_information['module_type']] == ModuleType.ATTENTION:
                modules.append(AttentionBlock(input_size = size, **module_information['module_kwargs']))
            size = modules[-1].output_size
        
        self.model = nn.Sequential(*modules)

    def forward(self, input, frame_no, inference_mode = InferenceMode.NONE):
        '''
        Forward propagation of the neural network. It also temporarily stores the computed values as self.output

        input: arrary of latent vectors of each time frame
        inference_mode = mode of inference
        '''
        shape_len = input.dim()
        assert (shape_len in [2,3]), "Invalid number of dimensions, should be in [2,3]"
        if(shape_len == 2):
            input = input.unsqueeze(0)

        if(inference_mode == InferenceMode.STORE):
            assert (input.shape[0] == 1), "Input batch size should == 1"

        output, frame_no, inference_mode = self.model((input, frame_no, inference_mode))

        if(shape_len == 2):
            output = output.squeeze(0)

        return output
    
    def reset(self):
        '''
        Reset all temporarily saved values
        '''
        for model in self.model:
            model.reset()
    
class MimeticSNAILActor(nn.Module):
    def __init__(self, base_net_architecture, snail_kwargs, out_net_architecture):
        '''
        MimeticSNAIL Actor Neural Network Model

        base_net_name: name of the pre-trained neural network found in https://pytorch.org/vision/stable/models.html
        latent_vector_size: size of the latent vector computed by the base_net
        snail_kwargs: keyword arguments to pass to the SNAIL model
        out_net_architecture: output neural network which may differ depending on whether it is a actor or a critic
        max_linear_velocity = maximum linear velocity
        max_angular_velocity = maximum angular velocity
        '''
        super().__init__()

        base_modules = []

        for module_information in base_net_architecture:
            exec(f"base_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.base_net = nn.Sequential(*base_modules)

        self.snail_net = Snail(**snail_kwargs)

        out_net_modules = []

        for module_information in out_net_architecture:
            exec(f"out_net_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        
        self.out_net = nn.Sequential(*out_net_modules)
        self.output_processor = OutputProcessor()
    
    def forward(self, input, input_latent=None, pre_output_latent=None, frame_no = None, noise = 0.0, inference_mode = InferenceMode.NONE):
        shape_len = input.dim()

        if(shape_len in [3,4]):
            if(shape_len == 3):
                input = input.unsqueeze(0)
            latent_vec = self.base_net(input)
            latent_vec = latent_vec.permute((1,0))
            if(input_latent is not None):
                if(input_latent.dim() == 1):
                    input_latent = input_latent.unsqueeze(1)
                latent_vec = torch.cat((latent_vec, input_latent), dim = 0)
            snail_out = self.snail_net(latent_vec, frame_no, inference_mode)
            if(pre_output_latent is not None):
                if(pre_output_latent.dim() == 1):
                    pre_output_latent = pre_output_latent.unsqueeze(1)
                snail_out = torch.cat((snail_out, pre_output_latent), dim = 0)
            if(shape_len == 3):
                snail_out = snail_out.squeeze(1)
            else:
                snail_out = snail_out.permute((1,0))

            output = self.out_net(snail_out)

        elif(shape_len == 5):
            output_list = []
            for input_ in input:
                latent_vec = self.base_net(input_)
                latent_vec = latent_vec.permute((1,0))
                if(input_latent is not None):
                    latent_vec = torch.cat((latent_vec, input_latent), dim = 0)
                snail_out = self.snail_net(latent_vec, frame_no, inference_mode)
                if(pre_output_latent is not None):
                    snail_out = torch.cat((snail_out, pre_output_latent), dim = 0)
                snail_out = snail_out.permute((1,0))
                output = self.out_net(snail_out)
                output_list.append(output)
            output = torch.stack(output_list)
        else:
            raise Exception("Invalid input shape")

        output = self.output_processor(output, noise)
        return output

    def reset(self):
        self.snail_net.reset()

class MimeticSNAILCritic(nn.Module):
    def __init__(self, base_net_architecture, snail_kwargs, agent_value):
        '''
        MimeticSNAIL Critic Neural Network Model

        base_net_name: name of the pre-trained neural network found in https://pytorch.org/vision/stable/models.html
        latent_vector_size: size of the latent vector computed by the base_net
        snail_kwargs: keyword arguments to pass to the SNAIL model
        agent_value: output network architecture for computing the agent value 
        user_value: output network architecture for computing the user value
        termination_flag_value: output network architecture for computing the termination_flag value
        '''
        super().__init__()

        base_modules = []

        for module_information in base_net_architecture:
            exec(f"base_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.base_net = nn.Sequential(*base_modules)

        self.snail_net = Snail(**snail_kwargs)

        agent_value_modules = []

        for module_information in agent_value:
            exec(f"agent_value_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.agent_value_net = nn.Sequential(*agent_value_modules)

    def forward(self, input, input_latent, pre_output_latent, frame_no, inference_mode = InferenceMode.NONE):
        shape_len = input.dim()

        if(shape_len in [3,4]):
            if(shape_len == 3):
                input = input.unsqueeze(0)
            latent_vec = self.base_net(input)
            latent_vec = latent_vec.permute((1,0))
            if(input_latent is not None):
                if(input_latent.dim() == 1):
                    input_latent = input_latent.unsqueeze(1)
                latent_vec = torch.cat((latent_vec, input_latent), dim = 0)

            snail_out = self.snail_net(latent_vec, frame_no, inference_mode)
            if(pre_output_latent.dim() == 1):
                pre_output_latent = pre_output_latent.unsqueeze(1)
            snail_out = torch.cat((snail_out, pre_output_latent[:-1]), dim = 0)

            if(shape_len == 3):
                snail_out = snail_out.squeeze(1)
            else:
                snail_out = snail_out.permute((1,0))

            output = self.agent_value_net(snail_out)
            output = output.squeeze(1)

        else:
            raise Exception("Invalid input shape")

        return output

    def reset(self):
        self.snail_net.reset()

class SSActor(nn.Module):
    def __init__(self, base_architecture, out_architecture):
        '''
        Single State Actor Module

        architecture: architecture of SSActor
        '''
        super().__init__()

        base_modules = []
        for module_information in base_architecture:
            exec(f"base_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.base_net = nn.Sequential(*base_modules)

        out_modules = []
        for module_information in out_architecture:
            exec(f"out_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.out_net = nn.Sequential(*out_modules)

        self.output_processor = OutputProcessor()
    
    def forward(self, input, input_latent=None, pre_output_latent=None, frame_no = None, noise = 0.0, inference_mode = InferenceMode.NONE):
        shape_len = input.dim()

        if(shape_len in [3,4]):
            if(shape_len == 3):
                input = input.unsqueeze(0)
            base_output = self.base_net(input)
            output = self.out_net(base_output)
        elif(shape_len == 5):
            output_list = []
            for input_ in input:
                base_output = self.base_net(input_)
                output = self.out_net(base_output)
                output_list.append(output)
            output = torch.stack(output_list)
        else:
            raise Exception("Invalid input shape")

        output = self.output_processor(output, noise)
         
        return output

    def reset(self):
        pass

class SSCritic(nn.Module):
    def __init__(self, base_architecture, out_architecture):
        '''
        Single State Actor Module

        architecture: architecture of SSActor
        '''
        super().__init__()

        base_modules = []
        for module_information in base_architecture:
            exec(f"base_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.base_net = nn.Sequential(*base_modules)

        out_modules = []
        for module_information in out_architecture:
            exec(f"out_modules.append(nn.{module_information['module_type']}(**{module_information['module_kwargs']}))")
        self.out_net = nn.Sequential(*out_modules)

        self.output_processor = OutputProcessor()
    
    def forward(self, input, input_latent=None, pre_output_latent=None, frame_no = None, noise = 0.0, inference_mode = InferenceMode.NONE):
        shape_len = input.dim()

        if(shape_len in [3,4]):
            if(shape_len == 3):
                input = input.unsqueeze(0)
            base_output = self.base_net(input)
            base_output = torch.cat((base_output, pre_output_latent.T), dim = 1)
            output = self.out_net(base_output)
            output = output.squeeze(1)
        elif(shape_len == 5):
            output_list = []
            for input_ in input:
                base_output = self.base_net(input_)
                base_output = torch.cat((base_output, pre_output_latent.T), dim = 1)
                output = self.out_net(base_output)
                output = output.squeeze(1)
                output_list.append(output)
            output = torch.stack(output_list)
        else:
            raise Exception("Invalid input shape")
         
        return output

    def reset(self):
        pass

class OutputProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY

    def forward(self, actor_output, noise = 0.0):
        device = actor_output.device
        multiplier = torch.tensor([self.max_linear_vel, self.max_angular_vel, 0.5], dtype = torch.float32).to(device)
        adder = torch.tensor([0.0, 0.0, 0.5], dtype = torch.float32).to(device)
        noise_tensor = ((torch.rand(actor_output.shape).to(device)*(2*multiplier) - multiplier)).to(device)
        actor_output = torch.tanh(actor_output)*multiplier
        actor_output = noise*noise_tensor + (1-noise)*actor_output
        actor_output += adder
        return actor_output