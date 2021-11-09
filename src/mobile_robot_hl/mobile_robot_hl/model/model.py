import torch
import torch.nn as nn
import torchvision.models as models
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

        return output
    
    def reset(self):
        '''
        Reset all temporarily saved values
        '''
        for model in self.model:
            model.reset()
    
class MimeticSNAIL(nn.Module):
    def __init__(self, base_net_name, latent_vector_size, snail_kwargs, out_net):
        '''
        MimeticSNAIL Neural Network Model

        base_net_name: name of the pre-trained neural network found in https://pytorch.org/vision/stable/models.html
        latent_vector_size: size of the latent vector computed by the base_net
        snail_kwargs: keyword arguments to pass to the SNAIL model
        out_net: output neural network which may differ depending on whether it is a actor or a critic
        '''
        super().__init__()
        exec(f'self.base_net = models.{base_net_name}(pretrained=True)')
        # reset weights of last linear layer
        self.base_net.classifier = torch.nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), 
            nn.Linear(in_features=1280, out_features=latent_vector_size, bias=True))

        self.snail_net = Snail(**snail_kwargs)
        self.out_net = out_net
    
    def forward(self, input, inference_mode):
        shape_len = input.dim()

        if(shape_len in [3,4]):
            if(shape_len == 3):
                input = input.unsqueeze(0)
            latent_vec = self.base_net(input)
            latent_vec = latent_vec.permute((1,0))
            snail_out = self.snail_net(latent_vec, inference_mode)
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
                snail_out = self.snail_net(latent_vec, inference_mode).permute((1,0))
                output = self.out_net(snail_out)
                output_list.append(output)
            output = torch.stack(output_list)
        else:
            raise Exception("Invalid input shape")

        return output

    def reset(self):
        self.snail_net.reset()

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

    snail_kwargs = dict(input_size = 100, seq_length= 100, architecture=architecture)

    actor_output = nn.Sequential(
        nn.Linear(536, 3)
    )

    msnail = MimeticSNAIL(
        base_net_name='efficientnet_b0', 
        latent_vector_size=100, 
        snail_kwargs=snail_kwargs, 
        out_net = actor_output)

    print("Model:", msnail)

    batch_of_images = torch.ones((2, 10, 3, 320, 460))
    image_across_time = torch.ones((10, 3, 320, 460))
    single_image = torch.ones((3, 320, 460))

    print("Batch of images size:", batch_of_images.shape)
    print("Batch output size:", msnail(batch_of_images, inference_mode = InferenceMode.WHOLE_BATCH).shape)

    print("Image across time size:", image_across_time.shape)
    print("Image across time output size:", msnail(image_across_time, inference_mode = InferenceMode.WHOLE_BATCH).shape)

    print("Single image size:", single_image.shape)
    print("Single image output size:", msnail(single_image, inference_mode = InferenceMode.WHOLE_BATCH).shape)

    print("Single image size:", single_image.shape)
    print("Single image output size:", msnail(single_image, inference_mode = InferenceMode.ONLY_LAST_FRAME).shape)
    print(msnail.snail_net.model[0].model[2].input.shape)

    print("Single image size:", single_image.shape)
    print("Single image output size:", msnail(single_image, inference_mode = InferenceMode.ONLY_LAST_FRAME).shape)
    print(msnail.snail_net.model[0].model[2].input.shape)