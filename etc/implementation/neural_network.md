# Neural Network

This file discusses the various implementation details of the neural network models/modules.

## General Module Structure

- The neural network module follows the structure of the pytorch neural network module class
- Since the module should be able to do both live inference and batch training, for live inference, the forward propagation stores the input of the module in self.input. This way, next time an inference occurs, it can pick the relevant input variables to feed to the model without having to recompute the output for the whole sequence of frames
- Modules should also have a reset function to set self.input => None

## Data Structure

### Input (Image)

> Batch x Channels x Width x Height
> Channels x Width x Height

### Input (Latent Vectors across time)

> Batch x Channels x Time
> Channels x Time

### Input (Images across time)

> Batch x Time x Channels x Width x Height
> Time x Channels x Width x Height

### Output (Modules)
 
In order to allow nn.Sequential to work for differing types of inference modes, the output of each module will be a tuple containing information of both the activation and the inference_mode

> tuple(output, inference_mode)

### Output (Model)

**Actor**

> tuple(Deterministic Action, Termination Flag)

**Critic**

> tuple(Q-value of a certain state-action pair)