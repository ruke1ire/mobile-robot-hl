# Neural Network

This file discusses the various implementation details of the neural network models/modules.

## General Module Structure

- The neural network module follows the structure of the pytorch neural network module class
- Since the module should be able to do both live inference and batch training, for live inference, the forward propagation stores the input of the module in self.input. This way, next time an inference occurs, it can pick the relevant input variables to feed to the model without having to recompute the output for the whole sequence of frames
- Modules should also have a reset function to set self.input => None