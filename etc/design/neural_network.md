# Neural Network

This file discusses the various design choices for the neural network models.

## Architecture

The following points lists the computational componenets to be used

- 1. Small pre-trained CNN model converts raw visual images into latent vectors
	- resnet18
- 2. A mix of caused convolution and attention mechanism across the time domain is used to compute a latent vector for each time frame
- 3. A Fully-connected layer to finally compute the output of the network

### SNAIL Architecture

Small CNN Model
> "{kernel size 5×5, 16 filters, stride 2, ReLU nonlinearity}, whose output is then flattened and then passed to a fully-connected layer to produce a feature vector of size 256."

Temporal components
> "The total trajectory length was T = 500 (2 episodes of 250 timesteps each). For the policy, we used: TCBlock(T, 32), AttentionBlock(16, 16), TCBlock(T, 32), AttentionBlock(16, 16). For the value function we used: TCBlock(T, 16), TCBlock(T, 16)."

## Types of networks

The actor-critic reinforcement learning methodology will be utilized.

- **Actor**: The actor model inputs the observation and outputs the predicted actions.
- **Critic**: The critic model inputs the observation and action and outputs the value of the state-action pair.

## Modes of operation

### Live inference

Latent vectors of each frame is computed from the previous frame. Therefore the neural network has to perform inference on the current frame. This means that a convolution mechanism turns into a simple MLP mechanism.

- **Input** 
	- Pre-computed latent vectors of every layer and every previous frame
	- Current frame
- **Output**
	- Output of the current frame

### Batch training

Output of a whole batch is computed in parallel using convolution neural network module.

- **Input**
	- Batch of frames
- **Output**
- Output of every frame