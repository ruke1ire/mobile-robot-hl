# Reinforcement Learning Notes

## DDPG

### Training the actor
- Maximize <img src="https://render.githubusercontent.com/render/math?math=\large Q(s_t, A(s_t))">

### Training the critic
- Equate bellman equation
	- <img src="https://render.githubusercontent.com/render/math?math=\large Q(s_t, a_t) = R + \gamma * Q(s_(t+1), A(s_(t+1)))">