# Reinforcement Learning Notes

## DDPG

### Training the actor
- Maximize Q(s_t, A(s_t))

### Training the critic
- Equate the bellman equation => Q(s_t, a_t) = R + γ * Q(s_(t+1), A(s_(t+1)))