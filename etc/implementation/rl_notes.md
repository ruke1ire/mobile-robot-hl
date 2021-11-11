# Reinforcement Learning Notes

## DDPG

### Training the actor
- Maximize Q(s_t, P(s_t))

#### Data needed from dataset
- Demonstration (s,a)
- Task Episode (s,a)

### Training the critic
- Equate the bellman equation => Q(s_t, a_t) = R + γ * Q(s_(t+1), P(s_(t+1)))

#### Data needed from dataset
- Demonstration (s,a)
- Task Episode (s,a)
- Task Rewards (R)

## PPO

### Training the actor
- Maximize the clipped objective function => L_clip = min(r_t * A(s_t, a_t) ​ , clip(r_t, 1−ε, 1+ε) * A(s_t, a_t))
	- r_t = P_new(a_t | s_t) / P_old(a_t | s_t)
	- ε = clip size

### Training the critic
- Equate the bellman equation => Q(s_t, a_t) = R + γ * Q(s_(t+1), A(s_(t+1)))