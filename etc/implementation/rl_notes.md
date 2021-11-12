# Reinforcement Learning Notes

## DDPG

### Training the actor
- Maximize Q(s_t, P(s_t))

#### Data needed from dataset
- Demonstration (s,a)
- Task Episode (s,a)
- Demonstration Flag (f)

### Training the critic
- Equate the bellman equation => Q(s_t, a_t) = R + γ * Q(s_(t+1), P(s_(t+1)))

#### Data needed from dataset
- Demonstration (s,a)
- Task Episode (s,a)
- Task Rewards (R)
- Demonstration Flag (f)

## PPO

### Training the actor
- Maximize the clipped objective function => L_clip = min(r_t * A(s_t, a_t) ​ , clip(r_t, 1−ε, 1+ε) * A(s_t, a_t))
	- r_t = P_new(a_t | s_t) / P_old(a_t | s_t)
	- ε = clip size

### Training the critic
- Equate the bellman equation => Q(s_t, a_t) = R + γ * Q(s_(t+1), A(s_(t+1)))

## Behavior cloning

- Equate P(s_t) = a_t

#### Data needed from dataset

- Demonstration (s,a)
- Task Episode (s,a)

## TD3

- Target network is a Q-network is used as the target for the real Q-network to converge to. But the target network lags behind the Q-network in terms of updating its weights.

- Same as DDPG but with additional tricks
	- 1. Clipped Double Q-Learning
		- Uses 2 Critic Networks and uses the one with the smaller output as the target for the bellman equation
			- Q(s_t, a_t) = R + γ * min_i(Q_i(s_(t+1), P'(s_(t+1))))
		- Usually there would be 2 actors to the 2 critic networks, but to reduce computation cost, we can update the 2 policies with the exact same target y = R + γ * min_i(Q_i(s_(t+1), P'(s_(t+1))))
	- 2. Delayed Policy Updates
		- The actor is updated less frequently than the critic
	- 3. Target Policy Smoothing
		- The action from the target actor is added with noise so that the Q-value estimation is regularized
			- P'(s) = clip(P'(s) + clip(ε,-c,+c), a_high, a_low)

### Networks
- P is used as the policy to be used on the real environment
- Q is trained to approximate the bellman equation
- P' is used to provide the target action when updating Q
- Q' is used to provide the target value when updating Q

#### Data and model needed to train actor
- Critic
- Demonstration (s,a)
- Task Episode (s,a)
- Demonstration Flag (f)

#### Data and model needed to train critics
- Target Actor
- Target Critics
- Demonstration (s,a)
- Task Episode (s,a)
- Task Rewards (R)
- Demonstration Flag (f)


1  1  1  1  1  0  0  0  1  1  1  1  0  0  
1  1  1  1 -4  0  0  0  1  1  1 -3  0  0  
