import pickle
import torch
import torch.nn.functional as F

from mobile_robot_hl.trainer.utils import *

from abc import ABC, abstractmethod

class Algorithm(ABC):
	@abstractmethod
	def __init__(self):
		'''
		Initializes the components needed for the algorithm. 
		Every component and variable should be set here and should not be changed later on.
		If a certain variable/component is to be changed, the algorithm object should be re-initilized
		'''
		pass

	@abstractmethod
	def train_one_epoch(self, stop_flag):
		'''
		Train the algorithm for 1 epoch

		:param stop_flag: A flag which indicates to stop the training if raised
		'''
		pass

	@abstractmethod
	def select_device(self, device_name):
		pass

class TD3(Algorithm):
	def __init__(self, 
				actor_model,
				critic_model,
				actor_optimizer_dict, 
				critic_optimizer_dict, 
				dataloader, 
				device,
				logger = None,
				discount = 0.99,
				tau = 0.005,
				policy_noise = 0.2,
				noise_clip = 0.5,
				actor_update_period = 2):
		'''
		TD3 algorithm implementation.

		Keyword arguments:
		actor_model -- the actor model to be trained
		critic_model -- the critic model to be trained
		actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model
		critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model
		dataloader -- the dataloader to be used to train the models
		device -- the device to put the models in
		logger -- logger to be used
		discount -- discount factor used to compute the value of a state-action pair (float)
		tau -- the rate at which the target policies are updated (float)
		policy_noise -- vector of noise for each action dimension (torch.tensor Size[N])
		noise_clip -- maximum values for the noise for each action dimension (torch.tensor Size[N])
		'''

		self.device = device

		self.actor_model = actor_model
		self.critic_model_1 = critic_model
		self.critic_model_2 = pickle.loads(pickle.dumps(self.critic_model_1))

		self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model))
		self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1))
		self.critic_model_2_target = pickle.loads(pickle.dumps(self.critic_model_1))

		self.actor_model.to(self.device)
		self.critic_model_1.to(self.device)
		self.critic_model_2.to(self.device)
		self.actor_model_target.to(self.device)
		self.critic_model_1_target.to(self.device)
		self.critic_model_2_target.to(self.device)

		self.dataloader = dataloader
		self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
		self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())
		self.critic_2_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_2.parameters())

		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		if(noise_clip.dim() == 2):
			self.noise_clip = noise_clip
		else:
			self.noise_clip = noise_clip.unsqueeze(1)
		self.actor_update_period = actor_update_period
		self.logger = None
	
	def train_one_epoch(self, stop_flag):
		for i, (images, lin_vels, ang_vels, term_flags, demo_flags, rewards) in enumerate(self.dataloader):

			if(stop_flag == True):
				return

			images = images.to(self.device)
			lin_vels = lin_vels.to(self.device)
			ang_vels = ang_vels.to(self.device)
			term_vels = term_vels.to(self.device)
			demo_flags = demo_flags.to(self.device)
			rewards = rewards.to(self.device)

			latent = torch.stack((lin_vels, ang_vels, term_flags, demo_flags))
			initial_action = torch.zeros_like(actions[:,0])
			initial_action[3] = 1.0
			prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)
			actions = latent[:-1,:]

			with torch.no_grad():
				# 1. Create noises for target action
				noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
				# 2. Compute target actions from target actor P'(s(t+1))
				target_actions = self.actor_model_target(input = images, input_latent = prev_latent).permute((1,0))
				# 3. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))
				target_q1 = self.critic_model_1_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)
				target_q2 = self.critic_model_2_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)
				# 4. Use smaller Q-value as the Q-value target
				target_q = torch.min(target_q1, target_q2)
				episode_values = compute_values(self.discout, rewards)
				target_q[rewards == 0] = episode_values[rewards == 0]
				# 5. Compute current Q-value with the reward
				target_q_next = torch.cat((target_q[1:],torch.zeros(1)), dim = 0)
				target_q = rewards + self.discount * target_q_next

			# 6. Compute Q-value from critics Q(s_t, a_t)
			q1 = self.critic_model_1(input = images, input_latent = prev_latent, pre_output_latent = actions)
			q2 = self.critic_model_2(input = images, input_latent = prev_latent, pre_output_latent = actions)
			# 7. Compute MSE loss for the critics
			critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
			
			# 8. Optimize critic
			self.critic_1_optimizer.zero_grad()
			self.critic_2_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_1_optimizer.step()
			self.critic_2_optimizer.step()

			# 9. Check whether to update the actor and the target policies
			if(i % self.actor_update_period == (self.actor_update_period - 1)):
				#10. Compute the actor's action using the real actor
				actor_actions = self.actor_model(input = images, input_latent = prev_latent)
				#11. Compute the negative critic values using the real critic
				actor_loss = -self.critic_model_1(input = images, input_latent = prev_latent, pre_output_latent = actor_actions).mean()
				#12. Optimize actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				#13. Update target networks
				for param, target_param in zip(self.critic_model_1.parameters(), self.critic_model_1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
	
	def select_device(self, device_name):
		self.device = device_name
		self.actor_model.to(self.device)
		self.critic_model_1.to(self.device)
		self.critic_model_2.to(self.device)
		self.actor_model_target.to(self.device)
		self.critic_model_1_target.to(self.device)
		self.critic_model_2_target.to(self.device)