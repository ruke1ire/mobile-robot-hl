import pickle
import torch
import torch.nn.functional as F
import time

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

class TD3(Algorithm):
	def __init__(self, 
				actor_model,
				critic_model,
				actor_optimizer_dict, 
				critic_optimizer_dict, 
				dataloader, 
				device1,
				device2,
				logger = None,
				discount = 0.99,
				tau = 0.005,
				policy_noise = 0.2,
				noise_clip = 0.5,
				actor_update_period = 2):
		'''
		TD3 algorithm implementation.

		Keyword arguments:
		actor_model -- the actor model to be trained *
		critic_model -- the critic model to be trained *
		actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
		critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
		dataloader -- the dataloader to be used to train the models *
		device -- the device to put the models in *
		logger -- logger to be used *
		discount -- discount factor used to compute the value of a state-action pair (float)
		tau -- the rate at which the target policies are updated (float)
		policy_noise -- vector of noise for each action dimension (list len(list) = N)
		noise_clip -- maximum values for the noise for each action dimension (list len(list) = N)
		'''

		self.device1 = device1
		self.device2 = device2

		self.actor_model = actor_model
		self.critic_model_1 = critic_model
		self.critic_model_2 = pickle.loads(pickle.dumps(self.critic_model_1))

		self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model)).eval()
		self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1)).eval()
		self.critic_model_2_target = pickle.loads(pickle.dumps(self.critic_model_1)).eval()

		self.actor_model.to(self.device1)
		self.critic_model_1.to(self.device1)
		self.critic_model_2.to(self.device1)
		self.actor_model_target.to(self.device1)
		self.critic_model_1_target.to(self.device1)
		self.critic_model_2_target.to(self.device1)

		self.dataloader = dataloader
		self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
		self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())
		self.critic_2_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_2.parameters())

		self.discount = discount
		self.tau = tau
		policy_noise = torch.tensor(policy_noise).to(device1)
		if(policy_noise.dim() == 2):
			self.policy_noise = policy_noise
		else:
			self.policy_noise = policy_noise.unsqueeze(1)
		noise_clip = torch.tensor(noise_clip).to(device1)
		if(noise_clip.dim() == 2):
			self.noise_clip = noise_clip
		else:
			self.noise_clip = noise_clip.unsqueeze(1)
		self.actor_update_period = actor_update_period
		self.logger = None
	
	def train_one_epoch(self, stop_flag):
		j = 0
		for (images, latent, frame_no, rewards) in self.dataloader:
			if(stop_flag == True):
				return

			print(f"Run No. {j+1}")
			print(f"Episode Length = {frame_no.shape[0]}")

			images = images.to(self.device1)
			latent = latent.to(self.device1)
			rewards = rewards.to(self.device1)

			actions = latent[:-1,:]
			demo_flag = latent[-1,:]
			initial_action = torch.zeros_like(latent[:,0])
			initial_action[3] = 1.0
			prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

			with torch.no_grad():
				# 1. Create noises for target action
				print("# 1. Create noises for target action")
				noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
				#time.sleep(5.0)
				# 2. Compute target actions from target actor P'(s(t+1))
				print("# 2. Compute target actions from target actor P'(s(t+1))")
				target_actions = self.actor_model_target(input = images, input_latent = prev_latent).permute((1,0)) + noise
				#time.sleep(5.0)
				# 3. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))
				print("# 3. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))")
				target_q1 = self.critic_model_1_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)
				target_q2 = self.critic_model_2_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)

				del target_actions
				#time.sleep(5.0)

				# 4. Use smaller Q-value as the Q-value target
				print("# 4. Use smaller Q-value as the Q-value target")
				target_q = torch.min(target_q1, target_q2)

				del target_q1, target_q2

				episode_values = compute_values(self.discount, rewards)
				target_q[rewards == 0] = episode_values[rewards == 0]

				del episode_values
				#time.sleep(5.0)
				# 5. Compute current Q-value with the reward
				print("# 5. Compute current Q-value with the reward")
				target_q_next = torch.cat((target_q[1:],torch.zeros(1).to(self.device1)), dim = 0)
				target_q = rewards + self.discount * target_q_next

				del target_q_next

			#time.sleep(5.0)
			# 6. Compute Q-value from critics Q(s_t, a_t)
			print("# 6.1 Compute Q-value from critics Q(s_t, a_t)")
			q1 = self.critic_model_1(input = images, input_latent = prev_latent, pre_output_latent = actions).squeeze(1)
			#time.sleep(5.0)
			# 7. Compute MSE loss for the critics
			print("# 7.1 Compute MSE loss for the critics")
			critic_loss = F.mse_loss(q1[demo_flag == 0.0], target_q[demo_flag == 0.0])
			self.critic_1_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_1_optimizer.step()

			q1.detach()
			critic_loss.detach()
			del q1, critic_loss
			torch.cuda.empty_cache() 

			print("# 6.2 Compute Q-value from critics Q(s_t, a_t)")
			q2 = self.critic_model_2(input = images, input_latent = prev_latent, pre_output_latent = actions).squeeze(1)

			print("# 7.2 Compute MSE loss for the critics")
			critic_loss = F.mse_loss(q2[demo_flag == 0.0], target_q[demo_flag == 0.0])
			critic_loss.backward()
			self.critic_2_optimizer.zero_grad()
			self.critic_2_optimizer.step()
			self.critic_2_optimizer.zero_grad()

			q2.detach()
			critic_loss.detach()
			del q2, critic_loss

			target_q.detach()
			del target_q
			torch.cuda.empty_cache() 
			
			#time.sleep(5.0)
			# 8. Optimize critic
			print("# 8. Optimize critic")
			self.critic_2_optimizer.zero_grad()
			self.critic_2_optimizer.step()

			#time.sleep(5.0)
			# 9. Check whether to update the actor and the target policies
			print("# 9. Check whether to update the actor and the target policies")
			if(j % self.actor_update_period == (self.actor_update_period - 1)):
				#time.sleep(5.0)
				#10. Compute the actor's action using the real actor
				print("#10. Compute the actor's action using the real actor")
				actor_actions = self.actor_model(input = images, input_latent = prev_latent).permute((1,0))
				#time.sleep(5.0)
				#11. Compute the negative critic values using the real critic
				print("#11. Compute the negative critic values using the real critic")
				dummy_critic = self.critic_model_1.to(self.device2)
				actor_loss = -dummy_critic(input = images.to(self.device2), input_latent = prev_latent.to(self.device2), pre_output_latent = actor_actions.to(self.device2)).mean()

				actor_actions.detach()
				del actor_actions

				#time.sleep(5.0)
				#12. Optimize actor
				print("#12. Optimize actor")
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()

				actor_loss.detach()
				del actor_loss, dummy_critic

				#time.sleep(5.0)
				#13. Update target networks
				print("#13. Update target networks")
				for param, target_param in zip(self.critic_model_1.to(self.device1).parameters(), self.critic_model_1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
				
				del param, target_param
				torch.cuda.empty_cache() 

			j += 1