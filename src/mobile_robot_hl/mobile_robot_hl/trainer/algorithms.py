import pickle
import torch
import torch.nn.functional as F
import time

from mobile_robot_hl.utils import *
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
	def train_one_epoch(self, trainer):
		'''
		Train the algorithm for 1 epoch

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
				logger_dict,
				discount = 0.99,
				tau = 0.005,
				noise = 0.5,
				actor_update_period = 2):
		'''
		TD3 algorithm implementation.

		Keyword arguments:
		actor_model -- the actor model to be trained *
		critic_model -- the critic model to be trained *
		output_processor -- actor's output processor
		actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
		critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
		dataloader -- the dataloader to be used to train the models *
		device -- the device to put the models in *
		logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
		discount -- discount factor used to compute the value of a state-action pair (float)
		tau -- the rate at which the target policies are updated (float)
		noise -- noise value from 0.0 - 1.0
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
		self.noise = noise
		self.actor_update_period = actor_update_period

		self.logger = create_logger_from_dict(logger_dict)
	
	def train_one_epoch(self, trainer):
		j = 0
		for (images, latent, frame_no, rewards_velocity, rewards_termination_flag) in self.dataloader:
			if(trainer.stop == True):
				return

			print(f"Run No. {j+1}")
			print(f"Episode Length = {frame_no.shape[0]}")

			images = images.to(self.device1)
			latent = latent.to(self.device1)
			rewards_velocity = rewards_velocity.to(self.device1)
			rewards_termination_flag = rewards_termination_flag.to(self.device1)

			actions = latent[:-1,:]
			demo_flag = latent[-1,:]
			initial_action = torch.zeros_like(latent[:,0])
			initial_action[3] = 1.0
			prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

			with torch.no_grad():
				print("# 1. Compute target actions from target actor P'(s(t+1))")
				target_actions = self.actor_model_target(input = images, input_latent = prev_latent, noise = self.noise).permute((1,0)) 

				print("# 2. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))")
				target_q1 = self.critic_model_1_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)
				target_q2 = self.critic_model_2_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions).squeeze(1)

				del target_actions

				print("# 3. Use smaller Q-value as the Q-value target")
				target_q = torch.min(target_q1, target_q2)

				del target_q1, target_q2

				episode_values = compute_values(self.discount, rewards_velocity)
				target_q[demo_flag == 0,0] = 0

				del episode_values

				print("# 4. Compute current Q-value with the reward")
				target_q_next = torch.cat((target_q[1:,0],torch.zeros(1).to(self.device1)), dim = 0)
				target_q[:,0] = rewards_velocity + self.discount * target_q_next
				target_q[:,1] = rewards_termination_flag

				del target_q_next

			print("# 5.1 Compute Q-value from critics Q(s_t, a_t)")
			q1 = self.critic_model_1(input = images, input_latent = prev_latent, pre_output_latent = actions).squeeze(1)

			print("# 6.1 Compute MSE loss for the critics")
			critic_loss = F.mse_loss(q1[demo_flag == 0.0], target_q[demo_flag == 0.0])

			print("# 7.1 Optimize critic")
			self.critic_1_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_1_optimizer.step()
			self.logger.log(DataType.num, critic_loss.item(), key = "td3/loss/critic1")

			del q1, critic_loss
			torch.cuda.empty_cache() 

			print("# 5.2 Compute Q-value from critics Q(s_t, a_t)")
			q2 = self.critic_model_2(input = images, input_latent = prev_latent, pre_output_latent = actions).squeeze(1)

			print("# 6.2 Compute MSE loss for the critics")
			critic_loss = F.mse_loss(q2[demo_flag == 0.0], target_q[demo_flag == 0.0])

			print("# 7.2 Optimize critic")
			self.critic_2_optimizer.zero_grad()
			critic_loss.backward()
			self.critic_2_optimizer.step()
			self.logger.log(DataType.num, critic_loss.item(), key = "td3/loss/critic2")

			del q2, critic_loss
			del target_q
			torch.cuda.empty_cache() 

			print("# 8. Check whether to update the actor and the target policies")
			if(j % self.actor_update_period == (self.actor_update_period - 1)):
				print("# 9. Compute the actor's action using the real actor")
				actor_actions = self.actor_model(input = images, input_latent = prev_latent).permute((1,0))
				print("# 10. Compute the negative critic values using the real critic")
				dummy_critic = self.critic_model_1.to(self.device2)
				actor_loss = -dummy_critic(input = images.to(self.device2), input_latent = prev_latent.to(self.device2), pre_output_latent = actor_actions.to(self.device2)).mean()

				actor_actions.detach()
				del actor_actions

				print("# 11. Optimize actor")
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				self.actor_optimizer.step()
				self.logger.log(DataType.num, actor_loss.item(), key = "td3/loss/actor")

				del actor_loss, dummy_critic
				torch.cuda.empty_cache() 

				print("# 12. Update target networks")
				for param, target_param in zip(self.critic_model_1.to(self.device1).parameters(), self.critic_model_1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
				
				del param, target_param

			j += 1


class IL(Algorithm):
	def __init__(self,
				actor_model,
				actor_optimizer_dict,
				dataloader,
				device,
				logger_dict):
		'''
		IL (Behavior Cloning) Algorithm Implementation.

		actor_model -- nn.Module model 
		actor_optimizer_dict -- dictionary with information about optimizer
		dataloader -- torch.utils.data.Dataloader
		device -- device name in strings Eg. "cuda:1"
		logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
		'''

		self.device = device

		self.actor_model = actor_model.to(self.device)

		self.optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
		self.dataloader = dataloader

		self.logger = create_logger_from_dict(logger_dict)

	def train_one_epoch(self, trainer):
		j = 0
		for (images, latent, frame_no) in self.dataloader:
			if(trainer.stop == True):
				return

			print(f"Run No. {j+1}")
			print(f"Episode Length = {frame_no.shape[0]}")

			images = images.to(self.device)
			latent = latent.to(self.device)

			actions = latent[:-1,:]
			initial_action = torch.zeros_like(latent[:,0]).to(self.device)
			initial_action[3] = 1.0
			dup_images = torch.cat((images, images), dim = 0)
			dup_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:], latent[:,:-1]), dim = 1)

			print("# 1. Compute actor output")
			actor_output = self.actor_model(dup_images, dup_latent)

			target = actions.T
			output = actor_output[frame_no.shape[0]:]

			print("# 2. Compute loss")
			loss = F.mse_loss(target, output)

			print("# 3. Optimize actor model")
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			self.logger.log(data_type = DataType.num, data = loss.item(), key = "il/loss/actor")

			j += 1