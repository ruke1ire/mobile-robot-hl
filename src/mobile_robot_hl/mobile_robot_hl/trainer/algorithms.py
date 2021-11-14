import pickle
import torch

from mobile_robot_hl.model.model import *
from mobile_robot_hl.model.utils import InferenceMode
from mobile_robot_hl.trainer.utils import *

class TD3():
	def __init__(self, 
				actor_model,
				critic_model,
				actor_optimizer_dict, 
				critic_optimizer_dict, 
				dataloader, 
				discount = 0.99,
				tau = 0.005,
				policy_noise = 0.2,
				noise_clip = 0.5,
				actor_update_period = 2):

		self.actor_model = actor_model
		self.critic_model_1 = critic_model
		self.critic_model_2 = pickle.loads(pickle.dumps(self.critic_model_1))

		self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model))
		self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1))
		self.critic_model_2_target = pickle.loads(pickle.dumps(self.critic_model_1))

		self.dataloader = dataloader
		self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
		self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())
		self.critic_2_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_2.parameters())

		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.actor_update_period = actor_update_period
	
	def train_one_epoch(self):
		for i, (images, lin_vels, ang_vels, term_flags, demo_flags, rewards) in enumerate(self.dataloader):
			actions = torch.stack((lin_vels, ang_vels, term_flags, demo_flags))
			initial_action = torch.zeros_like(actions[:,0])
			initial_action[3] = 1.0
			prev_actions = torch.cat((initial_action.unsqueeze(1), actions[:,:-1]), dim = 1)

			with torch.no_grad():
				pass
				# 1. Create noises for target action
				# 2. Compute target actions from target actor P'(s(t+1))
				# 3. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))
				# 4. Use smaller Q-value as the Q-value target
				# 5. Compute current Q-value with the reward

			# 6. Compute Q-value from critics Q(s_t, a_t)
			# 7. Compute MSE loss for the critics
			# 8. Backpropagate and update critics

			# 9. Check whether to update the actor and the target policies
			#10. Compute the actor's action using the real actor
			#11. Compute the negative critic values using the real critic
			#12. Backpropagate and update actor
			#13. Update target networks

			if(i % self.actor_update_period == (self.actor_update_period - 1)):
				actor_actions = self.actor_model(input = images, input_latent = prev_actions, inference_mode = InferenceMode.WHOLE_BATCH)
				neg_critic_values = -self.critic_model_1(input = images, input_latent = prev_actions, pre_output_latent = actor_actions, inferenceMode = InferenceMode.WHOLE_BATCH).mean()

				self.actor_optimizer.zero_grad()
				neg_critic_values.backward()
				self.actor_optimizer.step()

				for param, target_param in zip(self.critic_model_1.parameters(), self.critic_model_1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	