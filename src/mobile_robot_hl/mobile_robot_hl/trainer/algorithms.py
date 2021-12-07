import pickle
import torch
import torch.nn.functional as F
import time
import os

from mobile_robot_hl.utils import *
from mobile_robot_hl.trainer.utils import *

from abc import ABC, abstractmethod

CHECKPOINT_PATH = os.environ['MOBILE_ROBOT_HL_RUN_CHECKPOINT_PATH']

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
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                critic_model,
                actor_optimizer_dict, 
                critic_optimizer_dict, 
                dataloader, 
                device1,
                device2,
                logger_name = None,
                discount = 0.99,
                tau = 0.005,
                noise = 0.5,
                actor_update_period = 2,
                ):
        '''
        TD3 algorithm implementation.

        Keyword arguments:
        actor_model -- the actor model to be trained *
        critic_model -- the critic model to be trained *
        actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        dataloader -- the dataloader to be used to train the models *
        device -- the device to put the models in *
        logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
        discount -- discount factor used to compute the value of a state-action pair (float)
        tau -- the rate at which the target policies are updated (float)
        noise -- noise value from 0.0 - 1.0
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device1 = device1
        self.device2 = device2

        self.actor_model = actor_model
        self.critic_model_1 = critic_model
        self.critic_model_2 = pickle.loads(pickle.dumps(self.critic_model_1))

        self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model)).eval()
        self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1)).eval()
        self.critic_model_2_target = pickle.loads(pickle.dumps(self.critic_model_1)).eval()

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_model_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_model_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_model_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.critic_model_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
            self.critic_model_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)

        self.actor_model.to(self.device1)
        self.critic_model_1.to(self.device1)
        self.critic_model_2.to(self.device1)
        self.actor_model_target.to(self.device1)
        self.critic_model_1_target.to(self.device1)
        self.critic_model_2_target.to(self.device1)

        self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
        self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())
        self.critic_2_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_2.parameters())

        if(checkpoint is not None):
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
            self.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])

        self.dataloader = dataloader

        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.actor_update_period = actor_update_period

        self.checkpoint_every = checkpoint_every
        config_dict = dict(
            actor_optimizer = actor_optimizer_dict,
            critic_optimizer = critic_optimizer_dict,
            discount = self.discount, 
            tau = self.tau, 
            noise = self.noise, 
            actor_update_period = self.actor_update_period
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)
    
    def train_one_epoch(self, trainer):
        j = 0
        for (images, latent, frame_no, rewards_agent, rewards_user, rewards_termination_flag) in self.dataloader:
            if(trainer.stop == True):
                return

            print(f"Run No. {j+1}")
            print(f"Episode Length = {frame_no.shape[0]}")

            task_start_index = (frame_no == 1).nonzero()[1].item()

            images = images.to(self.device1)
            latent = latent.to(self.device1)
            rewards_agent = rewards_agent.to(self.device1)
            rewards_user = rewards_user.to(self.device1)
            rewards_termination_flag = rewards_termination_flag.to(self.device1)

            actions = latent[:-1,:]
            demo_flag = latent[-1,:]
            initial_action = torch.zeros_like(latent[:,0])
            initial_action[3] = 1.0
            prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

            with torch.no_grad():
                print("# 1. Compute target actions from target actor P'(s(t+1))")
                target_actions = self.actor_model_target(input = images, input_latent = prev_latent, noise = self.noise).permute((1,0)) 
                print("target_actions", target_actions)

                print("# 2. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))")
                target_q1 = self.critic_model_1_target(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = target_actions)
                target_q2 = self.critic_model_2_target(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = target_actions)

                del target_actions

                print("# 3. Use smaller Q-value as the Q-value target")
                target_q = torch.min(target_q1, target_q2)
                self.logger.log(DataType.num, target_q.mean().item(), "td3/avg-value")

                del target_q1, target_q2

                target_q[demo_flag == 1,0] = 0

                print("action",actions)
                print("rewards user", rewards_user)
                print("rewards agent", rewards_agent)

                print("# 4. Compute current Q-value with the reward")
                target_q_next = torch.cat((target_q[1:,0],torch.zeros(1).to(self.device1)), dim = 0)
                target_q[:,0] = rewards_agent + self.discount * target_q_next
                target_q[:,1] = rewards_user
                target_q[:,2] = rewards_termination_flag
                target_q = target_q[task_start_index:]
                print("target_q", target_q)

                del target_q_next

            print("# 5.1 Compute Q-value from critics Q(s_t, a_t)")
            q1 = self.critic_model_1(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = actions)
            q1 = q1[task_start_index:]

            print("# 6.1 Compute MSE loss for the critics")
            critic_loss_agent = F.mse_loss(q1[demo_flag[task_start_index:] == 0.0,0], target_q[demo_flag[task_start_index:] == 0.0,0])
            critic_loss_user = 10*F.mse_loss(q1[demo_flag[task_start_index:] == 1.0,1], target_q[demo_flag[task_start_index:] == 1.0,1])
            critic_loss_termination_flag = 10*F.mse_loss(q1[:,2], target_q[:,2])
            critic_loss = critic_loss_agent + critic_loss_user + critic_loss_termination_flag
            self.logger.log(DataType.num, critic_loss_agent.item(), key = "td3/loss/critic1_agent")
            self.logger.log(DataType.num, critic_loss_user.item(), key = "td3/loss/critic1_user")
            self.logger.log(DataType.num, critic_loss_termination_flag.item(), key = "td3/loss/critic1_termination_flag")
            self.logger.log(DataType.num, critic_loss.item(), key = "td3/loss/critic1_total")

            print("# 7.1 Optimize critic")
            self.critic_1_optimizer.zero_grad(set_to_none = True)
            critic_loss.backward()
            self.critic_1_optimizer.step()

            del q1, critic_loss, critic_loss_user, critic_loss_agent, critic_loss_termination_flag
            torch.cuda.empty_cache() 

            print("# 5.2 Compute Q-value from critics Q(s_t, a_t)")
            q2 = self.critic_model_2(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = actions)
            q2 = q2[task_start_index:]

            print("# 6.2 Compute MSE loss for the critics")
            critic_loss_agent = F.mse_loss(q2[demo_flag[task_start_index:] == 0.0,0], target_q[demo_flag[task_start_index:] == 0.0,0])
            critic_loss_user = 10*F.mse_loss(q2[demo_flag[task_start_index:] == 1.0,1], target_q[demo_flag[task_start_index:] == 1.0,1])
            critic_loss_termination_flag = 10*F.mse_loss(q2[:,2], target_q[:,2])
            critic_loss = critic_loss_agent + critic_loss_user + critic_loss_termination_flag
            self.logger.log(DataType.num, critic_loss_agent.item(), key = "td3/loss/critic2_agent")
            self.logger.log(DataType.num, critic_loss_user.item(), key = "td3/loss/critic2_user")
            self.logger.log(DataType.num, critic_loss_termination_flag.item(), key = "td3/loss/critic2_termination_flag")
            self.logger.log(DataType.num, critic_loss.item(), key = "td3/loss/critic2_total")

            print("# 7.2 Optimize critic")
            self.critic_2_optimizer.zero_grad(set_to_none = True)
            critic_loss.backward()
            self.critic_2_optimizer.step()

            del q2, critic_loss, critic_loss_agent, critic_loss_user, critic_loss_termination_flag
            del target_q
            torch.cuda.empty_cache() 

            print("# 8. Check whether to update the actor and the target policies")
            if(j % self.actor_update_period == (self.actor_update_period - 1)):
                print("# 9. Compute the actor's action using the real actor")
                actor_actions = self.actor_model(input = images, input_latent = prev_latent).permute((1,0))
                print("# 10. Compute the negative critic values using the real critic")
                dummy_critic = self.critic_model_1.to(self.device2)
                actor_losses = -dummy_critic(
                    input = images.to(self.device2), 
                    input_latent = prev_latent[:-1,:].to(self.device2), 
                    pre_output_latent = actor_actions.to(self.device2))[task_start_index:]
                actor_loss_agent = actor_losses[demo_flag[task_start_index:] == 0.0, 0].mean()
                actor_loss_user = actor_losses[demo_flag[task_start_index:] == 1.0, 1].mean()
                actor_loss_termination_flag = actor_losses[:, 2].mean()
                actor_loss = (actor_loss_agent + actor_loss_user + actor_loss_termination_flag)/3.0

                actor_actions.detach()
                del actor_actions

                print("# 11. Optimize actor")
                self.actor_optimizer.zero_grad(set_to_none = True)
                actor_loss.backward()
                self.actor_optimizer.step()
                self.logger.log(DataType.num, actor_loss.item(), key = "td3/loss/actor")

                print("# 12. Update target networks")
                for param, target_param in zip(self.critic_model_1.to(self.device1).parameters(), self.critic_model_1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
                
                del param, target_param

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                print("Saving checkpoint")
                torch.save({
                            'actor_state_dict': self.actor_model.state_dict(),
                            'actor_target_state_dict': self.actor_model_target.state_dict(),
                            'critic_1_state_dict': self.critic_model_1.state_dict(),
                            'critic_2_state_dict': self.critic_model_2.state_dict(),
                            'critic_1_target_state_dict': self.critic_model_1_target.state_dict(),
                            'critic_2_target_state_dict': self.critic_model_2_target.state_dict(),
                            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
                            'critic_2_optimizer_state_dict': self.critic_2_optimizer.state_dict(),
                            }, self.checkpoint_path)

            j += 1

class IL(Algorithm):
    def __init__(self,
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                actor_optimizer_dict,
                dataloader,
                device,
                logger_name= None):
        '''
        IL (Behavior Cloning) Algorithm Implementation.

        actor_model -- nn.Module model 
        actor_optimizer_dict -- dictionary with information about optimizer
        dataloader -- torch.utils.data.Dataloader
        device -- device name in strings Eg. "cuda:1"
        logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device = device

        self.actor_model = actor_model

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)
        
        self.actor_model = self.actor_model.to(self.device)

        self.optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())

        if(checkpoint is not None):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.dataloader = dataloader

        self.checkpoint_every = checkpoint_every
        config_dict = dict(
            actor_optimizer = actor_optimizer_dict,
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)

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
            dup_latent[-1,frame_no.shape[0]+1:] = 0.0

            print("# 1. Compute actor output")
            actor_output = self.actor_model(dup_images, dup_latent)

            target = actions.T
            output = actor_output[frame_no.shape[0]:]
            print("=======target")
            print(target)
            print("=======output")
            print(output)

            print("# 2. Compute loss")
            loss = F.mse_loss(target, output)

            print("# 3. Optimize actor model")
            self.optimizer.zero_grad(set_to_none = True)
            loss.backward()
            self.optimizer.step()

            self.logger.log(data_type = DataType.num, data = loss.item(), key = "il/loss/actor")

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                print("Saving checkpoint")
                torch.save({
                            'actor_state_dict': self.actor_model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            }, self.checkpoint_path)

            j += 1


class DDPG(Algorithm):
    def __init__(self, 
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                critic_model,
                actor_optimizer_dict, 
                critic_optimizer_dict, 
                dataloader, 
                device1,
                device2,
                logger_name = None,
                discount = 0.99,
                tau = 0.005,
                noise = 0.5,
                actor_update_period = 2,
                ):
        '''
        TD3 algorithm implementation.

        Keyword arguments:
        actor_model -- the actor model to be trained *
        critic_model -- the critic model to be trained *
        actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        dataloader -- the dataloader to be used to train the models *
        device -- the device to put the models in *
        logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
        discount -- discount factor used to compute the value of a state-action pair (float)
        tau -- the rate at which the target policies are updated (float)
        noise -- noise value from 0.0 - 1.0
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device1 = device1
        self.device2 = device2

        self.actor_model = actor_model
        self.critic_model_1 = critic_model

        self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model)).eval()
        self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1)).eval()

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_model_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_model_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_model_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)

        self.actor_model.to(self.device1)
        self.critic_model_1.to(self.device1)
        self.actor_model_target.to(self.device1)
        self.critic_model_1_target.to(self.device1)

        self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
        self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())

        if(checkpoint is not None):
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])

        self.dataloader = dataloader

        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.actor_update_period = actor_update_period

        self.checkpoint_every = checkpoint_every
        config_dict = dict(
            actor_optimizer = actor_optimizer_dict,
            critic_optimizer = critic_optimizer_dict,
            discount = self.discount, 
            tau = self.tau, 
            noise = self.noise, 
            actor_update_period = self.actor_update_period
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)
    
    def train_one_epoch(self, trainer):
        j = 0
        for (images, latent, frame_no, rewards_agent, rewards_user, rewards_termination_flag) in self.dataloader:
            if(trainer.stop == True):
                return

            print(f"Run No. {j+1}")
            print(f"Episode Length = {frame_no.shape[0]}")

            task_start_index = (frame_no == 1).nonzero()[1].item()

            images = images.to(self.device1)
            latent = latent.to(self.device1)
            rewards_agent = rewards_agent.to(self.device1)
            rewards_user = rewards_user.to(self.device1)
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
                target_q = self.critic_model_1_target(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = target_actions)
                target_q[demo_flag == 1,0] = 0
                self.logger.log(DataType.num, target_q.mean().item(), "td3/avg-value")

                del target_actions

                print("# 3. Compute current Q-value with the reward")
                target_q_next = torch.cat((target_q[1:,0],torch.zeros(1).to(self.device1)), dim = 0)
                target_q[:,0] = rewards_agent + self.discount * target_q_next
                target_q[:,1] = rewards_user
                target_q[:,2] = rewards_termination_flag
                target_q = target_q[task_start_index:]

                del target_q_next

            print("# 5.1 Compute Q-value from critics Q(s_t, a_t)")
            q1 = self.critic_model_1(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = actions)
            q1 = q1[task_start_index:]

            print("# 6.1 Compute MSE loss for the critics")
            critic_loss_agent = F.mse_loss(q1[demo_flag[task_start_index:] == 0.0,0], target_q[demo_flag[task_start_index:] == 0.0,0])
            critic_loss_user = 10*F.mse_loss(q1[demo_flag[task_start_index:] == 1.0,1], target_q[demo_flag[task_start_index:] == 1.0,1])
            critic_loss_termination_flag = 10*F.mse_loss(q1[:,2], target_q[:,2])
            critic_loss = critic_loss_agent + critic_loss_user + critic_loss_termination_flag
            self.logger.log(DataType.num, critic_loss_agent.item(), key = "td3/loss/critic1_agent")
            self.logger.log(DataType.num, critic_loss_user.item(), key = "td3/loss/critic1_user")
            self.logger.log(DataType.num, critic_loss_termination_flag.item(), key = "td3/loss/critic1_termination_flag")
            self.logger.log(DataType.num, critic_loss.item(), key = "td3/loss/critic1_total")

            print("# 7.1 Optimize critic")
            self.critic_1_optimizer.zero_grad(set_to_none = True)
            critic_loss.backward()
            self.critic_1_optimizer.step()

            del q1, critic_loss, critic_loss_user, critic_loss_agent, critic_loss_termination_flag

            print("# 8. Check whether to update the actor and the target policies")
            if(j % self.actor_update_period == (self.actor_update_period - 1)):
                print("# 9. Compute the actor's action using the real actor")
                actor_actions = self.actor_model(input = images, input_latent = prev_latent).permute((1,0))
                print("# 10. Compute the negative critic values using the real critic")
                dummy_critic = self.critic_model_1.to(self.device2)
                actor_losses = -dummy_critic(
                    input = images.to(self.device2), 
                    input_latent = prev_latent[:-1,:].to(self.device2), 
                    pre_output_latent = actor_actions.to(self.device2))[task_start_index:]
                actor_loss_agent = actor_losses[demo_flag[task_start_index:] == 0.0, 0].mean()
                actor_loss_user = actor_losses[demo_flag[task_start_index:] == 1.0, 1].mean()
                actor_loss_termination_flag = actor_losses[:, 2].mean()
                actor_loss = (actor_loss_agent + actor_loss_user + actor_loss_termination_flag)/3.0

                actor_actions.detach()
                del actor_actions

                print("# 11. Optimize actor")
                self.actor_optimizer.zero_grad(set_to_none = True)
                actor_loss.backward()
                self.actor_optimizer.step()
                self.logger.log(DataType.num, actor_loss.item(), key = "td3/loss/actor")

                print("# 12. Update target networks")
                for param, target_param in zip(self.critic_model_1.to(self.device1).parameters(), self.critic_model_1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
                
                del param, target_param

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                print("Saving checkpoint")
                torch.save({
                            'actor_state_dict': self.actor_model.state_dict(),
                            'actor_target_state_dict': self.actor_model_target.state_dict(),
                            'critic_1_state_dict': self.critic_model_1.state_dict(),
                            'critic_1_target_state_dict': self.critic_model_1_target.state_dict(),
                            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
                            }, self.checkpoint_path)

            j += 1

class MC(Algorithm):
    def __init__(self, 
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                critic_model,
                actor_optimizer_dict, 
                critic_optimizer_dict, 
                dataloader, 
                device1,
                device2,
                logger_name = None,
                discount = 0.99,
                tau = 0.005,
                noise = 0.5,
                actor_update_period = 2,
                ):
        '''
        MC algorithm implementation.

        Keyword arguments:
        actor_model -- the actor model to be trained *
        critic_model -- the critic model to be trained *
        actor_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        critic_optimizer_dict -- the dictionary representation of the optimizer for the actor model *
        dataloader -- the dataloader to be used to train the models *
        device -- the device to put the models in *
        logger_dict -- dictionary with information about the logger to initialize dict(name, kwargs)
        discount -- discount factor used to compute the value of a state-action pair (float)
        tau -- the rate at which the target policies are updated (float)
        noise -- noise value from 0.0 - 1.0
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device1 = device1
        self.device2 = device2

        self.actor_model = actor_model
        self.critic_model_1 = critic_model

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.critic_model_1.load_state_dict(checkpoint['critic_1_state_dict'])
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)

        self.actor_model.to(self.device1)
        self.critic_model_1.to(self.device1)

        self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())
        self.critic_1_optimizer = create_optimizer_from_dict(critic_optimizer_dict, self.critic_model_1.parameters())

        if(checkpoint is not None):
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])

        self.dataloader = dataloader

        self.discount = discount
        self.tau = tau
        self.noise = noise
        self.actor_update_period = actor_update_period

        self.checkpoint_every = checkpoint_every
        config_dict = dict(
            actor_optimizer = actor_optimizer_dict,
            critic_optimizer = critic_optimizer_dict,
            discount = self.discount, 
            tau = self.tau, 
            noise = self.noise, 
            actor_update_period = self.actor_update_period
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)
    
    def train_one_epoch(self, trainer):
        j = 0
        for (images, latent, frame_no, rewards_agent, rewards_user, rewards_termination_flag) in self.dataloader:
            if(trainer.stop == True):
                return

            print(f"Run No. {j+1}")
            print(f"Episode Length = {frame_no.shape[0]}")

            task_start_index = (frame_no == 1).nonzero()[1].item()

            images = images.to(self.device1)
            latent = latent.to(self.device1)
            rewards_agent = rewards_agent.to(self.device1)
            rewards_user = rewards_user.to(self.device1)
            rewards_termination_flag = rewards_termination_flag.to(self.device1)

            actions = latent[:-1,:]
            demo_flag = latent[-1,:]
            initial_action = torch.zeros_like(latent[:,0])
            initial_action[3] = 1.0
            prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

            with torch.no_grad():
                target_q = torch.zeros((frame_no.shape[0], 3)).to(self.device1)
                target_q[:,0] = compute_values(self.discount, rewards_agent)
                target_q[:,1] = rewards_user
                target_q[:,2] = rewards_termination_flag
                target_q = target_q[task_start_index:]

            print("# 5.1 Compute Q-value from critics Q(s_t, a_t)")
            self.critic_model_1.to(self.device1)
            q1 = self.critic_model_1(input = images, input_latent = prev_latent[:-1,:], pre_output_latent = actions)
            q1 = q1[task_start_index:]

            print("target", target_q)
            print("q1", q1)

            print("# 6.1 Compute MSE loss for the critics")
            critic_loss_agent = F.mse_loss(q1[demo_flag[task_start_index:] == 0.0,0], target_q[demo_flag[task_start_index:] == 0.0,0])
            critic_loss_user = F.binary_cross_entropy(q1[demo_flag[task_start_index:] == 1.0,1], target_q[demo_flag[task_start_index:] == 1.0,1])
            critic_loss_termination_flag = F.binary_cross_entropy(q1[:,2], target_q[:,2])
            critic_loss = critic_loss_agent + critic_loss_user + critic_loss_termination_flag
            self.logger.log(DataType.num, critic_loss_agent.item(), key = "loss/critic1_agent")
            self.logger.log(DataType.num, critic_loss_user.item(), key = "loss/critic1_user")
            self.logger.log(DataType.num, critic_loss_termination_flag.item(), key = "loss/critic1_termination_flag")
            self.logger.log(DataType.num, critic_loss.item(), key = "loss/critic1_total")

            print("# 7.1 Optimize critic")
            self.critic_1_optimizer.zero_grad(set_to_none = True)
            critic_loss.backward()
            self.critic_1_optimizer.step()

            print("# 8. Check whether to update the actor and the target policies")
            if(j % self.actor_update_period == (self.actor_update_period - 1)):
                print("# 9. Compute the actor's action using the real actor")
                actor_actions = self.actor_model(input = images, input_latent = prev_latent).permute((1,0))
                print("actor_actions", actor_actions.T[task_start_index:])
                print("# 10. Compute the negative critic values using the real critic")
                dummy_critic = self.critic_model_1.to(self.device2)
                actor_losses = -dummy_critic(
                    input = images.to(self.device2), 
                    input_latent = prev_latent[:-1,:].to(self.device2), 
                    pre_output_latent = actor_actions.to(self.device2))[task_start_index:]
                actor_loss_agent = actor_losses[demo_flag[task_start_index:] == 0.0, 0].mean()
                actor_loss_user = actor_losses[demo_flag[task_start_index:] == 1.0, 1].mean()
                actor_loss_termination_flag = actor_losses[:, 2].mean()
                actor_loss = (actor_loss_agent + actor_loss_user + actor_loss_termination_flag)/3.0

                actor_actions.detach()
                del actor_actions

                print("# 11. Optimize actor")
                self.actor_optimizer.zero_grad(set_to_none = True)
                actor_loss.backward()
                self.actor_optimizer.step()
                self.logger.log(DataType.num, actor_loss.item(), key = "loss/actor")

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                print("Saving checkpoint")
                torch.save({
                            'actor_state_dict': self.actor_model.state_dict(),
                            'critic_1_state_dict': self.critic_model_1.state_dict(),
                            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                            'critic_1_optimizer_state_dict': self.critic_1_optimizer.state_dict(),
                            }, self.checkpoint_path)

            j += 1