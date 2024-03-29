import pickle
import torch
import torch.nn.functional as F
import time
import os
import math
import random

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

class TD3_SLRL(Algorithm):
    def __init__(self, 
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                critic_model,
                actor_optimizer_dict, 
                critic_optimizer_dict, 
                dataloader, 
                device,
                logger_name = None,
                discount = 0.99,
                tau = 0.005,
                noise = 0.5,
                run_decay=0.0,
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
        run_decay -- decays the percentage that the model trains on a particular mini-batch. run_decay <= 0.0
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device = device
        self.run_decay = run_decay
        self.run_no = dict()

        self.actor_model = actor_model
        self.critic_model_1 = critic_model
        self.critic_model_2 = pickle.loads(pickle.dumps(self.critic_model_1))

        self.actor_model_target = pickle.loads(pickle.dumps(self.actor_model))
        self.critic_model_1_target = pickle.loads(pickle.dumps(self.critic_model_1))
        self.critic_model_2_target = pickle.loads(pickle.dumps(self.critic_model_1))

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            self.actor_model_target.load_state_dict(checkpoint['actor_target_state_dict'])
            self.critic_model_1.load_state_dict(checkpoint['critic_1_state_dict'])
            self.critic_model_2.load_state_dict(checkpoint['critic_2_state_dict'])
            self.critic_model_1_target.load_state_dict(checkpoint['critic_1_target_state_dict'])
            self.critic_model_2_target.load_state_dict(checkpoint['critic_2_target_state_dict'])
            if('run_no' in checkpoint.keys()):
                self.run_no = checkpoint['run_no']
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)

        self.actor_model.to(self.device)
        self.critic_model_1.to(self.device)
        self.critic_model_2.to(self.device)
        self.actor_model_target.to(self.device)
        self.critic_model_1_target.to(self.device)
        self.critic_model_2_target.to(self.device)

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
            actor_update_period = self.actor_update_period,
            run_decay = self.run_decay
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)
    
    def train_one_epoch(self, trainer):
        j = 0
        for (name, id_, images, latent, frame_no, rewards_agent, desired_termination_flag) in self.dataloader:
            if(trainer.stop == True):
                break

            name_id = f"{name}/{id_}"

            print(f"Episode Name/ID: {name_id}")
            print(f"Run No. {j+1}")
            print(f"Episode Length = {frame_no.shape[0]}")

            if(self.run_decay != 0.0 or self.run_decay is not None):
                if(name_id in self.run_no.keys()):
                    run_no = self.run_no[name_id]
                    run_percent = math.exp(self.run_decay*run_no)
                    self.run_no[name_id] = run_no + 1
                    if(random.uniform(0,1) > run_percent):
                        print("Skipped episode")
                        j+=1
                        continue
                else:
                    self.run_no[name_id] = 1

            log_dict = dict()

            task_start_index = (frame_no == 1).nonzero()[1].item()

            images = images.to(self.device)
            latent = latent.to(self.device)
            rewards_agent = rewards_agent.to(self.device)
            desired_termination_flag = desired_termination_flag.to(self.device)
            frame_no = frame_no.to(self.device)

            actions = latent[:-1,:]
            demo_flag = latent[-1,:]
            initial_action = torch.zeros_like(latent[:,0])
            initial_action[3] = 1.0
            prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

            with torch.no_grad():
                print("# 1. Compute target actions from target actor P'(s(t+1))")
                target_actions = self.actor_model_target(input = images, input_latent = prev_latent, frame_no = frame_no, noise = self.noise).permute((1,0)) 

                print("# 2. Compute Q-value of next state using the  target critics Q'(s(t+1), P'(s(t+1)))")
                target_q1 = self.critic_model_1_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions, frame_no = frame_no)
                target_q2 = self.critic_model_2_target(input = images, input_latent = prev_latent, pre_output_latent = target_actions, frame_no = frame_no)

                print("# 3. Use smaller Q-value as the Q-value target")
                target_q_min = torch.min(target_q1, target_q2)
                change_flag = torch.cat((torch.tensor([0]).to(self.device), demo_flag[1:]-demo_flag[:-1]))
                target_q_min[change_flag == 1] = 0.0

                print("# 4. Compute current Q-value with the reward")
                target_q_next = torch.cat((target_q_min[1:],torch.zeros(1).to(self.device)), dim = 0)
                target_q = rewards_agent + self.discount * target_q_next
                target_q = target_q[task_start_index:] 
                print("target_q", target_q)

            print("# 5.1 Compute Q-value from critics Q(s_t, a_t)")
            q1 = self.critic_model_1(input = images, input_latent = prev_latent, pre_output_latent = actions, frame_no = frame_no)
            q1 = q1[task_start_index:]
            #self.logger.log(DataType.num, q1[demo_flag[task_start_index:] == 0].mean().item(), "avg-value")
            log_dict['avg-value'] = q1[demo_flag[task_start_index:] == 0].mean().item()

            print("# 6.1 Compute MSE loss for the critics")
            critic_1_se = (q1 - target_q)**2
            critic_1_loss = critic_1_se.mean()
            #self.logger.log(DataType.num, critic_1_loss.item(), key = "loss/critic1")
            log_dict['loss/critic1'] = critic_1_loss.item()

            print("# 7.1 Optimize critic")
            self.critic_1_optimizer.zero_grad(set_to_none = True)
            critic_1_loss.backward()
            self.critic_1_optimizer.step()

            print("# 5.2 Compute Q-value from critics Q(s_t, a_t)")
            q2 = self.critic_model_2(input = images, input_latent = prev_latent, pre_output_latent = actions, frame_no = frame_no)
            q2 = q2[task_start_index:]

            print("# 6.2 Compute MSE loss for the critics")
            critic_2_se = (q2 - target_q)**2
            critic_2_loss = critic_2_se.mean()
            #self.logger.log(DataType.num, critic_2_loss.item(), key = "loss/critic2")
            log_dict['loss/critic2'] = critic_2_loss.item()

            print("# 7.2 Optimize critic")
            self.critic_2_optimizer.zero_grad(set_to_none = True)
            critic_2_loss.backward()
            self.critic_2_optimizer.step()

            print("# 8. Compute actor actions")
            actor_actions = self.actor_model(input = images, input_latent = prev_latent, frame_no = frame_no)
            print("actor actions", actor_actions[demo_flag == 0.0])
            actor_linear_vel = actor_actions[:,0]
            actor_angular_vel = actor_actions[:,1]
            actor_termination_flag = actor_actions[:,2]

            print("# 9. Compute actor loss")
            velocity_loss = -compute_similarity(actions[0,:], actions[1,:], actor_linear_vel, actor_angular_vel)[task_start_index:]
            velocity_loss[demo_flag[task_start_index:] == 0.0] = velocity_loss[demo_flag[task_start_index:] == 0.0]*0.03
            velocity_loss = velocity_loss.mean()
            termination_flag_loss = F.binary_cross_entropy(actor_termination_flag, desired_termination_flag)
            actor_loss = velocity_loss + termination_flag_loss
            #self.logger.log(DataType.num, velocity_loss.item(), key = "loss/actor_velocity")
            log_dict['loss/actor_velocity'] = velocity_loss.item()
            #self.logger.log(DataType.num, termination_flag_loss.item(), key = "loss/actor_termination_flag")
            log_dict['loss/actor_termination_flag'] = termination_flag_loss.item()

            if(j % self.actor_update_period == (self.actor_update_period - 1)):
                print("# 10. Compute the negative critic values using the real critic")
                negative_value = -self.critic_model_1(
                                    input = images,
                                    input_latent = prev_latent,
                                    pre_output_latent = actor_actions.T,
                                    frame_no = frame_no)[task_start_index:]
                negative_value = negative_value.mean()
                #self.logger.log(DataType.num, negative_value.item(), key = "loss/actor_neg_value")
                log_dict['loss/actor_neg_value'] = negative_value.item()
                actor_loss += negative_value*0.2

            print("# 11. Optimize actor")
            self.actor_optimizer.zero_grad(set_to_none = True)
            actor_loss.backward()
            self.actor_optimizer.step()

            print("# 12. Update target networks")
            for param, target_param in zip(self.critic_model_1.to(self.device).parameters(), self.critic_model_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_model_2.parameters(), self.critic_model_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor_model.parameters(), self.actor_model_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)	
            
            del param, target_param

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                self.checkpoint()

            self.logger.log(DataType.dict, log_dict, key = None)

            j += 1
        self.checkpoint()
    
    def checkpoint(self):
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
                    'run_no': self.run_no,
                    }, self.checkpoint_path)

class SL(Algorithm):
    def __init__(self, 
                run_name,
                run_id,
                checkpoint_every,
                actor_model,
                actor_optimizer_dict, 
                dataloader, 
                device,
                only_agent = False,
                logger_name = None,
                run_decay = 0.0
                ):
        '''
        Supervised Learning (SL) algorithm implementation.
        '''

        self.run_name = f"{run_name}-{self.__class__.__name__}"
        self.run_id = str(run_id)
        self.checkpoint_path = os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id, "checkpoint.pth")
        self.device = device
        self.only_agent = only_agent
        self.run_decay = run_decay
        self.run_no = dict()

        self.actor_model = actor_model

        checkpoint = None
        if(os.path.exists(self.checkpoint_path)):
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
            self.actor_model.load_state_dict(checkpoint['actor_state_dict'])
            if('run_no' in checkpoint.keys()):
                self.run_no = checkpoint['run_no']
        else:
            os.makedirs(os.path.join(CHECKPOINT_PATH, self.run_name, self.run_id), exist_ok=True)

        self.actor_model.to(self.device)

        self.actor_optimizer = create_optimizer_from_dict(actor_optimizer_dict, self.actor_model.parameters())

        if(checkpoint is not None):
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        self.dataloader = dataloader

        self.checkpoint_every = checkpoint_every
        config_dict = dict(
            actor_optimizer = actor_optimizer_dict,
            run_decay = self.run_decay
            )
        self.logger = create_logger(self.run_name, self.run_id, config_dict, logger_name)
    
    def train_one_epoch(self, trainer):
        j = 0
        for (name, id_, images, latent, frame_no, rewards_agent, desired_termination_flag) in self.dataloader:
            if(trainer.stop == True):
                break

            name_id = f"{name}/{id_}"

            print(f"Episode Name/ID: {name_id}")
            print(f"Run No. {j+1}")
            print(f"Episode Length = {frame_no.shape[0]}")

            if(self.run_decay != 0.0 or self.run_decay is not None):
                if(name_id in self.run_no.keys()):
                    run_no = self.run_no[name_id]
                    run_percent = math.exp(self.run_decay*run_no)
                    self.run_no[name_id] = run_no + 1
                    if(random.uniform(0,1) > run_percent):
                        print("Skipped episode")
                        j+=1
                        continue
                else:
                    self.run_no[name_id] = 1
            
            log_dict = dict()

            task_start_index = (frame_no == 1).nonzero()[1].item()

            images = images.to(self.device)
            latent = latent.to(self.device)
            desired_termination_flag = desired_termination_flag.to(self.device)
            frame_no = frame_no.to(self.device)

            actions = latent[:-1,:]
            demo_flag = latent[-1,:]
            initial_action = torch.zeros_like(latent[:,0])
            initial_action[3] = 1.0
            prev_latent = torch.cat((initial_action.unsqueeze(1), latent[:,:-1]), dim = 1)

            print("# 8. Compute actor actions")
            actor_actions = self.actor_model(input = images, input_latent = prev_latent, frame_no = frame_no)
            print("actor actions", actor_actions[demo_flag == 0.0])
            actor_linear_vel = actor_actions[:,0]
            actor_angular_vel = actor_actions[:,1]
            actor_termination_flag = actor_actions[:,2]

            print("# 9. Compute actor loss")
            velocity_loss = -compute_similarity(actions[0,:], actions[1,:], actor_linear_vel, actor_angular_vel)[task_start_index:]
            if(self.only_agent == True):
                velocity_loss = velocity_loss[demo_flag[task_start_index:] == 1.0]
                if(velocity_loss.shape[0] > 0):
                    velocity_loss = velocity_loss.mean()
                else:
                    velocity_loss = torch.tensor(0.0).to(self.device)
            else:
                velocity_loss[demo_flag[task_start_index:] == 0.0] = velocity_loss[demo_flag[task_start_index:] == 0.0]*0.1
                velocity_loss = velocity_loss.mean()
            termination_flag_loss = F.binary_cross_entropy(actor_termination_flag, desired_termination_flag)
            actor_loss = velocity_loss + termination_flag_loss
            log_dict['loss/actor_velocity'] = velocity_loss.item()
            log_dict['loss/actor_termination_flag'] = termination_flag_loss.item()

            print("# 11. Optimize actor")
            self.actor_optimizer.zero_grad(set_to_none = True)
            actor_loss.backward()
            self.actor_optimizer.step()

            if(j % self.checkpoint_every == self.checkpoint_every-1):
                self.checkpoint()

            self.logger.log(DataType.dict, log_dict, key = None)

            j += 1
        self.checkpoint()
    
    def checkpoint(self):
        print("Saving checkpoint")
        torch.save({
                    'actor_state_dict': self.actor_model.state_dict(),
                    'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'run_no': self.run_no,
                    }, self.checkpoint_path)