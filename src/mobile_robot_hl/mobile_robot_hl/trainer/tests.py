from .utils import *

def velocity_similarity_test(dataset):
	similarity = 0.0
	for (
			name, 
			id_, 
			images, 
			latent, 
			frame_no, 
			agent_linear_vel,
			agent_angular_vel,
			agent_termination_flag,
			user_linear_vel,
			user_angular_vel,
			user_termination_flag,
			) in dataset:

		demo_flag = latent[-1,:]
		task_start_index = (frame_no == 1).nonzero()[1].item()

		user_linear_vel = user_linear_vel[task_start_index:]
		user_angular_vel = user_angular_vel[task_start_index:]
		agent_linear_vel = agent_linear_vel[task_start_index:]
		agent_angular_vel = agent_angular_vel[task_start_index:]

		sim = compute_similarity(
			user_linear_vel[demo_flag[task_start_index:] == 0.0], 
			user_angular_vel[demo_flag[task_start_index:] == 0.0], 
			agent_linear_vel[demo_flag[task_start_index:] == 0.0], 
			agent_angular_vel[demo_flag[task_start_index:] == 0.0])
		similarity += sim.mean()/len(dataset)
	
	print(f"Similarity = {similarity}")
	return similarity

def failure_rate_test(dataset):
	failure_rate = 0.0
	for (
			name, 
			id_, 
			images, 
			latent, 
			frame_no, 
			agent_linear_vel,
			agent_angular_vel,
			agent_termination_flag,
			user_linear_vel,
			user_angular_vel,
			user_termination_flag,
			) in dataset:

		demo_flag = latent[-1,:]
		task_start_index = (frame_no == 1).nonzero()[1].item()
		if((demo_flag[task_start_index:] == 1.0).nonzero().shape[0] == 0):
			pass
		else:
			failure_rate += 1/len(dataset)
	
	print(f"Failure Rate = {failure_rate}")
	return failure_rate

def supervisor_ratio_test(dataset):
	supervisor_ratio = 0.0
	for (
			name, 
			id_, 
			images, 
			latent, 
			frame_no, 
			agent_linear_vel,
			agent_angular_vel,
			agent_termination_flag,
			user_linear_vel,
			user_angular_vel,
			user_termination_flag,
			) in dataset:

		demo_flag = latent[-1,:]
		task_start_index = (frame_no == 1).nonzero()[1].item()

		num_of_demo = (demo_flag[task_start_index:] == 1.0).nonzero().shape[0]

		supervisor_ratio += (num_of_demo/demo_flag[task_start_index:].shape[0])/len(dataset)
	
	print(f"Supervisor Ratio = {supervisor_ratio}")
	return supervisor_ratio

def max_output_test(dataset):
	max_output = 0.0
	for (
			name, 
			id_, 
			images, 
			latent, 
			frame_no, 
			agent_linear_vel,
			agent_angular_vel,
			agent_termination_flag,
			user_linear_vel,
			user_angular_vel,
			user_termination_flag,
			) in dataset:

		demo_flag = latent[-1,:]
		task_start_index = (frame_no == 1).nonzero()[1].item()

		agent_linear_vel = agent_linear_vel[demo_flag == 0.0]
		agent_angular_vel = agent_angular_vel[demo_flag == 0.0]

		agent_vel = torch.cat((agent_linear_vel.unsqueeze(1), agent_angular_vel.unsqueeze(1)), dim = 1)
		agent_mag = torch.sum(agent_vel**2, dim = 1)**0.5
		max_mag = torch.max(agent_mag)
		max_output += max_mag.mean()/len(dataset)
	
	print(f"Maximum Output Average = {max_output}")
	return max_output
