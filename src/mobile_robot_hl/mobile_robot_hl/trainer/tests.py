import matplotlib.pyplot as plt
from .utils import *

def velocity_similarity_test(dataset, save_path = None):
	similarity = 0.0
	similarity_arr = []
	x_arr = []
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

		sim = compute_similarity_normalized(
			user_linear_vel[demo_flag[task_start_index:] == 0.0], 
			user_angular_vel[demo_flag[task_start_index:] == 0.0], 
			agent_linear_vel[demo_flag[task_start_index:] == 0.0], 
			agent_angular_vel[demo_flag[task_start_index:] == 0.0])
		similarity += sim.mean()/len(dataset)
		x_arr.append(id_)
		similarity_arr.append(sim.mean())
	
	print(f"Similarity = {similarity}")
	if(save_path is not None):
		plt.plot(x_arr, similarity_arr)
		plt.savefig(save_path)
		print(f"Saved to {save_path}")

	return similarity

def failure_rate_test(dataset, save_path = None):
	failure_rate = 0.0
	x_arr = []
	failure_rate_arr = []
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
			failure_rate_arr.append(0)
		else:
			failure_rate += 1/len(dataset)
			failure_rate_arr.append(1)
		
		x_arr.append(id_)
	
	print(f"Failure Rate = {failure_rate}")
	if(save_path is not None):
		plt.plot(x_arr, failure_rate_arr)
		plt.savefig(save_path)
		print(f"Saved to {save_path}")

	return failure_rate

def average_supervised_frames_test(dataset, save_path = None):
	supervised_frames = 0
	x_arr = []
	supervised_frames_arr = []
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
		demo_flag = demo_flag[task_start_index:]

		num_take_over = (demo_flag[1:]-demo_flag[:-1] == 1.0).nonzero().shape[0]

		demo_flag = latent[-1,:]
		task_start_index = (frame_no == 1).nonzero()[1].item()

		num_of_demo = (demo_flag[task_start_index:] == 1.0).nonzero().shape[0]

		supervised_frames += (num_of_demo)/len(dataset)
		x_arr.append(id_)
		supervised_frames_arr.append((num_of_demo))
	
	print(f"Average supervised frames per episode = {supervised_frames}")
	if(save_path is not None):
		plt.plot(x_arr, supervised_frames_arr)
		plt.savefig(save_path)
		print(f"Saved to {save_path}")

	return supervised_frames

def average_take_over_count_test(dataset, save_path = None):
	take_over_count = 0
	x_arr = []
	take_over_count_arr = []
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
		demo_flag = demo_flag[task_start_index:]

		num_take_over = (demo_flag[1:]-demo_flag[:-1] == 1.0).nonzero().shape[0]

		take_over_count += (num_take_over)/len(dataset)
		x_arr.append(id_)
		take_over_count_arr.append((num_take_over))
	
	print(f"Average take over count = {take_over_count}")
	if(save_path is not None):
		plt.plot(x_arr, take_over_count_arr)
		plt.savefig(save_path)
		print(f"Saved to {save_path}")

	return take_over_count

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

def average_time_test(dataset, save_path = None):
	avg_time = 0.0
	x_arr = []
	avg_time_arr = []
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

		task_start_index = (frame_no == 1).nonzero()[1].item()

		avg_time += frame_no[-1].item()/len(dataset)
		x_arr.append(id_)
		avg_time_arr.append(frame_no[-1].item())
	
	print(f"Average Time = {avg_time}")
	if(save_path is not None):
		plt.plot(x_arr, avg_time_arr)
		plt.savefig(save_path)
		print(f"Saved to {save_path}")
	return avg_time
