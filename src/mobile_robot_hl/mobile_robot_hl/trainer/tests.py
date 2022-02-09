import matplotlib.pyplot as plt
import numpy as np
import sys

from .utils import *

def velocity_similarity_test(dataset, save_path = None, label = None):
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
        plt.title("Velocity Similarity")
        plt.xlabel("Episode")
        plt.ylabel("Similarity (Negative Normalized Euclidean Distance)")
        similarity_arr = np.array([x.item() for _, x in sorted(zip(x_arr, similarity_arr))])
        x_arr = np.array(sorted(x_arr))
        N = 9
        similarity_avg = np.convolve(similarity_arr, np.ones(N)/N, mode='valid')
        plt.scatter(x_arr, similarity_arr, s = 10, label = label, alpha = 0.5)
        if(label != None):
            label = label + f" Moving Average (N = {N})"
        plt.plot(x_arr[int(N/2):-int(N/2)], similarity_avg, label = label, linewidth = 2)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"Saved to {save_path}")

    return similarity, (x_arr, similarity_arr)

def failure_rate_test(dataset, save_path = None, label = None):
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
        plt.title("Failures")
        plt.xlabel("Episode")
        plt.ylabel("1 = Failure, 0 = Success")
        failure_rate_arr = np.array([x for _, x in sorted(zip(x_arr, failure_rate_arr))])
        x_arr = np.array(sorted(x_arr))
        N = 9
        failure_avg = np.convolve(failure_rate_arr, np.ones(N)/N, mode='valid')
        plt.scatter(x_arr, failure_rate_arr, s = 10, label = label, alpha = 0.5)
        if(label != None):
            label = label + f" Moving Average (N = {N})"
        plt.plot(x_arr[int(N/2):-int(N/2)], failure_avg, label = label, linewidth = 2)
        plt.grid(True)
        plt.legend(loc = 'lower left')
        plt.savefig(save_path)
        print(f"Saved to {save_path}")


    return failure_rate, (x_arr, failure_rate_arr)

def average_supervised_frames_test(dataset, save_path = None, label = None):
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
        plt.title("Supervised Frames")
        plt.xlabel("Episode")
        plt.ylabel("Supervised Frames")
        supervised_frames_arr = np.array([x for _, x in sorted(zip(x_arr, supervised_frames_arr))])
        x_arr = np.array(sorted(x_arr))
        N = 9
        supervised_avg = np.convolve(supervised_frames_arr, np.ones(N)/N, mode='valid')
        plt.scatter(x_arr, supervised_frames_arr, s = 10, label = label, alpha = 0.5)
        if(label != None):
            label = label + f" Moving Average (N = {N})"
        plt.plot(x_arr[int(N/2):-int(N/2)], supervised_avg, label = label, linewidth = 2)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"Saved to {save_path}")


    return supervised_frames, (x_arr, supervised_frames_arr)

def cumulative_supervised_frames_test(dataset, save_path = None, label = None):
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

        supervised_frames += (num_of_demo)
        x_arr.append(id_)
        supervised_frames_arr.append((num_of_demo))
    
    print(f"Cumulative supervised frames per episode = {supervised_frames}")
    if(save_path is not None):
        plt.title("Cumulative Supervised Frames")
        plt.xlabel("Episode")
        plt.ylabel("Supervised Frames")
        supervised_frames_arr = np.array([x for _, x in sorted(zip(x_arr, supervised_frames_arr))])
        supervised_frames_cum = np.cumsum(supervised_frames_arr)
        x_arr = np.array(sorted(x_arr))
        plt.plot(x_arr, supervised_frames_cum, label = label, linewidth = 2)
        plt.fill_between(x_arr, supervised_frames_cum, alpha = 0.3)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"Saved to {save_path}")

    return supervised_frames, (x_arr, supervised_frames_cum)


def average_take_over_count_test(dataset):
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
    return take_over_count, (x_arr, take_over_count_arr)

def average_time_test(dataset, save_path = None, label = None):
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
        plt.title("Time to Task Completion")
        plt.xlabel("Episode")
        plt.ylabel("Time to Task Completion (Frames)")
        avg_time_arr = np.array([x for _, x in sorted(zip(x_arr, avg_time_arr))])
        x_arr = np.array(sorted(x_arr))
        N = 9
        avg_time_avg = np.convolve(avg_time_arr, np.ones(N)/N, mode='valid')
        plt.scatter(x_arr, avg_time_arr, s = 10, label = label, alpha = 0.5)
        if(label != None):
            label = label + f" Moving Average (N = {N})"
        plt.plot(x_arr[int(N/2):-int(N/2)], avg_time_avg, label = label, linewidth = 2)
        plt.grid(True)
        plt.legend()
        plt.savefig(save_path)
        print(f"Saved to {save_path}")

    return avg_time,(x_arr, avg_time_arr)

def t_test(dataset, test_type):
    names = dict()
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
        if(name not in names.keys()):
            names[name] = []
        names[name].append((
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
            user_termination_flag,))

    test = getattr(sys.modules[__name__], test_type)

    ts = []
    names_list = list(names.keys())
    for i in range(len(names_list)):
        if(i == len(names_list) - 1):
            continue
        mean_a, arr_a = test(names[names_list[i]])
        for j in range(i+1, len(names_list)):
            mean_b, arr_b = test(names[names_list[j]])

            arr_a = np.array(arr_a)[1]
            arr_b = np.array(arr_b)[1]
            arr_a_sq = arr_a**2
            arr_b_sq = arr_b**2

            n_a = len(arr_a)
            n_b = len(arr_b)

            numerator = (mean_a - mean_b)
            denominator = np.sqrt((((sum(arr_a_sq) - (sum(arr_a)**2)/n_a)+(sum(arr_b_sq) - (sum(arr_b)**2)/n_b))/(n_a + n_b - 2))*(1/n_a+1/n_b))

            t = numerator/denominator
            t_tuple = (names_list[i], names_list[j], t)
            ts.append(t_tuple)

            print("T-Test = ",t_tuple)
    return ts