import matplotlib.pyplot as plt
import numpy as np
import sys
from statsmodels.stats.power import TTestIndPower

from .utils import *
from .tests_plot import *

def velocity_similarity_test(dataset, plot_kwargs = None):
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

        user_linear_vel = user_linear_vel[demo_flag[task_start_index:] == 0.0]
        user_angular_vel = user_angular_vel[demo_flag[task_start_index:] == 0.0]
        agent_linear_vel = agent_linear_vel[demo_flag[task_start_index:] == 0.0]
        agent_angular_vel =  agent_angular_vel[demo_flag[task_start_index:] == 0.0]

        max_tensor = torch.tensor([2*MAX_LINEAR_VELOCITY, 2*MAX_ANGULAR_VELOCITY])
        user_vel = torch.cat((user_linear_vel.unsqueeze(1), user_angular_vel.unsqueeze(1)), dim = 1)/max_tensor
        agent_vel = torch.cat((agent_linear_vel.unsqueeze(1), agent_angular_vel.unsqueeze(1)), dim = 1)/max_tensor
        sim = torch.mean(torch.sum((user_vel - agent_vel)**2, dim = 1))
        rmse_sim = -sim**0.5
        
        similarity += rmse_sim/len(dataset)
        x_arr.append(id_)
        similarity_arr.append(rmse_sim)

    similarity_arr = np.array([x.item() for _, x in sorted(zip(x_arr, similarity_arr))])
    x_arr = np.array(sorted(x_arr))

    print(f"Similarity = {similarity}")

    tuple_output = (x_arr, similarity_arr)
    if(plot_kwargs is not None):
        velocity_similarity_plot(tuple_output, **plot_kwargs)

    return similarity, tuple_output

def failure_rate_test(dataset, plot_kwargs = None):
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

    failure_rate_arr = np.array([x for _, x in sorted(zip(x_arr, failure_rate_arr))])
    x_arr = np.array(sorted(x_arr))
    tuple_output = (x_arr, failure_rate_arr)
    print(f"Failure Rate = {failure_rate}")

    if(plot_kwargs is not None):
        failure_rate_plot(tuple_output, **plot_kwargs)

    return failure_rate, tuple_output

def average_supervised_frames_test(dataset, plot_kwargs = None):
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
    
    supervised_frames_arr = np.array([x for _, x in sorted(zip(x_arr, supervised_frames_arr))])
    x_arr = np.array(sorted(x_arr))
    tuple_output = (x_arr, supervised_frames_arr)

    print(f"Average supervised frames per episode = {supervised_frames}")
    if(plot_kwargs is not None):
        average_supervised_frames_plot(tuple_output, **plot_kwargs)

    return supervised_frames, tuple_output

def cumulative_supervised_frames_test(dataset, plot_kwargs = None):
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

    supervised_frames_arr = np.array([x for _, x in sorted(zip(x_arr, supervised_frames_arr))])
    supervised_frames_cum = np.cumsum(supervised_frames_arr)
    x_arr = np.array(sorted(x_arr))
    tuple_output = (x_arr, supervised_frames_cum)

    print(f"Cumulative supervised frames per episode = {supervised_frames}")
    if(plot_kwargs is not None):
        cumulative_supervised_frames_plot(tuple_output, **plot_kwargs)

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

def average_time_test(dataset, plot_kwargs = None):
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

        t = (len(frame_no)-frame_no[task_start_index-1])
        avg_time += t/len(dataset)
        x_arr.append(id_)
        avg_time_arr.append(t)
    
    avg_time_arr = np.array([x for _, x in sorted(zip(x_arr, avg_time_arr))])
    x_arr = np.array(sorted(x_arr))

    tuple_output = (x_arr, avg_time_arr)
    print(f"Average Time = {avg_time}")

    if(plot_kwargs is not None):
        average_time_plot(tuple_output, **plot_kwargs)

    return avg_time,(x_arr, avg_time_arr)

def angular_acc_test(dataset, plot_kwargs = None):
    angular_change = 0.0
    angular_changes = []
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

        agent_angular_vel = agent_angular_vel[task_start_index:]
        next_angular_vel = agent_angular_vel[1:].numpy()
        current_angular_vel = agent_angular_vel[:-1].numpy()

        angular_change_ = np.mean(np.absolute(next_angular_vel - current_angular_vel)[demo_flag[1+task_start_index:] == 0])

        angular_change += angular_change_/len(dataset)
        x_arr.append(id_)
        angular_changes.append(angular_change_)

    angular_changes = np.array([x.item() for _, x in sorted(zip(x_arr, angular_changes))])
    x_arr = np.array(sorted(x_arr))

    print(f"Average Angular Velocity Difference = {angular_change}")

    tuple_output = (x_arr, angular_changes)
    if(plot_kwargs is not None):
        average_angular_change_plot(tuple_output, **plot_kwargs)

    return angular_change, tuple_output

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

def power_analysis_test(dataset, test_type, power = 0.8, alpha = 0.05):
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

    sample_sizes = []
    names_list = list(names.keys())
    for i in range(len(names_list)):
        if(i == len(names_list) - 1):
            continue
        mean_a, arr_a = test(names[names_list[i]])
        for j in range(i+1, len(names_list)):
            mean_b, arr_b = test(names[names_list[j]])

            arr_a = np.array(arr_a)[1]
            arr_b = np.array(arr_b)[1]
            std_a = np.std(arr_a)
            std_b = np.std(arr_b)
            n_a = len(arr_a)
            n_b = len(arr_b)

            effect = (abs(mean_a-mean_b)/((std_a+std_b)/2)).item()
            ratio = min(n_a, n_b)/max(n_a, n_b)
            print("Effect",effect)
            print("Ratio",ratio)
            print("Power",power)
            print("Alpha",alpha)

            analysis = TTestIndPower()
            sample_size = analysis.solve_power(effect, power=power, nobs1=None, ratio=ratio, alpha=alpha)

            sample_size_tuple = (names_list[i], names_list[j], sample_size)
            sample_sizes.append(sample_size_tuple)

            print("Sample Size = ", sample_size_tuple)
    return sample_sizes

def aggregator(dataset, test_type, group_name, plot_kwargs = None):
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

    aggregated_outputs = []

    for group in group_name.keys():
        group_arr = []
        for name_ in group_name[group]:
            mean, tuple_arr = test(names[name_])
            x_arr = tuple_arr[0]
            group_arr.append(tuple_arr[1])
        mean_arr = np.mean(np.stack(group_arr), axis = 0)
        aggregated_outputs.append((group, mean_arr))

        if(plot_kwargs is not None):
            plotter = getattr(sys.modules[__name__], test_type[:-5]+"_plot")
            tuple_output = (x_arr, mean_arr)
            plotter(tuple_output, label = group, **plot_kwargs)

    return aggregated_outputs