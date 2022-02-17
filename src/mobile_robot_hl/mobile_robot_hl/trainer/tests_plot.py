import matplotlib.pyplot as plt
import numpy as np

def velocity_similarity_plot(tuple_output, save_path = None, label = None, N = 9):
    x_arr, similarity_arr = tuple_output

    if(save_path == None):
        save_path = 'test.png'

    plt.title("Velocity Similarity")
    plt.xlabel("Episode")
    plt.ylabel("Similarity (Negative Normalized Euclidean Distance)")
    similarity_avg = np.convolve(similarity_arr, np.ones(N)/N, mode='valid')
    plt.scatter(x_arr, similarity_arr, s = 10, label = label, alpha = 0.5)
    if(label != None):
        label = label + f" Moving Average (N = {N})"
    plt.plot(x_arr[int(N/2):-int(N/2)], similarity_avg, label = label, linewidth = 2)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

def average_angular_change_plot(tuple_output, save_path = None, label = None, N = 9):
    x_arr, angular_changes = tuple_output

    if(save_path == None):
        save_path = 'test.png'

    plt.title("Average Absolute Change in Angular Velocity")
    plt.xlabel("Episode")
    plt.ylabel("Average Change (rad/s)")
    angular_changes_avg = np.convolve(angular_changes, np.ones(N)/N, mode='valid')
    plt.scatter(x_arr, angular_changes, s = 10, label = label, alpha = 0.5)
    if(label != None):
        label = label + f" Moving Average (N = {N})"
    plt.plot(x_arr[int(N/2):-int(N/2)], angular_changes_avg, label = label, linewidth = 2)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

def failure_rate_plot(tuple_output, save_path = None, label = None, N = 9):
    x_arr, failure_rate_arr = tuple_output

    if(save_path is None):
        save_path = 'test.png' 

    plt.title("Failures")
    plt.xlabel("Episode")
    plt.ylabel("1 = Failure, 0 = Success")
    failure_avg = np.convolve(failure_rate_arr, np.ones(N)/N, mode='valid')
    plt.scatter(x_arr, failure_rate_arr, s = 10, label = label, alpha = 0.5)
    if(label != None):
        label = label + f" Moving Average (N = {N})"
    plt.plot(x_arr[int(N/2):-int(N/2)], failure_avg, label = label, linewidth = 2)
    plt.grid(True)
    plt.legend(loc = 'lower left')
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

def average_supervised_frames_plot(tuple_output, save_path = None, label = None, N = 9):
    x_arr, supervised_frames_arr = tuple_output

    if(save_path is None):
        save_path = 'test.png'

    plt.title("Supervised Frames")
    plt.xlabel("Episode")
    plt.ylabel("Supervised Frames")
    supervised_avg = np.convolve(supervised_frames_arr, np.ones(N)/N, mode='valid')
    plt.scatter(x_arr, supervised_frames_arr, s = 10, label = label, alpha = 0.5)
    if(label != None):
        label = label + f" Moving Average (N = {N})"
    plt.plot(x_arr[int(N/2):-int(N/2)], supervised_avg, label = label, linewidth = 2)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

def cumulative_supervised_frames_plot(tuple_output, save_path = None, label = None):
    x_arr, supervised_frames_cum = tuple_output

    if(save_path is None):
        save_path = 'test.png'

    plt.title("Cumulative Supervised Frames")
    plt.xlabel("Episode")
    plt.ylabel("Supervised Frames")
    plt.plot(x_arr, supervised_frames_cum, label = label, linewidth = 2)
    plt.fill_between(x_arr, supervised_frames_cum, alpha = 0.3)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")

def average_time_plot(tuple_output, save_path = None, label = None, N = 9):
    x_arr, avg_time_arr = tuple_output

    if(save_path is None):
        save_path = 'test.png'

    plt.title("Time to Task Completion")
    plt.xlabel("Episode")
    plt.ylabel("Time to Task Completion (Frames)")
    avg_time_avg = np.convolve(avg_time_arr, np.ones(N)/N, mode='valid')
    plt.scatter(x_arr, avg_time_arr, s = 10, label = label, alpha = 0.5)
    if(label != None):
        label = label + f" Moving Average (N = {N})"
    plt.plot(x_arr[int(N/2):-int(N/2)], avg_time_avg, label = label, linewidth = 2)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
