import tkinter
from tkinter import StringVar, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import numpy as np
from threading import Thread

from mobile_robot_hl.episode_data import *
from mobile_robot_hl.utils import *
from mobile_robot_hl.gui.utils import *

#==============================================================

class Display(ROSWidget):
    def __init__(self, parent = None, episode_kwargs = dict()):

        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        super().__init__(self.parent)

        self.episode_frame = tkinter.ttk.Frame(self.parent)
        self.episode = Episode(parent = self.episode_frame, **episode_kwargs)

        self.current_frame = tkinter.ttk.Frame(self.parent)
        self.current = Current(parent = self.current_frame)

        self.episode_frame.grid(column = 0, row = 0, sticky = 'nsew')
        self.current_frame.grid(column = 0, row = 1, sticky = 'nsew')

        self.parent.columnconfigure(0, weight = 1)
        self.parent.rowconfigure(0, weight = 1)
        self.parent.rowconfigure(1, weight = 1)

#==============================================================

class Episode(ROSWidget):
    def __init__(self, parent = None, max_linear_vel = 1.0, max_angular_vel = 1.0):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel

        img = np.zeros([360,480,3],dtype=np.uint8)
        img.fill(100)
        self.blank_image = Image.fromarray(img)

        self.image_ = ImageTk.PhotoImage(self.blank_image)
        self.image = tkinter.Label(self.parent,image=self.image_)
        self.slider = tkinter.ttk.Scale(self.parent, from_=0, to_= 1, orient=tkinter.HORIZONTAL, command=self.slider_trigger)

        self.plot_sel_fig = plt.figure(figsize=(3,3),frameon=False)

        self.plot_sel_ax = self.plot_sel_fig.add_axes([0, 0, 1, 1])
        self.plot_sel_ax.spines['left'].set_position('center')
        self.plot_sel_ax.spines['bottom'].set_position('center')
        self.plot_sel_ax.spines['right'].set_color('none')
        self.plot_sel_ax.spines['top'].set_color('none')
        self.plot_sel_ax.margins(x=0.01,y=0.01)
        self.plot_sel_ax.grid(True)
        self.plot_sel_ax.set_ylim([-self.max_linear_vel*1.2,self.max_linear_vel*1.2])
        self.plot_sel_ax.set_xlim([-self.max_angular_vel*1.2,self.max_angular_vel*1.2])
        self.plot_sel_ax.tick_params(labelsize=5)
        self.plot_sel_plot = FigureCanvasTkAgg(self.plot_sel_fig, self.parent)
        self.plot_sel = self.plot_sel_plot.get_tk_widget()

        self.plot_full_frame = tkinter.ttk.Frame(self.parent, padding = "17 17 17 17")
        self.plot_full_fig = plt.figure(figsize=(3,2), frameon=False)

        self.plot_full_ax = self.plot_full_fig.add_axes([0, 0, 1, 1])
        self.plot_full_ax.spines['right'].set_color('none')
        self.plot_full_ax.spines['top'].set_color('none')
        self.plot_full_ax.spines['bottom'].set_position('center')
        self.plot_full_ax.spines['left'].set_color('none')
        self.plot_full_ax.set_ylim([-max(self.max_linear_vel,self.max_angular_vel)*1.2,max(self.max_linear_vel,self.max_angular_vel)*1.2])
        self.plot_full_ax.margins(x=0)
        self.plot_full_ax.grid(True)
        self.plot_full_ax.tick_params(labelsize=5)
        self.plot_full_plot = FigureCanvasTkAgg(self.plot_full_fig, self.plot_full_frame)
        self.plot_full = self.plot_full_plot.get_tk_widget()

        self.image.grid(column = 0, row = 0, sticky = 'nsew')
        self.plot_sel.grid(column=1, row =0, sticky='nsew')
        self.plot_full_frame.grid(column=0, row =1, columnspan=2, sticky='nsew')
        self.plot_full.grid(column = 0, row = 0, sticky = 'nsew')
        self.slider.grid(column=0, row = 2, columnspan = 2, sticky='ew')

        self.parent.columnconfigure(0, weight = 1)
        self.parent.columnconfigure(1, weight = 1)
        self.parent.rowconfigure(0, weight = 1)
        self.parent.rowconfigure(1, weight = 1)

        self.plot_full_frame.columnconfigure(0, weight=1)
        self.plot_full_frame.rowconfigure(0, weight=1)
    
    def slider_trigger(self, val):
        self.slider_value = float(val)

        if(self.ros_node.episode.length() > 0):
            no_of_divs = self.ros_noce.episode.length()
            current_selection = int((self.slider_value/(1/no_of_divs)))
            if(current_selection == no_of_divs):
                current_selection -= 1
            self.ros_node.episode_index = current_selection
            Thread(target=lambda: self.update_image()).start()
            Thread(target=lambda: self.update_plot_sel()).start()
    
    def update_image(self):
        episode_frame = EpisodeData(**self.ros_node.episode.get(self.ros_node.episode_index))
        image = episode_frame.observation.image.get(0)
        image = image.resize((480,360))
        image_current = ImageTk.PhotoImage(image = image)
        if(image_current == self.image_current):
            pass
        else:
            self.image_current = image_current
            self.image.configure(image=self.image_current)

    def update_plot_sel(self):
        episode_frame = EpisodeData(**self.ros_node.episode.get(self.ros_node.episode_index))
        if(episode_frame.action.controller.get(0) == ControllerType.USER):
            desired_vel = episode_frame.action.user.velocity.get(0)
        else:
            desired_vel = episode_frame.action.agent.velocity.get(0)

        try:
            self.current_action_desired_vel.remove()
            self.current_action_user_vel.remove()
            self.current_action_agent_vel.remove()
        except:
            pass

        self.current_action_desired_vel = self.plot_sel_ax.scatter(desired_vel['angular'], desired_vel['linear'],s = 100, c = 'tab:blue', label="desired velocity", alpha=1.0, marker='x')
        self.current_action_user_vel = self.plot_sel_ax.scatter(episode_frame.action.user.velocity.angular.get(0), episode_frame.action.user.velocity.linear.get(0), s = 50, c = 'tab:orange', label="user velocity", alpha=1.0, marker='o')
        self.current_action_agent_vel = self.plot_sel_ax.scatter(episode_frame.action.agent.velocity.angular.get(0), episode_frame.action.agent.velocity.linear, s = 20, c = 'tab:green', label="agent velocity", alpha = 1.0, marker='^')
        self.plot_sel_ax.legend(loc='upper right', prop={'size': 8})
        self.plot_sel_plot.draw_idle()

    def update_plot_full(self):
        episode = self.ros_node.episode
        try:
            self.action_plot_desired_vel_line_linear.pop(0).remove()
            self.action_plot_desired_vel_line_angular.pop(0).remove()
            self.action_plot_user_vel_line_linear.pop(0).remove()
            self.action_plot_user_vel_line_angular.pop(0).remove()
            self.action_plot_agent_vel_line_linear.pop(0).remove()
            self.action_plot_agent_vel_line_angular.pop(0).remove()
        except:
            pass

        if(episode.length() == 0):
            list_range = list(range(1,2))
        else:
            list_range = list(range(1,episode.length()+1))

        desired_vel_linear = [user_vel if(controller == ControllerType.USER) else agent_vel for (user_vel, agent_vel, controller) in zip(episode.action.user.velocity.linear.get(),episode.action.agent.velocity.linear.get() , episode.action.controller.get())]
        desired_vel_angular = [user_vel if(controller == ControllerType.USER) else agent_vel for (user_vel, agent_vel, controller) in zip(episode.action.user.velocity.angular.get(), episode.action.agent.velocity.angular.get(), episode.action.controller.get())]
        desired_vel = {'linear':desired_vel_linear, 'angular': desired_vel_angular}

        self.action_plot_desired_vel_line_linear = self.plot_full_ax.plot(list_range,desired_vel['linear'], c = 'tab:blue', label="desired linear velocity", alpha=1.0, marker='x', linewidth=8, markersize=10)
        self.action_plot_desired_vel_line_angular = self.plot_full_ax.plot(list_range,desired_vel['angular'], c = 'tab:cyan', label="desired angular velocity", alpha=1.0, marker = 'x', linewidth=8, markersize=10)
        self.action_plot_user_vel_line_linear =  self.plot_full_ax.plot(list_range, self.episode.data['action']['user']['velocity']['linear'],c = 'tab:orange', label="user linear velocity", alpha=1.0, marker='o', linewidth = 3, markersize = 7)
        self.action_plot_user_vel_line_angular =  self.plot_full_ax.plot(list_range, self.episode.data['action']['user']['velocity']['angular'],c = 'tab:brown', label="user angular velocity", alpha=1.0, marker='o', linewidth = 3, markersize = 7)
        self.action_plot_agent_vel_line_linear = self.plot_full_ax.plot(list_range, self.episode.data['action']['agent']['velocity']['linear'],c = 'tab:green', label="agent linear velocity", alpha = 1.0, marker='^', linewidth = 1, markersize = 4)
        self.action_plot_agent_vel_line_angular = self.plot_full_ax.plot(list_range, self.episode.data['action']['agent']['velocity']['angular'],c = 'olive', label="agent angular velocity", alpha = 1.0, marker='^', linewidth = 1, markersize = 4)

        self.plot_full_ax.set_xlim([0.5,max(episode.length(),1)+0.5])
        self.plot_full_ax.legend(loc='lower left', prop={'size': 8})

        self.plot_full_plot.draw_idle()

class Current(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

        img = np.zeros([360,480,3],dtype=np.uint8)
        img.fill(100)
        self.blank_image = Image.fromarray(img)

        self.image_ = ImageTk.PhotoImage(self.blank_image)
        self.image = tkinter.Label(self.parent,image=self.image_)

        self.info_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN, padding = "10 10 10 10")
        self.info = Info(self.info_frame)

        self.image.grid(column = 0, row = 0, sticky = 'nsew')
        self.info_frame.grid(column = 1, row = 0, sticky = 'nsew')

        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, weight=1)
        self.parent.rowconfigure(0, weight=1)
    
    def update_image(self):
        image = self.ros_node.image_raw
        image = image.resize((480,360))
        image_current = ImageTk.PhotoImage(image = image)
        if(image_current == self.image_current):
            pass
        else:
            self.image_current = image_current
            self.image.configure(image=self.image_current)

#==============================================================

class Info(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

        self.title = tkinter.ttk.Label(self.parent, text="Information")

        self.episode_frame = tkinter.ttk.Frame(self.parent, padding = "10 20 0 0")
        self.episode_controller = tkinter.ttk.Label(self.episode_frame, text="Controller: None")
        self.episode_type = tkinter.ttk.Label(self.episode_frame, text="Episode Type: None")
        self.episode_name = tkinter.ttk.Label(self.episode_frame, text="Episode Name: None")
        self.episode_id = tkinter.ttk.Label(self.episode_frame, text="Episode ID: None")

        self.model_frame = tkinter.ttk.Frame(self.parent, padding = "10 20 0 0")
        self.model_name = tkinter.ttk.Label(self.model_frame, text="Model Name: None")
        self.model_id = tkinter.ttk.Label(self.model_frame, text="Model ID: None")

        self.agent_frame = tkinter.ttk.Frame(self.parent, padding = "10 20 0 0")
        self.agent_velocity = tkinter.ttk.Label(self.agent_frame, text="Agent Velocity:\n    Linear: 0.0 m/s\n    Angular: 0.0 rad/s")
        self.agent_termination_flag = tkinter.ttk.Label(self.agent_frame, text="Agent Termination Flag: False")

        self.user_frame = tkinter.ttk.Frame(self.parent, padding = "10 20 0 0")
        self.user_velocity = tkinter.ttk.Label(self.user_frame, text="User Velocity:\n    Linear: 0.0 m/s\n    Angular: 0.0 rad/s")
        self.user_termination_flag = tkinter.ttk.Label(self.user_frame, text="User Termination Flag: False")

        self.title.grid(column=0, row=0)

        self.episode_frame.grid(column = 0, row = 1, sticky='w')
        self.episode_controller.grid(column = 0, row = 0, sticky = 'w')
        self.episode_type.grid(column = 0, row = 1, sticky = 'w')
        self.episode_name.grid(column = 0, row = 2, sticky = 'w')
        self.episode_id.grid(column = 0, row = 3, sticky = 'w')

        self.model_frame.grid(column = 0, row = 2, sticky='w')
        self.model_name.grid(column = 0, row = 0, sticky='w')
        self.model_id.grid(column = 0, row = 1, sticky='w')

        self.agent_frame.grid(column = 0, row = 3, sticky='w')
        self.agent_velocity.grid(column = 0, row = 0, sticky='w')
        self.agent_termination_flag.grid(column = 0, row = 1, sticky='w')

        self.user_frame.grid(column = 0, row = 4, sticky='w')
        self.user_velocity.grid(column = 0, row = 0, sticky='w')
        self.user_termination_flag.grid(column = 0, row = 1, sticky='w')

        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(5, weight=1)
    
    def update_info(self):
        episode_frame = EpisodeData(**self.ros_node.episode.get(self.ros_node.episode_index))
        episode_type = self.ros_node.episode_type.name
        episode_name = self.ros_node.episode_name
        episode_id = self.ros_node.episode_id
        model_name = self.ros_node.model_name
        model_id = self.ros_node.model_id

        user_linear_vel = episode_frame.action.user.velocity.linear.get(0)
        user_angular_vel = episode_frame.action.user.velocity.angular.get(0)
        if(type(user_linear_vel) == float and type(user_angular_vel) == float):
            self.user_velocity.configure(text=f"User Velocity: \n    Linear: {user_linear_vel:.2f} m/s\n    Angular: {user_angular_vel:.2f} rad/s")

        agent_linear_vel = episode_frame.action.agent.velocity.linear.get(0)
        agent_angular_vel = episode_frame.action.agent.velocity.angular.get(0)
        if(type(agent_linear_vel) == float and type(agent_angular_vel) == float):
            self.agent_velocity.configure(text=f"Agent Velocity: \n    Linear: {agent_linear_vel:.2f} m/s\n    Angular: {agent_angular_vel:.2f} rad/s")
        
        user_termination = episode_frame.action.user.termination_flag.get(0)
        if type(user_termination) == bool:
            self.user_termination_flag.configure(text=f"User Termination Flag: {user_termination}")

        agent_termination = episode_frame.action.agent.termination_flag.get(0)
        if type(agent_termination) == bool:
            self.agent_termination_flag.configure(text=f"Agent Termination Flag: {agent_termination}")

        controller = episode_frame.action.controller.get(0)
        if controller is not None:
            self.episode_controller.configure(text=f"Controller: {controller.name}")

        if type(episode_type) == str:
            self.episode_type.configure(text=f"Episode Type: {episode_type}")
        
        if type(episode_name) == str:
            self.episode_name.configure(text=f"Episode Name: {episode_name}")

        if type(episode_id) == str:
            self.episode_id.configure(text=f"Episode ID: {episode_id}")

        if type(model_name) == str:
            self.episode_id.configure(text=f"Model Name: {model_name}")

        if type(model_id) == str:
            self.episode_id.configure(text=f"Model ID: {model_id}")