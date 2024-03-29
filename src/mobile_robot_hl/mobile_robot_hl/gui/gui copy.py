import tkinter
from enum import Enum
from tkinter import StringVar, ttk
from PIL import Image, ImageTk
from numpy.core.fromnumeric import put
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
from threading import Thread

from .episode_data import *
from .utils import *
from .model.utils import *

class SupervisorGUI():
    def __init__(self, ros_node = None):

        self.ros_node = ros_node

        self.window = tkinter.Tk()
        self.window.title("Supervisor GUI")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

        self.state = SupervisorState.STANDBY
        self.selected_demo = None
        self.selected_model = dict(name = None, id = None)
        self.episode = EpisodeData(data=None)
        self.slider_value = 0
        self.selection = InformationType.NONE
        self.current_action = None # dict(user, agent, controller)
        self.episode_index = 0
        self.image_episode_temp = None

        sns.set('notebook')
        sns.set_style("white")
        self.setup_extras()
        self.setup_mainframe()
        self.setup_display_frame()
        self.setup_control_frame()

        self.model_update_name_entry()

        self.action_plot_timer = Thread(target=self.update_action_plot_trigger)
        self.current_action_plot_timer = Thread(target=self.update_current_action_plot_trigger)
        self.update_image_current_timer = Thread(target=self.update_image_current)

        self.action_plot_timer.start()
        self.current_action_plot_timer.start()
        self.update_image_current_timer.start()

    def setup_extras(self):
        img = np.zeros([360,480,3],dtype=np.uint8)
        img.fill(100)
        self.blank_image = Image.fromarray(img)

    def setup_mainframe(self):
        # setup frames
        self.mainframe = tkinter.ttk.Frame(self.window, padding = "3 3 3 3")
        self.display_frame = tkinter.ttk.Frame(self.mainframe,  padding = "10 10 10 10")
        self.control_frame = tkinter.ttk.Frame(self.mainframe, borderwidth =2,relief=tkinter.RIDGE, padding = "10 10 10 10")

        self.mainframe.grid(sticky='nsew')
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        self.display_frame.grid(column=0, row=0, sticky='nsew')
        self.control_frame.grid(column=1, row=0, sticky='nsew')
    
    def setup_display_frame(self):
        # setup display_frame section
        self.image_episode = ImageTk.PhotoImage(self.blank_image)
        self.image_current = ImageTk.PhotoImage(self.blank_image)
        self.image_episode_label = tkinter.Label(self.display_frame,image=self.image_episode)
        self.image_current_label = tkinter.Label(self.display_frame,image=self.image_current)
        self.image_slider = tkinter.ttk.Scale(self.display_frame, from_=0, to_= 1, orient=tkinter.HORIZONTAL, command=self.slider_trigger)
        self.action_plot_dummy_frame = tkinter.ttk.Frame(self.display_frame, padding = "17 17 17 17")

        self.info_frame = tkinter.ttk.Frame(self.display_frame, borderwidth=2, relief=tkinter.SUNKEN, padding = "10 10 10 10")
        self.info_frame_title = tkinter.ttk.Label(self.info_frame, text="Information")

        self.info_control_frame = tkinter.Frame(self.info_frame)
        self.info_controller = tkinter.ttk.Label(self.info_control_frame, text="Controller: NONE")

        self.info_user_frame = tkinter.Frame(self.info_control_frame)
        self.info_agent_frame = tkinter.Frame(self.info_control_frame)

        #self.info_desired_vel = tkinter.ttk.Label(self.info_frame, text="Desired Velocity: \n\tLinear: 0.0 m/s\n\tAngular: 0.0 rad/s")
        self.info_user_vel = tkinter.ttk.Label(self.info_user_frame, text="User Velocity:\n    Linear: 0.0 m/s\n    Angular: 0.0 rad/s")
        self.info_user_termination_flag = tkinter.ttk.Label(self.info_user_frame, text="User Termination Flag: False")

        self.info_agent_vel = tkinter.ttk.Label(self.info_agent_frame, text="Agent Velocity:\n    Linear: 0.0 m/s\n    Angular: 0.0 rad/s")
        self.info_agent_termination_flag = tkinter.ttk.Label(self.info_agent_frame, text="Agent Termination Flag: False")

        self.info_current_demo = tkinter.ttk.Label(self.info_frame, text="Selected Demonstration: None")

        self.info_model = tkinter.ttk.Label(self.info_frame, text="Selected Model: None")

        self.current_action_fig = plt.figure(figsize=(3,3),frameon=False)

        self.current_action_ax = self.current_action_fig.add_axes([0, 0, 1, 1])
        self.current_action_ax.spines['left'].set_position('center')
        self.current_action_ax.spines['bottom'].set_position('center')
        self.current_action_ax.spines['right'].set_color('none')
        self.current_action_ax.spines['top'].set_color('none')
        self.current_action_ax.margins(x=0.01,y=0.01)
        self.current_action_ax.grid(True)
        self.current_action_ax.set_ylim([-self.ros_node.max_linear_velocity*1.2,self.ros_node.max_linear_velocity*1.2])
        self.current_action_ax.set_xlim([-self.ros_node.max_angular_velocity*1.2,self.ros_node.max_angular_velocity*1.2])
        self.current_action_ax.tick_params(labelsize=5)

        self.current_action_plot = FigureCanvasTkAgg(self.current_action_fig, self.display_frame)

        self.action_plot_fig = plt.figure(figsize=(3,2), frameon=False)
        self.action_plot_ax = self.action_plot_fig.add_axes([0, 0, 1, 1])
        
        self.action_plot_ax.spines['right'].set_color('none')
        self.action_plot_ax.spines['top'].set_color('none')
        self.action_plot_ax.spines['bottom'].set_position('center')
        self.action_plot_ax.spines['left'].set_color('none')
        self.action_plot_ax.set_ylim([-max(self.ros_node.max_linear_velocity,self.ros_node.max_angular_velocity)*1.2,max(self.ros_node.max_linear_velocity,self.ros_node.max_angular_velocity)*1.2])
        self.action_plot_ax.margins(x=0)
        self.action_plot_ax.grid(True)
        self.action_plot_ax.tick_params(labelsize=5)

        self.action_plot_plot = FigureCanvasTkAgg(self.action_plot_fig, self.action_plot_dummy_frame)

        self.image_episode_label.grid(column=0, row=0)
        self.current_action_plot.get_tk_widget().grid(column=1, row =0, sticky='nsew')
        self.action_plot_dummy_frame.grid(column=0, row =1, columnspan=2, sticky='nsew')
        self.action_plot_plot.get_tk_widget().grid(column=0, row =0, sticky='ew')
        self.image_current_label.grid(column=0, row=4)
        self.image_slider.grid(column=0, row = 2, columnspan = 2, sticky='ew')

        self.info_frame.grid(column=1, row=3, rowspan=2,sticky='nsew')
        self.info_frame_title.grid(column=0, row=0)
        self.info_control_frame.grid(column = 0, row = 1, sticky='w')
        self.info_current_demo.grid(column = 0, row = 2, sticky='w')
        self.info_model.grid(column = 0, row = 3, sticky='w')
        self.info_controller.grid(column = 0, row = 0, sticky='w')
        self.info_user_frame.grid(column=0, row = 1, sticky='w')
        self.info_agent_frame.grid(column=0, row = 2, sticky='w')
        self.info_user_vel.grid(column = 0, row = 0, sticky = 'w')
        self.info_user_termination_flag.grid(column = 0, row = 1, sticky='w')
        self.info_agent_vel.grid(column = 0, row =0, sticky='w')
        self.info_agent_termination_flag.grid(column = 0, row = 1, sticky='w')

        self.action_plot_dummy_frame.columnconfigure(0, weight=1)

        self.info_frame.rowconfigure(0, weight=1)
        self.info_frame.columnconfigure(0, weight=1)
        self.info_frame.rowconfigure(7, weight=1)

        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.rowconfigure(4, weight=1)
        self.display_frame.columnconfigure(0, weight=1)
        self.display_frame.columnconfigure(1, weight=1)

    def setup_control_frame(self):
        # setup control_frame section
        self.automatic_control_frame = tkinter.ttk.Frame(self.control_frame, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.demo_control_frame = tkinter.ttk.Frame(self.control_frame, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.model_control_frame = tkinter.ttk.Frame(self.control_frame, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.task_queue_frame = tkinter.ttk.Frame(self.control_frame, borderwidth=2, relief=tkinter.SUNKEN, padding= "10 10 10 10")

        self.automatic_control_button_frame = tkinter.ttk.Frame(self.automatic_control_frame)
        self.automatic_start_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="start", command = self.agent_start_button_trigger)
        self.automatic_stop_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="stop", command = self.agent_stop_button_trigger)
        self.automatic_take_over_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="take over", command = self.agent_take_over_button_trigger)
        self.automatic_save_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="save", command = self.agent_save_button_trigger)
        self.automatic_save_button.state(['disabled'])
        self.automatic_stop_button.state(['disabled'])
        self.automatic_take_over_button.state(['disabled'])
        self.demo_control_button_frame = tkinter.ttk.Frame(self.demo_control_frame)
        self.demo_start_button = tkinter.ttk.Button(self.demo_control_button_frame, text="start", command = self.demo_start_button_trigger)
        self.demo_stop_button = tkinter.ttk.Button(self.demo_control_button_frame, text="stop", command = self.demo_stop_button_trigger)
        self.demo_stop_button.state(['disabled'])
        self.demo_save_button = tkinter.ttk.Button(self.demo_control_button_frame, text="save", command = self.demo_save_button_trigger)
        self.demo_save_button.state(['disabled'])
        self.demo_name_entry = tkinter.ttk.Combobox(self.demo_control_frame)

        self.model_control_button_frame = tkinter.ttk.Frame(self.model_control_frame)
        self.model_select_button = tkinter.ttk.Button(self.model_control_button_frame, text="select", command = self.model_select_button_trigger)
        self.model_name_entry = tkinter.ttk.Combobox(self.model_control_button_frame)
        self.model_name_entry.bind('<<ComboboxSelected>>', self.select_model_trigger)
        self.model_id_entry = tkinter.ttk.Combobox(self.model_control_button_frame)

        self.saved_task_episode_name_list = tkinter.Listbox(self.task_queue_frame)
        self.saved_demo_name_list = tkinter.Listbox(self.task_queue_frame)
        self.saved_demo_id_list = tkinter.Listbox(self.task_queue_frame)
        self.saved_demo_name_list.bind('<<ListboxSelect>>', self.saved_demo_name_list_trigger)
        self.saved_task_episode_name_list.bind('<<ListboxSelect>>', self.saved_task_episode_name_list_trigger)
        self.saved_demo_id_list.bind('<<ListboxSelect>>', self.saved_demo_id_list_trigger)
        self.task_add_button = tkinter.ttk.Button(self.task_queue_frame, text=">>", command= self.add_demo_trigger)
        self.task_remove_button = tkinter.ttk.Button(self.task_queue_frame, text="<<", command = self.remove_demo_trigger)
        self.queued_demo = StringVar(value=[])
        self.queued_demo_list = tkinter.Listbox(self.task_queue_frame, listvariable=self.queued_demo)
        self.demo_label = tkinter.ttk.Label(self.demo_control_frame, text="Demonstration Control Panel")
        self.automatic_label = tkinter.ttk.Label(self.automatic_control_frame, text="Automatic Control Panel")
        self.model_label = tkinter.ttk.Label(self.model_control_frame, text="Model Control Panel")
        self.saved_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Available Demo")
        self.saved_task_episode_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Available Task Episode")
        self.queued_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Queued Demo")

        self.automatic_control_frame.grid(column=0, row=0, columnspan = 2, sticky='nsew')
        self.automatic_label.grid(column=0, row=0)
        self.automatic_control_button_frame.grid(column=0, row = 2)
        self.automatic_start_button.grid(column=1, row = 0)
        self.automatic_stop_button.grid(column=0, row = 0)
        self.automatic_take_over_button.grid(column=2, row = 0)
        self.automatic_save_button.grid(column=3, row = 0)
        self.automatic_control_frame.rowconfigure(0, weight=1)
        self.automatic_control_frame.columnconfigure(0, weight=1)

        self.demo_control_frame.grid(column=0, row=1, sticky='nsew')
        self.demo_label.grid(column=0, row=0)
        self.demo_name_entry.grid(column = 0, row = 1)
        self.demo_control_button_frame.grid(column=0, row =2)
        self.demo_start_button.grid(column=0, row = 0)
        self.demo_stop_button.grid(column=1, row = 0)
        self.demo_save_button.grid(column=2, row = 0)
        self.demo_control_frame.rowconfigure(0, weight=1)
        self.demo_control_frame.columnconfigure(0, weight=1)

        self.model_control_frame.grid(column=1, row=1, sticky='nsew')
        self.model_control_button_frame.grid(column = 0, row = 1)
        self.model_label.grid(column = 0, row = 0)
        self.model_name_entry.grid(column = 0, row = 0)
        self.model_id_entry.grid(column = 1, row = 0)
        self.model_select_button.grid(column=0, row=1, columnspan = 2)

        self.model_control_frame.rowconfigure(0, weight=1)
        self.model_control_frame.columnconfigure(0, weight=1)

        self.task_queue_frame.grid(column=0, row=2, columnspan = 2, sticky='nsew')
        self.saved_demo_list_label.grid(column = 0,  row = 0)
        self.saved_task_episode_list_label.grid(column = 0, row = 3)
        self.queued_demo_list_label.grid(column = 3, row = 0)
        self.saved_demo_name_list.grid(column=0, row=1, rowspan=2, sticky='nsew')
        self.saved_demo_id_list.grid(column=1, row=1, rowspan=5, sticky='nsew')
        self.saved_task_episode_name_list.grid(column=0, row=4, rowspan=2, sticky='nsew')
        self.queued_demo_list.grid(column=3, row=1, rowspan=5, sticky = 'nsew')
        self.task_add_button.grid(column=2, row=2)
        self.task_remove_button.grid(column=2, row=4)

        self.task_queue_frame.rowconfigure(2, weight=1)
        self.task_queue_frame.rowconfigure(4, weight=1)
        self.task_queue_frame.columnconfigure(0, weight=1)
        self.task_queue_frame.columnconfigure(2, weight=1)

        self.control_frame.rowconfigure(0, weight=1)
        self.control_frame.rowconfigure(1, weight=1)
        self.control_frame.rowconfigure(2, weight=2)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
    
    def update_image_current(self):
        while True:
            if(self.ros_node.image_raw == None):
                pass
            else:
                image = self.ros_node.image_raw
                image = image.resize((480,360))
                image_current = ImageTk.PhotoImage(image = image)
                if(image_current == self.image_current):
                    pass
                else:
                    self.image_current = image_current
                    self.image_current_label.configure(image=self.image_current)
            time.sleep(0.2)
    
    def update_current_action_plot_trigger(self):
        while True:
            if(self.episode.data_empty == False):
                current_episode_frame = self.episode.get_data(index=self.episode_index)
                if(current_episode_frame['action']['controller'] == ControllerType.USER):
                    desired_vel = current_episode_frame['action']['user']['velocity']
                else:
                    desired_vel = current_episode_frame['action']['agent']['velocity']

                try:
                    self.current_action_desired_vel.remove()
                    self.current_action_user_vel.remove()
                    self.current_action_agent_vel.remove()
                except:
                    pass

                try:
                    self.current_action_desired_vel = self.current_action_ax.scatter(desired_vel['angular'], desired_vel['linear'],s = 100, c = 'tab:blue', label="desired velocity", alpha=1.0, marker='x')
                    self.current_action_user_vel = self.current_action_ax.scatter(current_episode_frame['action']['user']['velocity']['angular'], current_episode_frame['action']['user']['velocity']['linear'], s = 50, c = 'tab:orange', label="user velocity", alpha=1.0, marker='o')
                    self.current_action_agent_vel = self.current_action_ax.scatter(current_episode_frame['action']['agent']['velocity']['angular'], current_episode_frame['action']['agent']['velocity']['linear'], s = 20, c = 'tab:green', label="agent velocity", alpha = 1.0, marker='^')
                except:
                    pass

                self.current_action_ax.legend(loc='upper right', prop={'size': 8})
                try:
                    self.current_action_plot.draw_idle()
                except:
                    pass
            time.sleep(1.0)
    
    def update_info(self, user_vel=None, agent_vel=None, agent_termination=None, user_termination=None, selected_demo=None, controller=None, selected_model=None):
        # TODO: update this so that it just looks at the current episode frame
        if type(user_vel) == dict:
            self.info_user_vel.configure(text=f"User Velocity: \n    Linear: {user_vel['linear']:.2f} m/s\n    Angular: {user_vel['angular']:.2f} rad/s")
        if type(agent_vel) == dict:
            self.info_agent_vel.configure(text=f"Agent Velocity: \n    Linear: {agent_vel['linear']:.2f} m/s\n    Angular: {agent_vel['angular']:.2f} rad/s")
        if type(agent_termination) == bool:
            self.info_agent_termination_flag.configure(text=f"Agent Termination Flag: {agent_termination}")
        if type(user_termination) == bool:
            self.info_user_termination_flag.configure(text=f"User Termination Flag: {user_termination}")
        if type(selected_demo) == str:
            self.info_current_demo.configure(text=f"Selected Demonstration: {selected_demo}")
        if controller is not None:
            self.info_controller.configure(text=f"Controller: {controller.name}")
        if type(selected_model) == str:
            self.info_model.configure(text=f"Selected Model: {selected_model}")

    def update_action_plot_trigger(self):
        while True:
            try:
                self.action_plot_desired_vel_line_linear.pop(0).remove()
                self.action_plot_desired_vel_line_angular.pop(0).remove()
                self.action_plot_user_vel_line_linear.pop(0).remove()
                self.action_plot_user_vel_line_angular.pop(0).remove()
                self.action_plot_agent_vel_line_linear.pop(0).remove()
                self.action_plot_agent_vel_line_angular.pop(0).remove()
            except:
                pass

            if(self.episode.data_empty == True):
                list_range = list(range(1,2))
            else:
                try:
                    list_range = list(range(1,self.episode.get_episode_length()+1))
                except:
                    continue

            desired_vel_linear = [user_vel if(controller == ControllerType.USER) else agent_vel for (user_vel, agent_vel, controller) in zip(self.episode.data['action']['user']['velocity']['linear'] ,self.episode.data['action']['agent']['velocity']['linear'], self.episode.data['action']['controller'])]
            desired_vel_angular = [user_vel if(controller == ControllerType.USER) else agent_vel for (user_vel, agent_vel, controller) in zip(self.episode.data['action']['user']['velocity']['angular'] ,self.episode.data['action']['agent']['velocity']['angular'], self.episode.data['action']['controller'])]
            desired_vel = {'linear':desired_vel_linear, 'angular': desired_vel_angular}

            try:
                self.action_plot_desired_vel_line_linear = self.action_plot_ax.plot(list_range,desired_vel['linear'], c = 'tab:blue', label="desired linear velocity", alpha=1.0, marker='x', linewidth=8, markersize=10)
                self.action_plot_desired_vel_line_angular = self.action_plot_ax.plot(list_range,desired_vel['angular'], c = 'tab:cyan', label="desired angular velocity", alpha=1.0, marker = 'x', linewidth=8, markersize=10)
                self.action_plot_user_vel_line_linear =  self.action_plot_ax.plot(list_range, self.episode.data['action']['user']['velocity']['linear'],c = 'tab:orange', label="user linear velocity", alpha=1.0, marker='o', linewidth = 3, markersize = 7)
                self.action_plot_user_vel_line_angular =  self.action_plot_ax.plot(list_range, self.episode.data['action']['user']['velocity']['angular'],c = 'tab:brown', label="user angular velocity", alpha=1.0, marker='o', linewidth = 3, markersize = 7)
                self.action_plot_agent_vel_line_linear = self.action_plot_ax.plot(list_range, self.episode.data['action']['agent']['velocity']['linear'],c = 'tab:green', label="agent linear velocity", alpha = 1.0, marker='^', linewidth = 1, markersize = 4)
                self.action_plot_agent_vel_line_angular = self.action_plot_ax.plot(list_range, self.episode.data['action']['agent']['velocity']['angular'],c = 'olive', label="agent angular velocity", alpha = 1.0, marker='^', linewidth = 1, markersize = 4)
            except:
                continue

            self.action_plot_ax.set_xlim([0.5,max(self.episode.get_episode_length(),1)+0.5])

            self.action_plot_ax.legend(loc='lower left', prop={'size': 8})
            try:
                self.action_plot_plot.draw_idle()
            except:
                pass
            time.sleep(3.0)
    
    def agent_start_button_trigger(self):
        if self.selected_demo == None or self.selected_demo == "":
            print("[INFO] No demonstration selected")
            return
        if(self.state==SupervisorState.TASK_PAUSED or self.state == SupervisorState.STANDBY or self.state == SupervisorState.TASK_TAKE_OVER):
            self.automatic_take_over_button.configure(text="take over")
            self.automatic_start_button.configure(text="pause")
            self.automatic_take_over_button.state(['!disabled'])
            self.automatic_stop_button.state(['!disabled'])
            self.automatic_save_button.state(['!disabled'])
            self.demo_start_button.state(['disabled'])
            self.demo_save_button.state(['disabled'])
            response = self.ros_node.call_service('agent/start', self.selected_demo)
            if(response.success == False):
                if(self.state == SupervisorState.STANDBY):
                    self.state = SupervisorState.STANDBY
                    self.automatic_start_button.configure(text="start")
                    self.automatic_take_over_button.configure(text="take over")
                    self.automatic_take_over_button.state(['disabled'])
                    self.automatic_stop_button.state(['disabled'])
                    self.automatic_save_button.state(['disabled'])
                    self.demo_start_button.state(['!disabled'])
                    return
            if(self.state == SupervisorState.STANDBY):
                self.set_episode(self.ros_node.demo_handler.get(self.selected_demo.split('.')[0],self.selected_demo.split('.')[1]))
            self.state = SupervisorState.TASK_RUNNING

            print("[INFO] Automatic control started")
        elif(self.state == SupervisorState.TASK_RUNNING):
            self.state = SupervisorState.TASK_PAUSED
            self.automatic_start_button.configure(text="start")
            self.ros_node.call_service('agent/pause')
            print("[INFO] Automatic control paused")

        self.ros_node.update_state(self.state)
    
    def agent_stop_button_trigger(self):
        self.state = SupervisorState.STANDBY
        self.automatic_start_button.configure(text="start")
        self.automatic_take_over_button.configure(text="take over")
        self.automatic_take_over_button.state(['disabled'])
        self.automatic_stop_button.state(['disabled'])
        self.automatic_save_button.state(['disabled'])
        self.demo_start_button.state(['!disabled'])
        try:
            self.ros_node.call_service('agent/stop')
            self.ros_node.update_state(self.state)
            self.ros_node.reset_episode()
        except:
            pass
        print("[INFO] Automatic control stopped")

    def agent_take_over_button_trigger(self):
        if(self.state == SupervisorState.TASK_RUNNING or self.state == SupervisorState.TASK_PAUSED):
            self.automatic_take_over_button.configure(text="pause")
            self.state = SupervisorState.TASK_TAKE_OVER
            self.automatic_start_button.configure(text="start")
            try:
                self.ros_node.call_service('agent/start', self.selected_demo)
                self.ros_node.update_state(self.state)
            except:
                pass
            print("[INFO] Supervisor take-over")
        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            self.state = SupervisorState.TASK_PAUSED
            self.automatic_take_over_button.configure(text="take over")
            print("[INFO] Supervisor take-over paused")
            try:
                self.ros_node.call_service('agent/pause')
                self.ros_node.update_state(self.state)
            except:
                pass

    def agent_save_button_trigger(self):
        self.state = SupervisorState.TASK_PAUSED
        self.automatic_start_button.configure(text="start")
        self.automatic_take_over_button.configure(text="take over")
        self.ros_node.update_state(self.state)
        print(f"[INFO] Saved task episode to {self.selected_demo}")
        Thread(target=lambda: self.agent_save_asynchronous()).start()
    
    def agent_save_asynchronous(self):
        selected_demo_split = self.selected_demo.split('.')
        demo_name = selected_demo_split[0]
        demo_id = selected_demo_split[1]
        self.ros_node.save_task_episode(demo_name, demo_id)
        self.update_available_task_episode_name(self.ros_node.task_handler.get_names())
    
    def demo_start_button_trigger(self):
        if(self.state == SupervisorState.STANDBY or self.state == SupervisorState.DEMO_PAUSED):
            self.demo_start_button.configure(text="pause")
            self.demo_stop_button.state(['!disabled'])
            self.demo_save_button.state(['!disabled'])
            self.automatic_start_button.state(['disabled'])
            if(self.state == SupervisorState.STANDBY):
                self.ros_node.reset_episode()
                self.set_episode(self.ros_node.episode)
            self.state = SupervisorState.DEMO_RECORDING
            print("[INFO] Demonstration recording started")
        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.state = SupervisorState.DEMO_PAUSED
            self.demo_start_button.configure(text="start")
            print("[INFO] Demonstration recording paused")
        try:
            self.ros_node.update_state(self.state)
        except:
            pass
    
    def demo_stop_button_trigger(self):
        self.state = SupervisorState.STANDBY
        self.demo_start_button.configure(text="start")
        self.demo_stop_button.state(['disabled'])
        self.demo_save_button.state(['disabled'])
        self.automatic_start_button.state(['!disabled'])
        try:
            self.ros_node.update_state(self.state)
            self.ros_node.reset_episode()
        except:
            pass
        print("[INFO] Demonstration recording stopped")

    def demo_save_button_trigger(self):
        self.state = SupervisorState.DEMO_PAUSED
        self.demo_start_button.configure(text="start")
        self.ros_node.update_state(self.state)
        Thread(target= lambda: self.demo_save_asynchronous()).start()
    
    def demo_save_asynchronous(self):
        demo_name = self.demo_name_entry.get()
        if demo_name == "":
            print("[INFO] Demonstration name not specified")
        else:
            self.ros_node.save_demo(demo_name)
            self.update_available_demo_name(self.ros_node.demo_handler.get_names())
            print(f"[INFO] Demonstration saved as {demo_name}")

    def update_available_demo_name(self, name_array):
        self.saved_demo_name_list.delete(0, tkinter.END)
        for name in name_array:
            self.saved_demo_name_list.insert(0, name)
        self.demo_name_entry['values'] = tuple(name_array)

    def update_available_demo_id(self, id_array):
        self.saved_demo_id_list.delete(0,tkinter.END)
        for id_ in id_array:
            self.saved_demo_id_list.insert(tkinter.END, id_)

    def update_available_task_episode_name(self, name_array):
        self.saved_task_episode_name_list.delete(0, tkinter.END)
        for name in name_array:
            self.saved_task_episode_name_list.insert(0, name)
    
    def model_update_name_entry(self):
        name_array = self.ros_node.model_handler.get_names(ModelType.ACTOR)
        self.model_name_entry['values'] = tuple(name_array)

    def model_update_id_entry(self, name):
        id_array = self.ros_node.model_handler.get_ids(ModelType.ACTOR, name)
        self.model_id_entry['values'] = tuple(id_array)
    
    def select_model_trigger(self, val):
        try:
            selected_model_name = self.model_name_entry.get()
            self.model_update_id_entry(selected_model_name)
        except:
            pass

    def model_select_button_trigger(self):
        model_name = self.model_name_entry.get()
        if(model_name == ''):
            return
        model_id = self.model_id_entry.get()
        if(model_id == ''):
            return
        model_string = f"{model_name}/{model_id}"
        response = self.ros_node.call_service('agent/select_model', model_string)
        if response.success == True:
            self.selected_model = dict(name = model_name, id = model_id)
            self.update_info(selected_model=model_string)
    
    def add_demo_trigger(self):
        if(self.selection == InformationType.DEMO):
            demo_name = self.saved_demo_name_list.get(tkinter.ANCHOR)
            demo_id = self.saved_demo_id_list.get(tkinter.ANCHOR)
            if(demo_id == "" or demo_name == ""):
                return
            demo = demo_name+"."+str(demo_id)
            self.queued_demo_list.insert(tkinter.END, demo)
            self.selected_demo = self.queued_demo_list.get(0)
            self.ros_node.episode = self.ros_node.demo_handler.get(demo_name, demo_id)
            self.update_info(selected_demo = self.selected_demo)

    def remove_demo_trigger(self):
        selected_index = self.queued_demo_list.curselection()
        if(0 in selected_index and self.state not in [SupervisorState.STANDBY, SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
            print("[INFO] Stop processes before deselecting the current demonstration")
            return
        self.queued_demo_list.delete(tkinter.ANCHOR)
        self.selected_demo = self.queued_demo_list.get(0)
        self.update_info(selected_demo = self.selected_demo)

    def saved_demo_name_list_trigger(self, event):
        demo_name = self.saved_demo_name_list.get(tkinter.ANCHOR)
        try:
            ids = self.ros_node.demo_handler.get_ids(demo_name)
        except:
            ids = []
        self.selection = InformationType.DEMO
        self.update_available_demo_id(ids)

    def saved_task_episode_name_list_trigger(self, event):
        demo_name = self.saved_task_episode_name_list.get(tkinter.ANCHOR)
        try:
            ids = self.ros_node.task_handler.get_ids(demo_name)
        except:
            ids = []
        self.selection = InformationType.TASK_EPISODE
        self.update_available_demo_id(ids)
    
    def saved_demo_id_list_trigger(self, event):
        if(self.selection == InformationType.DEMO):
            demo_name = self.saved_demo_name_list.get(tkinter.ANCHOR)
            demo_id = self.saved_demo_id_list.get(tkinter.ANCHOR)
            if(demo_id == '' or demo_id == None):
                return
            try:
                Thread(target=lambda: self.set_episode(episode = self.ros_node.demo_handler.get(demo_name, demo_id))).start()
            except:
                pass
            print("[INFO] Displaying selected demonstration")
        elif(self.selection == InformationType.TASK_EPISODE):
            demo_name = self.saved_task_episode_name_list.get(tkinter.ANCHOR)
            task_id = self.saved_demo_id_list.get(tkinter.ANCHOR)
            if(task_id == '' or task_id == None):
                return
            try:
                Thread(target=lambda: self.set_episode(episode = self.ros_node.task_handler.get(demo_name, task_id))).start()
            except:
                pass
            print("[INFO] Displaying selected task episode")

    def set_episode(self, episode):
        self.episode = episode
        self.ros_node.episode = self.episode
        self.slider_trigger(self.slider_value)

    def update_episode_image(self, episode_index):
        if(episode_index >= self.episode.get_episode_length()):
            return

        image_episode_temp = self.episode.get_data(episode_index)['observation']['image']

        if(image_episode_temp == None):
            return

        if(image_episode_temp != self.image_episode_temp):
            self.image_episode_temp = image_episode_temp
            image = self.image_episode_temp.resize((480,360))
            self.image_episode = ImageTk.PhotoImage(image = image)
            self.image_episode_label.configure(image=self.image_episode)

    def slider_trigger(self, val):
        self.slider_value = float(val)

        if(self.episode.data_empty == False):
            no_of_divs = self.episode.get_episode_length()
            current_selection = int((self.slider_value/(1/no_of_divs)))
            if(current_selection == no_of_divs):
                current_selection -= 1
            self.episode_index = current_selection
            Thread(target=lambda: self.update_episode_image(current_selection)).start()
    