import tkinter
from enum import Enum
from tkinter import StringVar, ttk
from PIL import Image, ImageTk
from numpy.core.fromnumeric import put
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SupervisorGUI():
    def __init__(self, ros_node = None):

        self.ros_node = ros_node

        self.window = tkinter.Tk()
        self.window.title("Supervisor GUI")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

        self.state = SupervisorState.STANDBY
        self.model_training_state = 'no'
        self.selected_demo = None

        sns.set('notebook')
        sns.set_style("white")
        self.setup_extras()
        self.setup_mainframe()
        self.setup_display_frame()
        self.setup_control_frame()

    def setup_extras(self):
        img = np.zeros([360,480,3],dtype=np.uint8)
        img.fill(255)
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
        self.image_model = ImageTk.PhotoImage(self.blank_image)
        self.image_current = ImageTk.PhotoImage(self.blank_image)
        self.image_model_label = tkinter.Label(self.display_frame,image=self.image_model)
        self.image_current_label = tkinter.Label(self.display_frame,image=self.image_current)
        self.image_slider = tkinter.ttk.Scale(self.display_frame, from_=0, to_= 100, orient=tkinter.HORIZONTAL)
        self.info_frame = tkinter.ttk.Frame(self.display_frame, borderwidth=2, relief=tkinter.SUNKEN, padding = "10 10 10 10")
        self.info_frame_title = tkinter.ttk.Label(self.info_frame, text="Information")
        self.info_desired_vel = tkinter.ttk.Label(self.info_frame, text="Desired Velocity: \n\tLinear: 0.0 m/s\n\tAngular: 0.0 rad/s")
        self.info_user_vel = tkinter.ttk.Label(self.info_frame, text="User Velocity:\n\tLinear: 0.0 m/s\n\tAngular: 0.0 rad/s")
        self.info_user_termination_flag = tkinter.ttk.Label(self.info_frame, text="User Termination Flag: False")
        self.info_agent_vel = tkinter.ttk.Label(self.info_frame, text="Agent Velocity: \n\tLinear: 0.0 m/s\n\tAngular: 0.0 rad/s")
        self.info_agent_termination_flag = tkinter.ttk.Label(self.info_frame, text="Agent Termination Flag: False")
        self.info_current_demo = tkinter.ttk.Label(self.info_frame, text="Selected Demonstration: None")

        self.current_action_fig = plt.figure(figsize=(3,3),frameon=False)

        self.current_action_ax = self.current_action_fig.add_axes([0, 0, 1, 1])
        self.current_action_ax.spines['left'].set_position('center')
        self.current_action_ax.spines['bottom'].set_position('center')
        self.current_action_ax.spines['right'].set_color('none')
        self.current_action_ax.spines['top'].set_color('none')
        self.current_action_ax.margins(x=0.01,y=0.01)
        self.current_action_ax.grid(True)
        self.current_action_ax.set_ylim([-1,1])
        self.current_action_ax.set_xlim([-1,1])
        self.current_action_ax.tick_params(labelsize=5)

        self.current_action_plot = FigureCanvasTkAgg(self.current_action_fig, self.display_frame)

        self.action_plot_fig = plt.figure(figsize=(3,2), frameon=False)
        self.action_plot_ax = self.action_plot_fig.add_axes([0, 0, 1, 1])
        
        self.action_plot_ax.spines['right'].set_color('none')
        self.action_plot_ax.spines['top'].set_color('none')
        self.action_plot_ax.spines['bottom'].set_position('center')
        self.action_plot_ax.spines['left'].set_color('none')
        self.action_plot_ax.set_ylim([-1,1])
        self.action_plot_ax.margins(x=0.015)
        self.action_plot_ax.grid(True)
        self.action_plot_ax.tick_params(labelsize=5)

        self.action_plot_plot = FigureCanvasTkAgg(self.action_plot_fig, self.display_frame)

        self.image_model_label.grid(column=0, row=0)
        self.current_action_plot.get_tk_widget().grid(column=1, row =0, sticky='nsew')
        self.action_plot_plot.get_tk_widget().grid(column=0, row =1, columnspan=2, sticky='ew')
        self.image_current_label.grid(column=0, row=4)
        self.image_slider.grid(column=0, row = 2, columnspan = 2, sticky='ew')
        self.info_frame.grid(column=1, row=3, rowspan=2,sticky='nsew')
        self.info_frame_title.grid(column=0, row=0)
        self.info_desired_vel.grid(column = 0, row=1, sticky = 'w')
        self.info_user_vel.grid(column = 0, row = 2, sticky = 'w')
        self.info_user_termination_flag.grid(column = 0, row = 3, sticky='w')
        self.info_agent_vel.grid(column = 0, row =4, sticky='w')
        self.info_agent_termination_flag.grid(column = 0, row = 5, sticky='w')
        self.info_current_demo.grid(column = 0, row = 6, sticky='w')

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
        self.automatic_stop_button.state(['disabled'])
        self.automatic_take_over_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="take over", command = self.agent_take_over_button_trigger)
        self.automatic_take_over_button.state(['disabled'])
        self.automatic_save_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="save", command = self.agent_save_button_trigger)
        self.automatic_save_button.state(['disabled'])
        self.demo_control_button_frame = tkinter.ttk.Frame(self.demo_control_frame)
        self.demo_start_button = tkinter.ttk.Button(self.demo_control_button_frame, text="start", command = self.demo_start_button_trigger)
        self.demo_stop_button = tkinter.ttk.Button(self.demo_control_button_frame, text="stop", command = self.demo_stop_button_trigger)
        self.demo_stop_button.state(['disabled'])
        self.demo_save_button = tkinter.ttk.Button(self.demo_control_button_frame, text="save", command = self.demo_save_button_trigger)
        self.demo_save_button.state(['disabled'])
        self.demo_name = StringVar()
        self.demo_name_entry = tkinter.ttk.Combobox(self.demo_control_frame, textvariable=self.demo_name)

        self.model_control_button_frame = tkinter.ttk.Frame(self.model_control_frame)
        self.model_start_button = tkinter.ttk.Button(self.model_control_button_frame, text="start", command = self.model_start_button_trigger)
        self.saved_demo = StringVar(value=[])
        self.saved_demo_list = tkinter.Listbox(self.task_queue_frame, listvariable=self.saved_demo)
        self.saved_demo_list.insert(tkinter.END, "TEMP 1")
        self.saved_demo_list.insert(tkinter.END, "TEMP 2")
        self.saved_demo_list.insert(tkinter.END, "TEMP 3")
        self.saved_demo_list.insert(tkinter.END, "TEMP 4")
        self.saved_demo_list.insert(tkinter.END, "TEMP 5")
        self.saved_demo_list.insert(tkinter.END, "TEMP 6")
        self.saved_demo_list.insert(tkinter.END, "TEMP 7")
        self.saved_demo_list.insert(tkinter.END, "TEMP 8")
        self.saved_demo_list.insert(tkinter.END, "TEMP 9")
        self.task_add_button = tkinter.ttk.Button(self.task_queue_frame, text=">>", command= self.add_demo_trigger)
        self.task_remove_button = tkinter.ttk.Button(self.task_queue_frame, text="<<", command = self.remove_demo_trigger)
        self.queued_demo = StringVar(value=[])
        self.queued_demo_list = tkinter.Listbox(self.task_queue_frame, listvariable=self.queued_demo)
        self.demo_label = tkinter.ttk.Label(self.demo_control_frame, text="Demonstration Control Panel")
        self.automatic_label = tkinter.ttk.Label(self.automatic_control_frame, text="Automatic Control Panel")
        self.model_label = tkinter.ttk.Label(self.model_control_frame, text="Model Control Panel")
        self.saved_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Available Demo")
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
        self.model_start_button.grid(column=0, row=0)
        self.model_control_frame.rowconfigure(0, weight=1)
        self.model_control_frame.columnconfigure(0, weight=1)

        self.task_queue_frame.grid(column=0, row=2, columnspan = 2, sticky='nsew')
        self.saved_demo_list_label.grid(column = 0, row = 0)
        self.queued_demo_list_label.grid(column = 2, row = 0)
        self.saved_demo_list.grid(column=0, row=1, rowspan=4, sticky='nsew')
        self.queued_demo_list.grid(column=2, row=1, rowspan=4, sticky = 'nsew')
        self.task_add_button.grid(column=1, row=2)
        self.task_remove_button.grid(column=1, row=3)

        self.task_queue_frame.rowconfigure(2, weight=1)
        self.task_queue_frame.rowconfigure(3, weight=1)
        self.task_queue_frame.columnconfigure(0, weight=1)
        self.task_queue_frame.columnconfigure(2, weight=1)

        self.control_frame.rowconfigure(0, weight=1)
        self.control_frame.rowconfigure(1, weight=1)
        self.control_frame.rowconfigure(2, weight=2)
        self.control_frame.columnconfigure(0, weight=1)
        self.control_frame.columnconfigure(1, weight=1)
    
    def update_image_current(self, img_arr):
        image = Image.fromarray(img_arr).resize((480,360))
        self.image_current = ImageTk.PhotoImage(image = image)
        self.image_current_label.configure(image=self.image_current)
    
    def update_current_action_plot(self, desired_vel = None, user_vel = None, agent_vel = None):
        if type(desired_vel) == dict:
            try:
                self.current_action_desired_vel.remove()
            except:
                pass
            self.current_action_desired_vel = self.current_action_ax.scatter(desired_vel['angular'], desired_vel['linear'],c = 'tab:blue', label="desired velocity", alpha=0.8, marker='x')
        if type(user_vel) == dict:
            try:
                self.current_action_user_vel.remove()
            except:
                pass
            self.current_action_user_vel = self.current_action_ax.scatter(user_vel['angular'], user_vel['linear'],c = 'tab:orange', label="user velocity", alpha=0.8, marker='o')
        if type(agent_vel) == dict:
            try:
                self.current_action_agent_vel.remove()
            except:
                pass
            self.current_action_agent_vel = self.current_action_ax.scatter(agent_vel['angular'], agent_vel['linear'],c = 'tab:green', label="agent velocity", alpha = 0.8, marker='^')

        self.current_action_ax.legend(loc='upper right', prop={'size': 8})
        self.current_action_plot.draw_idle()
    
    def update_info(self, desired_vel=None, user_vel=None, agent_vel=None, agent_termination=None, user_termination=None, selected_demo=None):
        if type(desired_vel) == dict:
            self.info_desired_vel.configure(text=f"Desired Velocity: \n\tLinear: {desired_vel['linear']:.2f} m/s\n\tAngular: {desired_vel['angular']:.2f} rad/s")
        if type(user_vel) == dict:
            self.info_user_vel.configure(text=f"User Velocity: \n\tLinear: {user_vel['linear']:.2f} m/s\n\tAngular: {user_vel['angular']:.2f} rad/s")
        if type(agent_vel) == dict:
            self.info_agent_vel.configure(text=f"Agent Velocity: \n\tLinear: {agent_vel['linear']:.2f} m/s\n\tAngular: {agent_vel['angular']:.2f} rad/s")
        if type(agent_termination) == bool:
            self.info_agent_termination_flag.configure(text=f"Agent Termination Flag: {agent_termination}")
        if type(user_termination) == bool:
            self.info_agent_termination_flag.configure(text=f"User Termination Flag: {user_termination}")
        if type(selected_demo) == str:
            self.info_current_demo.configure(text=f"Selected Demonstration: {selected_demo}")

    def update_action_plot(self, desired_vel = None, user_vel = None, agent_vel = None):
        if type(desired_vel) == dict:
            try:
                self.action_plot_desired_vel_line_linear.pop(0).remove()
                self.action_plot_desired_vel_line_angular.pop(0).remove()
            except:
                pass
            list_range_desired_vel = list(range(1,1+len(desired_vel['linear'])))
            self.action_plot_desired_vel_line_linear = self.action_plot_ax.plot(list_range_desired_vel,desired_vel['linear'], c = 'tab:blue', label="desired linear velocity", alpha=0.8)
            self.action_plot_desired_vel_line_angular = self.action_plot_ax.plot(list_range_desired_vel,desired_vel['angular'], c = 'tab:cyan', label="desired angular velocity", alpha=0.8)
        if type(user_vel) == dict:
            try:
                self.action_plot_user_vel_line_linear.pop(0).remove()
                self.action_plot_user_vel_line_angular.pop(0).remove()
            except:
                pass
            list_range_user_vel = list(range(1,1+len(user_vel['linear'])))
            self.action_plot_user_vel_line_linear =  self.action_plot_ax.plot(list_range_user_vel, user_vel['linear'],c = 'tab:orange', label="user linear velocity", alpha=0.8)
            self.action_plot_user_vel_line_angular =  self.action_plot_ax.plot(list_range_user_vel, user_vel['angular'],c = 'tab:brown', label="user angular velocity", alpha=0.8)
            #self.action_plot_user_vel_line = self.action_plot_ax.plot(user_vel['linear'], user_vel['angular'],c = 'tab:orange', alpha=0.8)
        if type(agent_vel) == dict:
            try:
                self.action_plot_agent_vel_line_linear.pop(0).remove()
                self.action_plot_agent_vel_line_angular.pop(0).remove()
            except:
                pass
            list_range_agent_vel = list(range(1, 1+len(agent_vel['linear'])))
            self.action_plot_agent_vel_line_linear = self.action_plot_ax.plot(list_range_agent_vel, agent_vel['linear'],c = 'tab:green', label="agent linear velocity", alpha = 0.8)
            self.action_plot_agent_vel_line_angular = self.action_plot_ax.plot(list_range_agent_vel, agent_vel['angular'],c = 'olive', label="agent angular velocity", alpha = 0.8)

        self.action_plot_ax.legend(loc='lower left', prop={'size': 8})
        self.action_plot_plot.draw_idle()
    
    def agent_start_button_trigger(self):
        if self.selected_demo == None:
            print("[INFO] No demonstration selected")
            return
        if(self.state==SupervisorState.TASK_PAUSED or self.state == SupervisorState.STANDBY or self.state == SupervisorState.TASK_TAKE_OVER):
            self.state = SupervisorState.TASK_RUNNING
            self.automatic_take_over_button.configure(text="take over")
            self.automatic_start_button.configure(text="pause")
            self.automatic_take_over_button.state(['!disabled'])
            self.automatic_stop_button.state(['!disabled'])
            self.automatic_save_button.state(['!disabled'])
            self.demo_start_button.state(['disabled'])
            self.demo_save_button.state(['disabled'])
            print("[INFO] Automatic control started")
        elif(self.state == SupervisorState.TASK_RUNNING):
            self.state = SupervisorState.TASK_PAUSED
            self.automatic_start_button.configure(text="start")
            print("[INFO] Automatic control paused")

        try:
            status = self.ros_node.update_state(self.state)
        except:
            pass

    def agent_stop_button_trigger(self):
        self.state = SupervisorState.STANDBY
        self.automatic_start_button.configure(text="start")
        self.automatic_take_over_button.configure(text="take over")
        self.automatic_take_over_button.state(['disabled'])
        self.automatic_stop_button.state(['disabled'])
        self.automatic_save_button.state(['disabled'])
        self.demo_start_button.state(['!disabled'])
        try:
            self.ros_node.update_state(self.state)
        except:
            pass
        print("[INFO] Automatic control stopped")

    def agent_take_over_button_trigger(self):
        if(self.state == SupervisorState.TASK_RUNNING or self.state == SupervisorState.TASK_PAUSED):
            self.automatic_take_over_button.configure(text="pause")
            self.state = SupervisorState.TASK_TAKE_OVER
            self.automatic_start_button.configure(text="start")
            print("[INFO] Supervisor take-over")
        elif(self.state == SupervisorState.TASK_TAKE_OVER):
            self.state = SupervisorState.TASK_PAUSED
            self.automatic_take_over_button.configure(text="take over")
            print("[INFO] Supervisor take-over paused")
        try:
            self.ros_node.update_state(self.state)
        except:
            pass

    def agent_save_button_trigger(self):
        self.state = SupervisorState.TASK_PAUSED
        self.automatic_start_button.configure(text="start")
        self.automatic_take_over_button.configure(text="take over")
        print(f"[INFO] Saved task episode to {self.selected_demo}")
        try:
            self.ros_node.update_state(self.state)
        except:
            pass
    
    def demo_start_button_trigger(self):
        if(self.state == SupervisorState.STANDBY or self.state == SupervisorState.DEMO_PAUSED):
            self.state = SupervisorState.DEMO_RECORDING
            self.demo_start_button.configure(text="pause")
            self.demo_stop_button.state(['!disabled'])
            self.demo_save_button.state(['!disabled'])
            self.automatic_start_button.state(['disabled'])
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
        print("[INFO] Demonstration recording stopped")
        try:
            self.ros_node.update_state(self.state)
        except:
            pass

    def demo_save_button_trigger(self):
        demo_name = self.demo_name_entry.get()
        if demo_name == "":
            print("[INFO] Demonstration name not specified")
        else:
            print(f"[INFO] Demonstration saved as {demo_name}")
        self.state = SupervisorState.DEMO_PAUSED
        self.demo_start_button.configure(text="start")
        try:
            self.ros_node.update_state(self.state)
        except:
            pass

    def demo_update_entry(self, demo_names_tuple):
        self.demo_name_entry['values'] = demo_names_tuple
    
    def model_start_button_trigger(self):
        if(self.model_training_state == 'no'):
            try:
                self.ros_node.call_service(service_name = 'trainer/start')
            except:
                pass

            self.model_training_state = 'yes'
            self.model_start_button.configure(text="pause")
            print("[INFO] Model training started")
        elif(self.model_training_state == 'yes'):
            try:
                self.ros_node.call_service(service_name = 'trainer/pause')
            except:
                pass

            self.model_training_state = 'no'
            self.model_start_button.configure(text="start")
            print("[INFO] Model training stopped")
    
    def add_demo_trigger(self):
        demo_name = self.saved_demo_list.get(tkinter.ANCHOR)
        self.queued_demo_list.insert(tkinter.END, demo_name)
        self.selected_demo = self.queued_demo_list.get(0)

    def remove_demo_trigger(self):
        selected_index = self.queued_demo_list.curselection()
        if(0 in selected_index and self.state not in [SupervisorState.STANDBY, SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
            print("[INFO] Stop processes before deselecting the current demonstration")
            return
        self.queued_demo_list.delete(tkinter.ANCHOR)
        self.selected_demo = self.queued_demo_list.get(0)

class SupervisorState(Enum):
    STANDBY = 0
    TASK_RUNNING = 101
    TASK_PAUSED = 102
    TASK_TAKE_OVER = 103
    DEMO_RECORDING = 201
    DEMO_PAUSED = 202

def new_thread(gui):
    import math
    import time
    data = []
    i = 0
    while True:
        if len(data) > 50:
            del data[0]
        data.append(i)
        gui.update_current_action_plot(desired_vel = {'linear':math.cos(i),'angular':math.cos(i)}, user_vel={'linear':math.cos(i),'angular':math.sin(i)}, agent_vel ={'linear':2*math.cos(2*i),'angular':math.sin(2*i)})
        gui.update_info(desired_vel = {'linear':math.cos(i),'angular':math.cos(i)}, user_vel={'linear':math.cos(i),'angular':math.sin(i)}, agent_vel ={'linear':2*math.cos(2*i),'angular':math.sin(2*i)})
        gui.update_action_plot(desired_vel = {'linear':[math.cos(j) for j in data], 'angular':[math.cos(j) for j in data]}, agent_vel = {'linear':[2*math.cos(2*j) for j in data], 'angular':[math.sin(2*j) for j in data]}, user_vel = {'linear':[math.cos(j) for j in data], 'angular':[math.sin(j) for j in data]})
        i += 0.1

if __name__ == "__main__":
    from threading import Thread

    gui = SupervisorGUI()
    t = Thread(target=new_thread, args=(gui,))
    t.start()
    gui.window.mainloop()
