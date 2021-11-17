import tkinter
from tkinter import StringVar, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import numpy as np

#==============================================================

class Display():
    def __init__(self, parent = None, episode_kwargs = dict()):

        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

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

class Episode():
    def __init__(self, parent = None, max_linear_vel = 1.0, max_angular_vel = 1.0):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
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
        self.plot_sel = FigureCanvasTkAgg(self.plot_sel_fig, self.parent).get_tk_widget()

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
        self.plot_full = FigureCanvasTkAgg(self.plot_full_fig, self.plot_full_frame).get_tk_widget()

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
        # TODO:
        pass

class Current():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

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

#==============================================================

class Info():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

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
