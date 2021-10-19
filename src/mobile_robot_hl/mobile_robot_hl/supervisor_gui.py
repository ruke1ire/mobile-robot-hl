import tkinter
from tkinter import ttk
from PIL import Image, ImageTk
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SupervisorGUI():
    def __init__(self):

        self.window = tkinter.Tk()
        self.window.title("Supervisor GUI")
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

        sns.set_theme('notebook')
        sns.set_style("white")
        self.setup_extras()
        self.setup_mainframe()
        self.setup_display_frame()
        self.setup_control_frame()

    def setup_extras(self):
        img = np.zeros([300,300,3],dtype=np.uint8)
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
        self.image_model_label = tkinter.ttk.Label(self.display_frame,image=self.image_model)
        self.image_current_label = tkinter.ttk.Label(self.display_frame,image=self.image_current)
        self.image_slider = tkinter.ttk.Scale(self.display_frame, from_=0, to_= 100, orient=tkinter.HORIZONTAL)
        self.save_episode_button = tkinter.ttk.Button(self.display_frame, text="save")
        self.info_frame = tkinter.ttk.Frame(self.display_frame, borderwidth=2, relief=tkinter.SUNKEN)
        self.info_frame_title = tkinter.ttk.Label(self.info_frame, text="Information")

        self.current_action_fig = plt.figure(figsize=(3,3),frameon=False)

        self.current_action_ax = self.current_action_fig.add_axes([0, 0, 1, 1])
        self.current_action_ax.spines['left'].set_position('center')
        self.current_action_ax.spines['bottom'].set_position('center')
        self.current_action_ax.spines['right'].set_color('none')
        self.current_action_ax.spines['top'].set_color('none')
        self.current_action_ax.margins(x=0.01,y=0.01)
        self.current_action_ax.grid(True)
        self.current_action_ax.tick_params(labelsize=5)

        self.current_action_plot = FigureCanvasTkAgg(self.current_action_fig, self.display_frame)
        self.action_plot_fig = plt.figure(figsize=(3,1), frameon=False)
        #self.action_plot_ax = self.action_plot_fig.add_subplot(1,1,1)
        self.action_plot_ax = self.action_plot_fig.add_axes([0, 0, 1, 1])
        
        self.action_plot_ax.spines['right'].set_color('none')
        self.action_plot_ax.spines['top'].set_color('none')
        self.action_plot_ax.spines['bottom'].set_position('center')
        self.action_plot_ax.spines['left'].set_color('none')
        self.action_plot_ax.set_ylim([-3,3])
        self.action_plot_ax.margins(x=0.015)
        self.action_plot_ax.grid(True)
        self.action_plot_ax.tick_params(labelsize=5)
        self.action_plot_ax.scatter([0], [0])
        self.action_plot_ax.plot([0], [0])

        #plt.show()
        self.action_plot_plot = FigureCanvasTkAgg(self.action_plot_fig, self.display_frame)

        self.image_model_label.grid(column=0, row=0)
        self.current_action_plot.get_tk_widget().grid(column=1, row =0, sticky='nsew')
        self.action_plot_plot.get_tk_widget().grid(column=0, row =1, columnspan=2, sticky='ew')
        self.image_current_label.grid(column=0, row=4)
        self.image_slider.grid(column=0, row = 2, columnspan = 2, sticky='ew')
        self.save_episode_button.grid(column=0, row = 3)
        self.info_frame.grid(column=1, row=3, rowspan=2,sticky='nsew')
        self.info_frame_title.grid(column=0, row=0)
        self.info_frame.rowconfigure(0, weight=1)
        self.info_frame.columnconfigure(0, weight=1)

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
        self.automatic_start_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="start")
        self.automatic_stop_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="stop")
        self.automatic_take_over_button = tkinter.ttk.Button(self.automatic_control_button_frame, text="take over")
        self.demo_control_button_frame = tkinter.ttk.Frame(self.demo_control_frame)
        self.demo_start_button = tkinter.ttk.Button(self.demo_control_button_frame, text="start")
        self.demo_stop_button = tkinter.ttk.Button(self.demo_control_button_frame, text="stop")
        self.model_control_button_frame = tkinter.ttk.Frame(self.model_control_frame)
        self.model_start_button = tkinter.ttk.Button(self.model_control_button_frame, text="start")
        self.saved_demo_list = tkinter.Listbox(self.task_queue_frame)
        self.task_add_button = tkinter.ttk.Button(self.task_queue_frame, text=">>")
        self.task_remove_button = tkinter.ttk.Button(self.task_queue_frame, text="<<")
        self.queued_demo_list = tkinter.Listbox(self.task_queue_frame)
        self.demo_label = tkinter.ttk.Label(self.demo_control_frame, text="Demonstration Control Panel")
        self.automatic_label = tkinter.ttk.Label(self.automatic_control_frame, text="Automatic Control Panel")
        self.model_label = tkinter.ttk.Label(self.model_control_frame, text="Model Control Panel")
        self.saved_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Available Demo")
        self.queued_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Queued Demo")

        self.automatic_control_frame.grid(column=0, row=0, columnspan = 2, sticky='nsew')
        self.automatic_label.grid(column=0, row=0)
        self.automatic_control_button_frame.grid(column=0, row = 1)
        self.automatic_start_button.grid(column=1, row = 0)
        self.automatic_stop_button.grid(column=0, row = 0)
        self.automatic_take_over_button.grid(column=2, row = 0)
        self.automatic_control_frame.rowconfigure(0, weight=1)
        self.automatic_control_frame.columnconfigure(0, weight=1)

        self.demo_control_frame.grid(column=0, row=1, sticky='nsew')
        self.demo_label.grid(column=0, row=0)
        self.demo_control_button_frame.grid(column=0, row =1)
        self.demo_start_button.grid(column=0, row = 0)
        self.demo_stop_button.grid(column=1, row = 0)
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
        self.image_current = ImageTk.PhotoImage(img_arr)
        self.image_current_label.configure(image=self.image_current)

if __name__ == "__main__":
    gui = SupervisorGUI()
    gui.window.mainloop()
