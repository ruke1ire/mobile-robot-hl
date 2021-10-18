import tkinter
from tkinter import ttk

class SupervisorGUI():
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title("Supervisor GUI")

        # setup frames
        self.mainframe = tkinter.ttk.Frame(self.window, padding = "3 3 3 3")
        self.display_frame = tkinter.ttk.Frame(self.mainframe)
        self.control_frame = tkinter.ttk.Frame(self.mainframe)

        self.mainframe.grid(column=0, row=0, sticky='nsew')
        self.display_frame.grid(column=0, row=0, columnspan = 2, rowspan = 3)
        self.control_frame.grid(column=2, row=0, columnspan = 2, rowspan = 3)

        # setup display_frame section
        self.image_model = tkinter.ttk.Label(self.display_frame,text="image_model")
        self.image_current = tkinter.ttk.Label(self.display_frame,text="image_current")
        self.image_slider = tkinter.ttk.Scale(self.display_frame, from_=0, to_= 100, orient=tkinter.HORIZONTAL)
        self.save_episode_button = tkinter.ttk.Button(self.display_frame, text="save")
        self.info_frame = tkinter.ttk.Frame(self.display_frame)
        self.info_frame_title = tkinter.ttk.Label(self.info_frame, text="Information")

        self.image_model.grid(column=0, row=0)
        self.image_current.grid(column=0, row=1)
        self.image_slider.grid(column=0, row = 2)
        self.save_episode_button.grid(column=0, row = 3)
        self.info_frame.grid(column=1, row=3)
        self.info_frame_title.grid(column=0, row=0)

        # setup control_frame section
        self.automatic_control_frame = tkinter.ttk.Frame(self.control_frame)
        self.demo_control_frame = tkinter.ttk.Frame(self.control_frame)
        self.model_control_frame = tkinter.ttk.Frame(self.control_frame)
        self.task_queue_frame = tkinter.ttk.Frame(self.control_frame)
        self.automatic_start_button = tkinter.ttk.Button(self.automatic_control_frame, text="start/pause")
        self.automatic_stop_button = tkinter.ttk.Button(self.automatic_control_frame, text="stop")
        self.automatic_take_over_button = tkinter.ttk.Button(self.automatic_control_frame, text="take over")
        self.demo_start_button = tkinter.ttk.Button(self.demo_control_frame, text="start/pause")
        self.demo_stop_button = tkinter.ttk.Button(self.demo_control_frame, text="stop")
        self.model_start_button = tkinter.ttk.Button(self.model_control_frame, text="start/pause")
        self.saved_demo_list = tkinter.Listbox(self.task_queue_frame)
        self.task_add_button = tkinter.ttk.Button(self.task_queue_frame, text=">>")
        self.task_remove_button = tkinter.ttk.Button(self.task_queue_frame, text="<<")
        self.queued_demo_list = tkinter.Listbox(self.task_queue_frame)
        self.demo_label = tkinter.ttk.Label(self.demo_control_frame, text="Demonstration Control Panel")
        self.automatic_label = tkinter.ttk.Label(self.automatic_control_frame, text="Automatic Control Panel")
        self.model_label = tkinter.ttk.Label(self.model_control_frame, text="Model Control Panel")
        self.saved_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Available Demonstrations")
        self.queued_demo_list_label = tkinter.ttk.Label(self.task_queue_frame, text="Queued Demonstrations")

        self.automatic_control_frame.grid(column=0, row=0)
        self.automatic_label.grid(column=0, row=0, columnspan = 3)
        self.automatic_start_button.grid(column=1, row = 1)
        self.automatic_stop_button.grid(column=0, row = 1)
        self.automatic_take_over_button.grid(column=2, row = 1)

        self.demo_control_frame.grid(column=0, row=1)
        self.demo_label.grid(column=0, row=0, columnspan = 2)
        self.demo_start_button.grid(column=0, row = 1)
        self.demo_stop_button.grid(column=1, row = 1)

        self.model_control_frame.grid(column=1, row=1)
        self.model_label.grid(column = 0, row = 0)
        self.model_start_button.grid(column=0, row=1)

        self.task_queue_frame.grid(column=0, row=2)
        self.saved_demo_list_label.grid(column = 0, row = 0)
        self.queued_demo_list_label.grid(column = 2, row = 0)
        self.saved_demo_list.grid(column=0, row=1, rowspan=2)
        self.queued_demo_list.grid(column=2, row=1, rowspan=2)
        self.task_add_button.grid(column=1, row=1)
        self.task_remove_button.grid(column=1, row=2)

if __name__ == "__main__":
    gui = SupervisorGUI()
    gui.window.mainloop()

