import tkinter
from tkinter import StringVar, ttk
import seaborn as sns

from control import Control
from display import Display

class GUI():
    def __init__(self, ros_node = None):

        self.ros_node = ros_node

        self.window = tkinter.Tk()
        self.window.title("GUI")
        self.window.attributes('-fullscreen', True)
        self.window.rowconfigure(0, weight=1)
        self.window.columnconfigure(0, weight=1)

        self.mainframe = tkinter.ttk.Frame(self.window, padding = "3 3 3 3")
        self.display_frame = tkinter.ttk.Frame(self.mainframe,  padding = "10 10 10 10")
        self.display = Display(parent = self.display_frame, ros_node = self.ros_node)
        self.control_frame = tkinter.ttk.Frame(self.mainframe, borderwidth =2,relief=tkinter.RIDGE, padding = "10 10 10 10")
        self.control = Control(parent = self.control_frame, ros_node = self.ros_node)

        self.mainframe.grid(sticky='nsew')
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.columnconfigure(0, weight=1)
        self.display_frame.grid(column=0, row=0, sticky='nsew')
        self.control_frame.grid(column=1, row=0, sticky='nsew')

        sns.set('notebook')
        sns.set_style("white")

        self.window.mainloop()