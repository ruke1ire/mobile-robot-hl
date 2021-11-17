import tkinter
from tkinter import ttk

#==============================================================

class Control():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.demo_frame = tkinter.ttk.Frame(self.parent)
        self.demo = Demo(parent = self.demo_frame)
        self.model_frame = tkinter.ttk.Frame(self.parent)
        self.model = Model(parent = self.model_frame)
        self.task_frame = tkinter.ttk.Frame(self.parent)
        self.task = Task(parent= self.task_frame)

        self.demo_frame.grid(row = 0, column = 0, sticky = 'nsew')
        self.model_frame.grid(row = 0, column = 1, sticky = 'nsew')
        self.task_frame.grid(row = 1, column = 0, columnspan = 2, sticky = 'nsew')

        self.parent.rowconfigure(0, weight = 1)
        self.parent.rowconfigure(1, weight = 1)
        self.parent.columnconfigure(0, weight = 1)
        self.parent.columnconfigure(1, weight = 1)

#==============================================================

class Task():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        self.title = tkinter.ttk.Label(self.parent, text="Task Control Panel")
        self.buttons_frame = tkinter.ttk.Frame(self.parent)
        self.buttons_stop = tkinter.ttk.Button(self.buttons_frame, text="stop", command = self.buttons_stop_trigger)
        self.buttons_start_pause = tkinter.ttk.Button(self.buttons_frame, text="start", command = self.buttons_start_pause_trigger)
        self.buttons_take_over = tkinter.ttk.Button(self.buttons_frame, text="take over", command = self.buttons_take_over_trigger)
        self.buttons_save= tkinter.ttk.Button(self.buttons_frame, text="save", command = self.buttons_save_trigger)

        self.selection_frame = tkinter.ttk.Frame(self.parent, padding = "0 10 0 0")
        self.selection = Selection(parent = self.selection_frame)

        self.title.grid(row = 0, column = 0)
        self.buttons_frame.grid(row = 1, column = 0)
        self.buttons_stop.grid(column=0, row = 0)
        self.buttons_start_pause.grid(column=1, row = 0)
        self.buttons_take_over.grid(column=2, row = 0)
        self.buttons_save.grid(column=3, row = 0)
        self.selection_frame.grid(row = 2, column = 0, sticky = 'nsew')

        self.parent.rowconfigure(0, weight=1)
        self.parent.rowconfigure(2, weight=1)
        self.parent.columnconfigure(0, weight=1)
    
    def buttons_start_pause_trigger(self):
        pass

    def buttons_stop_trigger(self):
        pass

    def buttons_take_over_trigger(self):
        pass

    def buttons_save_trigger(self):
        pass


class Demo():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.title = tkinter.ttk.Label(self.parent, text="Demonstration Control Panel")
        self.entry = tkinter.ttk.Combobox(self.parent)
        self.buttons_frame = tkinter.ttk.Frame(self.parent)
        self.buttons_start = tkinter.ttk.Button(self.buttons_frame, text="start", command = self.buttons_start_trigger)
        self.buttons_stop = tkinter.ttk.Button(self.buttons_frame, text="stop", command = self.buttons_stop_trigger)
        self.buttons_save= tkinter.ttk.Button(self.buttons_frame, text="save", command = self.buttons_save_trigger)

        self.buttons_stop.state(['disabled'])
        self.buttons_save.state(['disabled'])

        self.title.grid(column=0, row=0)
        self.entry.grid(column = 0, row = 1)
        self.buttons_frame.grid(column=0, row =2)
        self.buttons_start.grid(column=0, row = 0)
        self.buttons_stop.grid(column=1, row = 0)
        self.buttons_save.grid(column=2, row = 0)

        self.parent.rowconfigure(0, weight=1)
        self.parent.columnconfigure(0, weight=1)
    
    def buttons_start_trigger(self):
        pass

    def buttons_stop_trigger(self):
        pass

    def buttons_save_trigger(self):
        pass


class Model():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        self.title = tkinter.ttk.Label(self.parent, text="Model Control Panel")
        self.entries_frame = tkinter.ttk.Frame(self.parent)
        self.entries_name = tkinter.ttk.Combobox(self.entries_frame)
        self.entries_id = tkinter.ttk.Combobox(self.entries_frame)
        self.select = tkinter.ttk.Button(self.parent, text="select", command = self.select_trigger)

        self.entries_name.bind('<<ComboboxSelected>>', self.entries_name_trigger)

        self.title.grid(column = 0, row = 0)
        self.entries_frame.grid(column = 0, row = 1)
        self.entries_name.grid(column = 0, row = 0)
        self.entries_id.grid(column = 1, row = 0)
        self.select.grid(column=0, row=2)

        self.parent.rowconfigure(0, weight=1)
        self.parent.columnconfigure(0, weight=1)
    
    def entries_name_trigger(self, val):
        pass

    def select_trigger(self):
        pass

#==============================================================

class Selection():
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        self.demo_label = tkinter.ttk.Label(self.parent, text="Available Demo")
        self.demo_box = tkinter.Listbox(self.parent)
        self.task_label = tkinter.ttk.Label(self.parent, text="Available Task Episode")
        self.task_box = tkinter.Listbox(self.parent)
        self.id_label = tkinter.ttk.Label(self.parent, text="Available ID")
        self.id_box = tkinter.Listbox(self.parent)
        self.queue_label = tkinter.ttk.Label(self.parent, text="Queued Episode")
        self.queue_box = tkinter.Listbox(self.parent)
        self.add = tkinter.ttk.Button(self.parent, text=">>", command= self.add_trigger)
        self.remove = tkinter.ttk.Button(self.parent, text="<<", command = self.remove_trigger)
        
        self.demo_box.bind('<<ListboxSelect>>', self.demo_box_trigger)
        self.task_box.bind('<<ListboxSelect>>', self.task_box_trigger)
        self.id_box.bind('<<ListboxSelect>>', self.id_box_trigger)

        self.demo_label.grid(column = 0,  row = 0)
        self.task_label.grid(column = 0, row = 3)
        self.queue_label.grid(column = 3, row = 0)
        self.id_label.grid(column = 1, row = 0)
        self.demo_box.grid(column=0, row=1, rowspan=2, sticky='nsew')
        self.id_box.grid(column=1, row=1, rowspan=5, sticky='nsew')
        self.task_box.grid(column=0, row=4, rowspan=2, sticky='nsew')
        self.queue_box.grid(column=3, row=1, rowspan=5, sticky = 'nsew')
        self.add.grid(column=2, row=2)
        self.remove.grid(column=2, row=4)

        self.parent.rowconfigure(2, weight=1)
        self.parent.rowconfigure(4, weight=1)
        self.parent.columnconfigure(0, weight=1)
        self.parent.columnconfigure(1, weight=1)
        self.parent.columnconfigure(3, weight=1)
    
    def demo_box_trigger(self, event):
        pass

    def task_box_trigger(self, event):
        pass

    def id_box_trigger(self, event):
        pass
    
    def add_trigger(self):
        pass

    def remove_trigger(self):
        pass


