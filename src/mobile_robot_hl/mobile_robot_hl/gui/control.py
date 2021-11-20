import tkinter
from tkinter import ttk
from threading import Thread
import json

from mobile_robot_hl.gui.utils import *
from mobile_robot_hl.model import *
from mobile_robot_hl.utils import *

#==============================================================

class Control():
    def __init__(self, parent = None, ros_node = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.ros_node = ros_node
        
        self.demo_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.demo = Demo(parent = self.demo_frame, ros_node = self.ros_node)
        self.model_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.model = Model(parent = self.model_frame, ros_node= self.ros_node)
        self.task_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.task = Task(parent= self.task_frame, ros_node = self.ros_node)
        self.selection_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.selection = Selection(parent = self.selection_frame, ros_node = self.ros_node)

        self.demo_frame.grid(row = 0, column = 0, sticky = 'nsew')
        self.model_frame.grid(row = 0, column = 1, sticky = 'nsew')
        self.task_frame.grid(row = 1, column = 0, columnspan = 2, sticky = 'nsew')
        self.selection_frame.grid(row = 2, column = 0, columnspan = 2, sticky = 'nsew')

        self.parent.rowconfigure(0, weight = 1)
        self.parent.rowconfigure(1, weight = 1)
        self.parent.rowconfigure(2, weight = 2)
        self.parent.columnconfigure(0, weight = 1)
        self.parent.columnconfigure(1, weight = 1)

#==============================================================

class Task():
    def __init__(self, parent = None, ros_node = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.ros_node = ros_node

        self.title = tkinter.ttk.Label(self.parent, text="Task Control Panel")
        self.buttons_frame = tkinter.ttk.Frame(self.parent)
        self.buttons_stop = tkinter.ttk.Button(self.buttons_frame, text="stop", command = self.buttons_stop_trigger)
        self.buttons_start_pause = tkinter.ttk.Button(self.buttons_frame, text="start", command = self.buttons_start_pause_trigger)
        self.buttons_take_over = tkinter.ttk.Button(self.buttons_frame, text="take over", command = self.buttons_take_over_trigger)
        self.buttons_save= tkinter.ttk.Button(self.buttons_frame, text="save", command = self.buttons_save_trigger)

        self.title.grid(row = 0, column = 0)
        self.buttons_frame.grid(row = 1, column = 0)
        self.buttons_stop.grid(column=0, row = 0)
        self.buttons_start_pause.grid(column=1, row = 0)
        self.buttons_take_over.grid(column=2, row = 0)
        self.buttons_save.grid(column=3, row = 0)

        self.parent.rowconfigure(0, weight=1)
        self.parent.columnconfigure(0, weight=1)
    
    def buttons_start_pause_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.STANDBY, SupervisorState.TASK_PAUSED]):
            self.ros_node.call_service('supervisor/start', command = 'task')
        elif(self.ros_node.variables.supervisor_state == SupervisorState.TASK_RUNNING):
            if(self.ros_node.variables.supervisor_controller == ControllerType.AGENT):
                self.ros_node.call_service('supervisor/pause')
            elif(self.ros_node.variables.supervisor_controller == ControllerType.USER):
                self.ros_node.call_service('supervisor/select_controller', command = 'agent')

    def buttons_stop_trigger(self):
        self.ros_node.call_service('supervisor/stop')

    def buttons_take_over_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.TASK_PAUSED]):
            self.ros_node.call_service('supervisor/select_controller', command='user')
            self.ros_node.call_service('supervisor/start', command = 'task')
        elif(self.ros_node.variables.supervisor_state == SupervisorState.TASK_RUNNING):
            if(self.ros_node.variables.supervisor_controller == ControllerType.AGENT):
                self.ros_node.call_serivce('supervisor/select_controller', command = 'user')
            elif(self.ros_node.variables.supervisor_controller == ControllerType.USER):
                self.ros_node.call_service('supervisor/pause')

    def buttons_save_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.TASK_PAUSED, SupervisorState.TASK_RUNNING]):
            self.ros_node.call_service('supervisor/save')
            self.ros_node.variables.task_names = self.ros_node.task_handler.get_names()
    
    def update_buttons(self):
        if(self.ros_node.variables.supervisor_state == SupervisorState.STANDBY):
            self.buttons_stop['state'] = tkinter.DISABLED
            self.buttons_start_pause['state'] = tkinter.NORMAL
            self.buttons_start_pause['text'] = "start"
            self.buttons_take_over['state'] = tkinter.DISABLED
            self.buttons_save['state'] = tkinter.DISABLED
        elif(self.ros_node.variables.supervisor_state == SupervisorState.TASK_RUNNING):
            if(self.ros_node.variables.supervisor_controller == ControllerType.AGENT):
                self.buttons_stop['state'] = tkinter.NORMAL
                self.buttons_start_pause['state'] = tkinter.NORMAL
                self.buttons_start_pause['text'] = "pause"
                self.buttons_take_over['state'] = tkinter.NORMAL
                self.buttons_take_over['text'] = "take over"
                self.buttons_save['state'] = tkinter.NORMAL
            else:
                self.buttons_stop['state'] = tkinter.NORMAL
                self.buttons_start_pause['state'] = tkinter.NORMAL
                self.buttons_start_pause['text'] = "start"
                self.buttons_take_over['state'] = tkinter.NORMAL
                self.buttons_take_over['text'] = "pause"
                self.buttons_save['state'] = tkinter.NORMAL
        elif(self.ros_node.variables.supervisor_state == SupervisorState.TASK_PAUSED):
            self.buttons_stop['state'] = tkinter.NORMAL
            self.buttons_start_pause['state'] = tkinter.NORMAL
            self.buttons_start_pause['text'] = "start"
            self.buttons_take_over['state'] = tkinter.NORMAL
            self.buttons_take_over['text'] = "take over"
            self.buttons_save['state'] = tkinter.NORMAL
        elif(self.ros_node.variables.supervisor_state in [SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
            self.buttons_stop['state'] = tkinter.DISABLED
            self.buttons_start_pause['state'] = tkinter.DISABLED
            self.buttons_start_pause['text'] = "start"
            self.buttons_take_over['state'] = tkinter.DISABLED
            self.buttons_take_over['text'] = "take over"
            self.buttons_save['state'] = tkinter.DISABLED

class Demo():
    def __init__(self, parent = None, ros_node = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.ros_node = ros_node
        
        self.title = tkinter.ttk.Label(self.parent, text="Demonstration Control Panel")
        self.entry = tkinter.ttk.Combobox(self.parent)
        self.buttons_frame = tkinter.ttk.Frame(self.parent)
        self.buttons_start = tkinter.ttk.Button(self.buttons_frame, text="start", command = self.buttons_start_trigger)
        self.buttons_stop = tkinter.ttk.Button(self.buttons_frame, text="stop", command = self.buttons_stop_trigger)
        self.buttons_save = tkinter.ttk.Button(self.buttons_frame, text="save", command = self.buttons_save_trigger)

        self.title.grid(column=0, row=0)
        self.entry.grid(column = 0, row = 1)
        self.buttons_frame.grid(column=0, row =2)
        self.buttons_start.grid(column=0, row = 0)
        self.buttons_stop.grid(column=1, row = 0)
        self.buttons_save.grid(column=2, row = 0)

        self.parent.rowconfigure(0, weight=1)
        self.parent.columnconfigure(0, weight=1)
    
        try:
            self.update_entry()
        except:
            pass
    
    def buttons_start_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.STANDBY, SupervisorState.DEMO_PAUSED]):
            if(self.ros_node.variables.supervisor_state == SupervisorState.STANDBY):
                self.ros_node.call_service('supervisor/start', 'demo')
        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.ros_node.call_service('supervisor/pause')

    def buttons_stop_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
            self.ros_node.call_service('supervisor/stop')

    def buttons_save_trigger(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
            self.ros_node.call_service('supervisor/save')
            self.ros_node.variables.demo_names = self.ros_node.demo_handler.get_names()

    def update_entry(self):
        self.entry['values'] = tuple(self.ros_node.variables.demo_names)

    def update_buttons(self):
        if(self.ros_node.variables.supervisor_state == SupervisorState.STANDBY):
            self.buttons_stop['state'] = tkinter.DISABLED
            self.buttons_start_pause['state'] = tkinter.NORMAL
            self.buttons_start_pause['text'] = "start"
            self.buttons_save['state'] = tkinter.DISABLED
        elif(self.ros_node.variables.supervisor_state == SupervisorState.DEMO_RECORDING):
            self.buttons_stop['state'] = tkinter.NORMAL
            self.buttons_start_pause['state'] = tkinter.NORMAL
            self.buttons_start_pause['text'] = "pause"
            self.buttons_save['state'] = tkinter.NORMAL
        elif(self.ros_node.variables.supervisor_state == SupervisorState.DEMO_PAUSED):
            self.buttons_stop['state'] = tkinter.NORMAL
            self.buttons_start_pause['state'] = tkinter.NORMAL
            self.buttons_start_pause['text'] = "start"
            self.buttons_save['state'] = tkinter.NORMAL
        elif(self.ros_node.variables.supervisor_state in [SupervisorState.TASK_RUNNING, SupervisorState.TASK_PAUSED]):
            self.buttons_stop['state'] = tkinter.DISABLED
            self.buttons_start_pause['state'] = tkinter.DISABLED
            self.buttons_start_pause['text'] = "start"
            self.buttons_save['state'] = tkinter.DISABLED

class Model():
    def __init__(self, parent = None, ros_node = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.ros_node = ros_node

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

        try:
            self.entries_name_update()
        except:
            pass
    
    def entries_name_trigger(self, val):
        try:
            model_name = self.entries_name.get()
            self.ros_node.variables.model_names = self.ros_node.model_handler.get_ids(ModelType.ACTOR, model_name)
            self.update_entries_id(model_name)
        except:
            pass

    def select_trigger(self):
        model_name = self.entries_name.get()
        if(model_name == ''):
            return
        model_id = self.entries_id.get()
        if(model_id == ''):
            return
        model_string = json.dumps(dict(name = model_name, id = model_id))

        response = self.ros_node.call_service('agent/select_model', model_string)

        if response == True:
            self.ros_node.variables.episode_name = model_name
            self.ros_node.variables.episode_id = model_id
    
    def update_entries_name(self):
        self.entries_name['values'] = tuple(self.ros_node.variables.model_names)

    def update_entries_id(self):
        self.entries_id['values'] = tuple(self.ros_node.variables.model_ids)

    def update_buttons(self):
        if(self.ros_node.variables.supervisor_state in [SupervisorState.STANDBY, SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
            self.select['state'] = tkinter.NORMAL
        else:
            self.select['state'] = tkinter.DISABLED

class Selection():
    def __init__(self, parent = None, ros_node = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        self.ros_node = ros_node

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

        self.selected_type = InformationType.NONE
    
    def demo_box_trigger(self, event):
        demo_name = self.demo_box.get(tkinter.ANCHOR)
        try:
            self.ros_node.variables.ids = self.ros_node.demo_handler.get_ids(demo_name)
        except:
            self.ros_node.variables.ids = []
        self.selected_type = InformationType.DEMO

    def task_box_trigger(self, event):
        task_name = self.task_box.get(tkinter.ANCHOR)
        try:
            self.ros_node.variables.ids = self.ros_node.task_handler.get_ids(task_name)
        except:
            self.ros_node.variables.ids = []
        self.selected_type = InformationType.TASK_EPISODE

    def id_box_trigger(self, event):
        if(self.selected_type == InformationType.DEMO):
            demo_name = self.demo_box.get(tkinter.ANCHOR)
            demo_id = self.id_box.get(tkinter.ANCHOR)
            if(demo_id == '' or demo_id == None):
                return
            if(self.ros_node.variables.supervisor_state == SupervisorState.STANDBY):
                episode_event = dict(function = self.ros_node.update_episode, kwargs = dict(type = InformationType.DEMO, name = demo_name, id = demo_id))
                self.ros_node.episode_event_queue.put(episode_event)

        elif(self.selected_type == InformationType.TASK_EPISODE):
            demo_name = self.saved_task_episode_name_list.get(tkinter.ANCHOR)
            task_id = self.id_box.get(tkinter.ANCHOR)
            if(task_id == '' or task_id == None):
                return
            if(self.ros_node.variables.supervisor_state == SupervisorState.STANDBY):
                episode_event = dict(function = self.ros_node.update_episode, kwargs = dict(type = InformationType.TASK_EPISODE, name = demo_name, id = task_id))
                self.ros_node.episode_event_queue.put(episode_event)
    
    def add_trigger(self):
        if(self.ros_node.variables.episode_type == InformationType.DEMO):
            demo_name = self.demo_box.get(tkinter.ANCHOR)
            demo_id = self.id_box.get(tkinter.ANCHOR)
            if(demo_id == "" or demo_name == ""):
                return
            demo = demo_name+"."+str(demo_id)
            self.queue_box.insert(tkinter.END, demo)
            self.ros_node.variables.task_queue.append(demo)

    def remove_trigger(self):
        selected_index = self.queue_box.curselection()
        if(0 in selected_index and self.ros_node.variables.supervisor_state not in [SupervisorState.STANDBY, SupervisorState.DEMO_PAUSED, SupervisorState.DEMO_RECORDING]):
            return
        self.queue_box.delete(tkinter.ANCHOR)
        self.ros_node.variables.task_queue.pop(selected_index)

    def update_id(self):
        self.id_box.delete(0,tkinter.END)
        for id_ in self.ros_node.variables.ids:
            self.id_box.insert(tkinter.END, id_)

    def update_demo(self):
        self.demo_box.delete(0,tkinter.END)
        for id_ in self.ros_node.variables.demo_names:
            self.demo_box.insert(tkinter.END, id_)

    def update_task(self):
        self.task_box.delete(0,tkinter.end)
        for id_ in self.ros_node.variables.task_names:
            self.task_box.insert(tkinter.end, id_)
    
    def update_queue(self):
        self.queue_box.delete(0,tkinter.end)
        for id_ in self.ros_node.variables.task_queue:
            self.queue_box.insert(tkinter.end, id_)