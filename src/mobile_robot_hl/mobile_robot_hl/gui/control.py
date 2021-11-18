import tkinter
from tkinter import ttk

from mobile_robot_hl.gui.utils import *
from mobile_robot_hl.model import *
from mobile_robot_hl.utils import *

#==============================================================

class Control(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)
        
        self.demo_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.demo = Demo(parent = self.demo_frame)
        self.model_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.model = Model(parent = self.model_frame)
        self.task_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.task = Task(parent= self.task_frame)
        self.selection_frame = tkinter.ttk.Frame(self.parent, borderwidth=2, relief=tkinter.SUNKEN,padding="10 10 10 10")
        self.selection = Selection(parent = self.selection_frame)

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

class Task(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

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
        if(self.ros_node.state in [SupervisorState.TASK_PAUSED, SupervisorState.STANDBY, SupervisorState.TASK_TAKE_OVER]):
            response = self.ros_node.call_service('supervisor/start', 'demo')
        elif(self.state == SupervisorState.TASK_RUNNING):
            self.state = SupervisorState.TASK_PAUSED
            self.automatic_start_button.configure(text="start")
            self.ros_node.call_service('agent/pause')
            print("[INFO] Automatic control paused")

        self.ros_node.update_state(self.state)

    def buttons_stop_trigger(self):
        pass

    def buttons_take_over_trigger(self):
        pass

    def buttons_save_trigger(self):
        pass

    def update_state(self):
        raise NotImplementedError()



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

















class Demo(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent
        
        super().__init__(self.parent)

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
        if(self.ros_node.state == SupervisorState.STANDBY or self.ros_node.state == SupervisorState.DEMO_PAUSED):
            if(self.ros_node.state == SupervisorState.STANDBY):
                self.ros_node.call_service('supervisor/start', 'demo')
                self.ros_node.episode.reset()
        elif(self.state == SupervisorState.DEMO_RECORDING):
            self.ros_node.call_service('supervisor/pause')

    def buttons_stop_trigger(self):
        if(self.ros_node.state in [SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
            self.ros_node.call_service('supervisor/stop')
            self.ros_node.episode.reset()

    def buttons_save_trigger(self):
        if(self.ros_node.state in [SupervisorState.DEMO_RECORDING, SupervisorState.DEMO_PAUSED]):
            self.ros_node.call_service('supervisor/save')

    def update_entry(self):
        name_array = self.ros_node.demo_handler.get_names()
        self.entry.delete(0, tkinter.END)
        for name in name_array:
            self.entry.insert(0, name)
        self.entry['values'] = tuple(name_array)

class Model(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

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
            self.update_entries_id(model_name)
        except:
            pass
    
    def update_entries_name(self):
        name_array = self.ros_node.model_handler.get_names(ModelType.ACTOR)
        self.entries_name['values'] = tuple(name_array)

    def update_entries_id(self, model_name):
        id_array = self.ros_node.model_handler.get_ids(ModelType.ACTOR, model_name)
        self.entries_id['values'] = tuple(id_array)

    def select_trigger(self):
        model_name = self.entries_name.get()
        if(model_name == ''):
            return
        model_id = self.entries_id.get()
        if(model_id == ''):
            return
        model_string = f"{model_name}/{model_id}"

        response = self.ros_node.call_service('agent/select_model', model_string)

        if response.success == True:
            self.ros_node.episode_name = model_name
            self.ros_node.episode_id = model_id
            self.ros_node.update_info()

class Selection(ROSWidget):
    def __init__(self, parent = None):
        if parent == None:
            self.parent = tkinter.Tk()
        else:
            self.parent = parent

        super().__init__(self.parent)

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