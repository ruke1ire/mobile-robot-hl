import os
import glob
import yaml
import shutil
from PIL import Image as PImage

class DemoHandler():
    DEMO_NAME_INFO_FILE = "info.yaml"
    DEMO_ID_INFO_FILE= "info.yaml"
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        '''
        Return the array of images, actions, etc.
        '''
        try:
            with open(f"{self.path}/{name}/{id_}/{DemoHandler.DEMO_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception("Invalid demo name and id! Name: {name} ID: {id_}")

        velocity = info['actions']['velocity']
        termination_flag = info['actions']['termination_flag']
        images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observations']['image_id']]
        return images, velocity, termination_flag
    
    def get_names(self):
        '''
        Get all available demo names
        '''
        demos = [os.path.basename(x) for x in glob.glob(self.path+"/*")]
        return demos
    
    def get_ids(self, name):
        '''
        Get all available demo ids
        '''
        demos = [int(os.path.basename(x)) for x in glob.glob(self.path+"/"+name+"/*") if "yaml" not in os.path.basename(x)]
        return demos

    def save(self, demo, name, id_=None):
        '''
        Save a demonstration => dict(observation=dict(image_id), actions=dict(velocity, termination_flag))

        demo: list(dict(image, velocity, termination_flag))
        name: name of demonstration
        id_: id of demonstration
        '''
        if not os.path.exists(f"{self.path}/{name}"):
            os.mkdir(f"{self.path}/{name}")

        if(id_ == None):
            try:
                taken_id = max([x for x in self.get_ids(name) if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0
        else:
            next_id = id_

        if os.path.exists(f"{self.path}/{name}/{next_id}"):
            shutil.rmtree(f"{self.path}/{name}/{next_id}", ignore_errors=True)
        os.mkdir(f"{self.path}/{name}/{next_id}")

        demo = demo[:-1]

        image_ids = list(range(len(demo)))
        for i in image_ids:
            img = PImage.fromarray(demo[i]["image"])
            img.save(f"{self.path}/{name}/{next_id}/{i}.png")
        
        velocity = [data['velocity'] for data in demo]
        termination_flag = [data['termination_flag'] for data in demo]
        demo_dict = {
                        'observations':{
                            'image_id':image_ids
                        },
                        'actions':{
                            'velocity': velocity,
                            'termination_flag': termination_flag
                        }
                    }
        
        with open(f'{self.path}/{name}/{next_id}/{DemoHandler.DEMO_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(demo_dict, outfile)

class TaskHandler():
    TASK_NAME_INFO_FILE = "info.yaml"
    TASK_ID_INFO_FILE= "info.yaml"
    def __init__(self, path, demo_handler):
        self.path = path
        self.demo_hander = demo_handler
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        '''
        Return the array of images, actions, etc.
        '''

        try:
            with open(f"{self.path}/{name}/{id_}/{TaskHandler.TASK_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception("Invalid task name and id! Name: {name} ID: {id_}")
        demo_id = info['demo_id']

        demo_images, demo_velocity, demo_termination_flag = self.demo_handler.get(name, demo_id)

        task_velocity = info['actions']['velocity']
        task_termination_flag = info['actions']['termination_flag']
        task_controller = info['actions']['controller']
        task_images = [PImage.open(self.path+"/"+name+"/"+id_+"/"+image_id+".png") 
                    for image_id in info['observations']['image_id']]
        
        velocity = demo_velocity+task_velocity
        termination_flag = demo_termination_flag + task_termination_flag
        images = demo_images + task_images

        return images, velocity, termination_flag, task_controller
    
    def get_names(self):
        '''
        Get all available task names
        '''
        demos = [os.basename(x) for x in glob.glob(self.path+"/*")]
        return demos
    
    def get_ids(self, name):
        '''
        Get all available task ids
        '''
        task_episodes = [int(os.path.basename(x)) for x in glob.glob(self.path+"/"+name+"/*") if "yaml" not in os.path.basename(x)]
        return task_episodes

    def save(self, episode, demo_name, demo_id, task_id = None):
        '''
        Save a task episode => dict(observation=dict(image_id), actions=dict(velocity, termination_flag, controller), demonstration)

        episode: array of dict(image, velocity, termination_flag, controller)
        demo_name: name of demonstration
        demo_id: id of demonstration
        task_id: id of task episode
        '''

        if not os.path.exists(f"self.path/{demo_name}"):
            os.mkdir(f"{self.path}/{demo_name}")

        if(task_id == None):
            try:
                taken_id = max([x for x in self.get_ids() if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0
        else:
            next_id = task_id

        image_ids = list(range(len(episode)))
        for i in image_ids:
            img = PImage.fromarray(episode[i]["image"])
            img.save(f"{self.task_path}/{demo_name}/{next_id}/{i}.png")
        
        velocity = [data['velocity'] for data in episode]
        termination_flag = [data['termination_flag'] for data in episode]
        controller = [data['controller'] for data in episode]
        task_episode_dict = {
                        'observations':{
                            'image_id':image_ids
                        },
                        'actions':{
                            'velocity': velocity,
                            'termination_flag': termination_flag,
                            'controller': controller
                        },
                        'demonstration': demo_id
                    }
        
        with open('task_info_info.yaml', 'w') as outfile:
            yaml.dump(task_episode_dict, outfile)

class ModelHandler():
    def __init__(self):
        pass