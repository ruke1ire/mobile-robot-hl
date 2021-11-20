import os
import yaml
import glob
import shutil
from PIL import Image as PImage

from mobile_robot_hl.utils import ControllerType
from mobile_robot_hl.episode_data.episode_data import EpisodeData

class DemoHandler():
    DEMO_ID_INFO_FILE= "info.yaml"
    def __init__(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        '''
        Return the demonstration with DataStructure dict_list
        '''
        try:
            with open(f"{self.path}/{name}/{id_}/{DemoHandler.DEMO_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception(f'Invalid demo name: {name} ID: {id_}')

        images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]

        info['observation']['image'] = images
        del info['observation']['image_id']
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        info['action']['controller'] = task_controller

        episode_data = EpisodeData(**info)
        return episode_data
    
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
        demo: demonstration data with DataStructure = list_dict
        name: name of demonstration
        id_: id of demonstration
        '''

        demo_copy = EpisodeData(**demo.get())

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

        dict_data = demo_copy.get()

        image_ids = list(range(demo_copy.length()))
        for i in image_ids:
            img = dict_data['observation']['image'][i]
            img.save(f"{self.path}/{name}/{next_id}/{i}.png")
        
        del dict_data['observation']['image']
        dict_data['observation']['image_id'] = image_ids
        dict_data['action']['controller'] = [cont.name for cont in dict_data['action']['controller']]

        with open(f'{self.path}/{name}/{next_id}/{DemoHandler.DEMO_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(dict_data, outfile)

class TaskHandler():
    TASK_ID_INFO_FILE= "info.yaml"
    def __init__(self, path, demo_handler):
        self.path = path
        self.demo_handler = demo_handler
        os.makedirs(self.path, exist_ok = True)

    def get(self, name, id_):
        try:
            with open(f"{self.path}/{name}/{id_}/{TaskHandler.TASK_ID_INFO_FILE}", 'r') as stream:
                info = yaml.safe_load(stream)
        except:
            raise Exception(f'Invalid task name: {name} ID: {id_}')

        demo_id = info['demonstration']

        demo_episode_data = self.demo_handler.get(name, demo_id)

        task_images = [PImage.open(f"{self.path}/{name}/{id_}/{image_id}.png") 
                    for image_id in info['observation']['image_id']]
        task_controller = [ControllerType[cont] for cont in info['action']['controller']]
        
        info['observation']['image'] = task_images
        info['action']['controller'] = task_controller
        del info['observation']['image_id']
        del info['demonstration']

        task_episode_data = EpisodeData(**info)
        demo_episode_data.append(task_episode_data)

        return demo_episode_data
    
    def get_names(self):
        '''
        Get all available task names
        '''
        demos = [os.path.basename(x) for x in glob.glob(self.path+"/*")]
        return demos
    
    def get_ids(self, name):
        '''
        Get all available task ids
        '''
        task_episodes = [int(os.path.basename(x)) for x in glob.glob(self.path+"/"+name+"/*") if "yaml" not in os.path.basename(x)]
        return task_episodes

    def save(self, episode, demo_name, demo_id, task_id = None):
        '''
        episode: DataStructure = LIST_DICT
        demo_name: name of demonstration
        demo_id: id of demonstration
        task_id: id of task episode

        storing DataStructure = DICT_LIST
        '''

        if not os.path.exists(f"{self.path}/{demo_name}"):
            os.mkdir(f"{self.path}/{demo_name}")

        if(task_id == None):
            try:
                taken_id = max([x for x in self.get_ids(demo_name) if type(x) == int])
                next_id = taken_id + 1
            except:
                next_id = 0
        else:
            next_id = task_id

        if os.path.exists(f"{self.path}/{demo_name}/{next_id}"):
            shutil.rmtree(f"{self.path}/{demo_name}/{next_id}", ignore_errors=True)
        os.mkdir(f"{self.path}/{demo_name}/{next_id}")

        episode_copy = EpisodeData(**episode.get())

        task_index = episode_copy.action.controller.get().index(ControllerType.AGENT)
        episode_copy.remove(task_index-1, leftwards=True)

        dict_data = episode_copy.get()

        image_ids = list(range(episode_copy.length()))
        for i in image_ids:
            img = dict_data['observation']['image'][i]
            img.save(f"{self.path}/{demo_name}/{next_id}/{i}.png")
        
        del dict_data['observation']['image']
        dict_data['observation']['image_id'] = image_ids
        dict_data['action']['controller'] = [cont.name for cont in dict_data['action']['controller']]
        dict_data['demonstration'] = demo_id

        with open(f'{self.path}/{demo_name}/{next_id}/{TaskHandler.TASK_ID_INFO_FILE}', 'w') as outfile:
            yaml.dump(dict_data, outfile)