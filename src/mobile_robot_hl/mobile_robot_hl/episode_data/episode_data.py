from mobile_robot_hl.utils import ControllerType

import copy

class EpisodeData:
    def __init__(self, data):
        if data == None:
            self.init_empty_structure()
        else:
            # TODO: error if data is just an empty structure
            self.data = copy.deepcopy(data)
            self.length = self.get_episode_length_()
            self.data_empty = False

    def init_empty_structure(self):
        self.data = dict(
                        observation=dict(image=[None]),
                        action=dict(
                            agent=dict(
                                velocity=dict(
                                    linear=[None],
                                    angular=[None]
                                ),
                                termination_flag = [None]
                            ),
                            user=dict(
                                velocity=dict(
                                    linear=[None],
                                    angular=[None]
                                ),
                                termination_flag = [None]
                            ),
                            controller=[None]))
        self.data_empty = True
        self.length = 0
    
    def append_episode_data(self, episode):
        if(episode.data_empty == True):
            return
        else:
            self.data['observation']['image'] += episode.data['observation']['image']
            self.data['action']['agent']['velocity']['linear'] += episode.data['action']['agent']['velocity']['linear']
            self.data['action']['agent']['velocity']['angular'] += episode.data['action']['agent']['velocity']['angular']
            self.data['action']['agent']['termination_flag'] += episode.data['action']['agent']['termination_flag']
            self.data['action']['user']['velocity']['linear'] += episode.data['action']['user']['velocity']['linear']
            self.data['action']['user']['velocity']['angular'] += episode.data['action']['user']['velocity']['angular']
            self.data['action']['user']['termination_flag'] += episode.data['action']['user']['termination_flag']
            self.data['action']['controller'] += episode.data['action']['controller']
            self.data_empty = False

            self.length += episode.length

    def append_data(
        self, 
        image, 
        agent_linear_vel, agent_angular_vel, agent_termination_flag,
        user_linear_vel, user_angular_vel, user_termination_flag,
        controller):
        if(self.data_empty == True):
            self.data['observation']['image'] = [image]
            self.data['action']['agent']['velocity']['linear'] = [agent_linear_vel]
            self.data['action']['agent']['velocity']['angular'] = [agent_angular_vel]
            self.data['action']['agent']['termination_flag'] = [agent_termination_flag]
            self.data['action']['user']['velocity']['linear'] = [user_linear_vel]
            self.data['action']['user']['velocity']['angular'] = [user_angular_vel]
            self.data['action']['user']['termination_flag'] = [user_termination_flag]
            self.data['action']['controller'] = [controller]
            self.data_empty = False
        else:
            self.data['observation']['image'].append(image)
            self.data['action']['agent']['velocity']['linear'].append(agent_linear_vel)
            self.data['action']['agent']['velocity']['angular'].append(agent_angular_vel)
            self.data['action']['agent']['termination_flag'].append(agent_termination_flag)
            self.data['action']['user']['velocity']['linear'].append(user_linear_vel)
            self.data['action']['user']['velocity']['angular'].append(user_angular_vel)
            self.data['action']['user']['termination_flag'].append(user_termination_flag)
            self.data['action']['controller'].append(controller)

        self.length += 1

    def set_key_value(self, key, value, index = None):
        '''
        key = string of keys separated by "."
        Eg. "action.agent.velocity.linear"
        '''
        key_split = key.split('.')
        dict_string = ''
        for k in key_split:
            dict_string += "['"
            dict_string += k
            dict_string += "']"
        if type(index) is not int:
            exec(f"self.data{dict_string} = {value}")
        else:
            exec(f"self.data{dict_string}[{index}] = {value}")

    def set_data(
        self, 
        index,
        image=None, 
        agent_linear_vel=None, agent_angular_vel=None, agent_termination_flag=None,
        user_linear_vel=None, user_angular_vel=None, user_termination_flag=None,
        controller=ControllerType.NONE):
        if(self.data_empty == False):
            assert index <= self.get_episode_length() - 1
            self.data['observation']['image'][index] = image
            self.data['action']['agent']['velocity']['linear'][index] = agent_linear_vel
            self.data['action']['agent']['velocity']['angular'][index] = agent_angular_vel
            self.data['action']['agent']['termination_flag'][index] = agent_termination_flag
            self.data['action']['user']['velocity']['linear'][index] = user_linear_vel
            self.data['action']['user']['velocity']['angular'][index] = user_angular_vel
            self.data['action']['user']['termination_flag'][index] = user_termination_flag
            self.data['action']['controller'][index] = controller
        else:
            self.append_data(image, agent_linear_vel, agent_angular_vel, agent_termination_flag,
            user_linear_vel, user_angular_vel, user_termination_flag,
            controller)

    def get_episode_length_(self):
        return len(self.data['observation']['image'])

    def get_episode_length(self):
        return self.length
    
    def get_data(self, index = None):
        if(type(index) is not int):
            return self.data
        else:
            if(self.data_empty == False):
                return dict(
                    observation = dict(
                        image = self.data['observation']['image'][index]
                    ),
                    action = dict(
                        controller = self.data['action']['controller'][index],
                        agent = dict(
                            velocity = dict(
                                linear = self.data['action']['agent']['velocity']['linear'][index],
                                angular = self.data['action']['agent']['velocity']['angular'][index]
                            ),
                            termination_flag = self.data['action']['agent']['termination_flag']
                        ),
                        user = dict(
                            velocity = dict(
                                linear = self.data['action']['user']['velocity']['linear'][index],
                                angular = self.data['action']['user']['velocity']['angular'][index]
                            ),
                            termination_flag = self.data['action']['user']['termination_flag']
                        ),
                    )
                )
            else:
                return dict(
                    observation = dict(
                        image = None
                    ),
                    action = dict(
                        controller = None,
                        agent = dict(
                            velocity = dict(
                                linear = None,
                                angular = None
                            ),
                            termination_flag = None,
                        ),
                        user = dict(
                            velocity = dict(
                                linear = None,
                                angular = None
                            ),
                            termination_flag = None
                        ),
                    )
                )

    def remove_data(self, index, leftwards=True):
        list_strings = get_leaf_string(self.data)
        for s in list_strings:
            if leftwards == True:
                exec(f"self.data{s} = self.data{s}[{index+1}:]")
            else:
                exec(f"self.data{s} = self.data{s}[:{index}]")
        
        self.length = self.get_episode_length_()
        if self.length == 0:
            self.data_empty = True

def get_leaf_string(dict_, string = ""):
    try:
        for key in dict_.keys():
            s = f"['{key}']"
            ss = string + s
            yield from get_leaf_string(dict_[key], ss)
    except:
        yield string