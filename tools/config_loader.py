import yaml
import os

class ConfigLoader():
    '''
    ConfigLoader: load and process global configs from a single json file
        work directory will be appended if 'work_dir' in the keys of dict
    '''
    def __init__(self, glob_conf_path:str) -> None:
        self.glob_conf_path = glob_conf_path
        with open(self.glob_conf_path, 'r', encoding='utf-8') as fp:
            self.glob_conf = yaml.load(fp, Loader=yaml.SafeLoader)
        # append work_dir to every values under 'paths'
        if 'work_dir' in self.glob_conf.keys():
            self.process_path(self.glob_conf['work_dir'], self.glob_conf)
        # load subfolder as dict
        for dir_name in os.listdir('./configs'):
            if os.path.isdir(os.path.join('./configs', dir_name)):
                self.glob_conf[dir_name] = {}
                for file in os.listdir(os.path.join('./configs', dir_name)):
                    with open(os.path.join('./configs', dir_name, file), 'r', encoding='utf-8') as fp:
                        self.glob_conf[dir_name][file.split('.')[0]] = yaml.load(fp, Loader=yaml.SafeLoader)
    
    def process_path(self, root, conf_dict):
        for key in conf_dict.keys():
            if key == 'paths':
                for k2 in conf_dict[key].keys():
                    conf_dict[key][k2] = os.path.join(root, conf_dict[key][k2])
            if isinstance(conf_dict[key], dict):
                self.process_path(root, conf_dict[key])

    def __getitem__(self, __key: str):
        return self.glob_conf[__key]

GLOBAL_CONF = ConfigLoader('./configs/global.yml')
GLOBAL_PATH = GLOBAL_CONF['paths']


        
        