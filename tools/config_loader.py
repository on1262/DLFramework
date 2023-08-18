import json
import os

class ConfigLoader():
    '''
    ConfigLoader: load and process global configs from a single json file
        work directory will be appended if 'work_dir' in the keys of dict
    '''
    def __init__(self, glob_conf_path:str) -> None:
        self.glob_conf_path = glob_conf_path
        with open(self.glob_conf_path, 'r',encoding='utf-8') as fp:
            self.glob_conf = json.load(fp)
        # append work_dir to every values under 'paths'
        if 'work_dir' in self.glob_conf.keys():
            self.process_path(self.glob_conf['work_dir'], self.glob_conf)
        
    def process_path(self, root, conf_dict):
        for key in conf_dict.keys():
            if key == 'paths':
                for k2 in conf_dict[key].keys():
                    conf_dict[key][k2] = os.path.join(root, conf_dict[key][k2])
            if isinstance(conf_dict[key], dict):
                self.process_path(root, conf_dict[key])

    def __getitem__(self, __key: str):
        return self.glob_conf[__key]
    

class Config:
    '''
        加载配置表
        cache_path: 自动配置
        manual_path: 手动配置
    '''
    def __init__(self, cache_path, manual_path) -> None:
        self.cache_path = cache_path
        self.manual_path = manual_path
        self.configs = {}
        with open(manual_path, 'r', encoding='utf-8') as fp:
            manual_conf = json.load(fp)
        if os.path.exists(cache_path):
            with open(cache_path, 'r', encoding='utf-8') as fp:
                self.configs = json.load(fp)
        for key in manual_conf.keys(): # 覆盖
            self.configs[key] = manual_conf[key]

    def __getitem__(self, idx):
        return self.configs.copy()[idx]
    
    def dump(self):
        with open(self.manual_path, 'w', encoding='utf-8') as fp:
            json.dump(self.configs, fp)

GLOBAL_CONF_LOADER = ConfigLoader('./configs/global_config.json')
        
        