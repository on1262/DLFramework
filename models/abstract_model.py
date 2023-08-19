import torch
import yaml

class AbstractModel(torch.nn.Module):

    __name = 'AbstractModel'

    def __init__(self):
        super().__init__()
        self.model_conf = self.load_conf()

    def __call__(self, x):
        raise NotImplementedError
    
    def name(self):
        return self.__name
    
    def load_conf(self):
        with open('./model_config.yml', 'r',encoding='utf-8') as fp:
            conf = yaml.load(fp, Loader=yaml.SafeLoader)
        return conf
    


