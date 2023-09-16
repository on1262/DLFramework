from datasets.abstract_dataset import AbstractDataset
import tools
from tools import logger
import torch

class DataContainer():
    '''存放数据和一些与模型无关的内容, 实例在多线程中会被多次初始化
    '''
    def __init__(self):
        self._conf = tools.GLOBAL_CONF['data_container'] # 这部分是global, 对外界不可见
        self.n_fold = self._conf['n_fold']
        self.seed = self._conf['seed']
        self.available_devices = [torch.device(t) for t in tools.GLOBAL_CONF['available_devices']]
        self.dataset = None

    def register_dataset(self, dataset_type:type):
        if not type(self.dataset) == dataset_type:
            logger.info(f'Register dataset: {dataset_type}')
            self.dataset = dataset_type
        else:
            logger.warning(f'DataContainer will keep dataset instance: {dataset_type}')
    
    def get_analyzer_params(self, analyzer_name:str) -> dict:
        '''根据数据集和模型名不同, 获取所需的模型参数'''
        if '@' in analyzer_name:
            params = tools.GLOBAL_CONF['analyzers'][analyzer_name.split('@')[0]]
        else:
            params = tools.GLOBAL_CONF['analyzers'][analyzer_name]
        # check devices
        assert('n_devices' in params)
        return params.copy()