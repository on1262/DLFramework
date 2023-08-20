from datasets.abstract_dataset import AbstractDataset
import tools
from tools import logger
import torch

class DataContainer():
    '''存放数据和一些与模型无关的内容'''
    def __init__(self):
        self._conf = tools.GLOBAL_CONF['analyzer']['data_container'] # 这部分是global, 对外界不可见
        self.n_fold = self._conf['n_fold']
        self.seed = self._conf['seed']
        self.devices = [torch.device(t) for t in tools.GLOBAL_CONF['devices']]
        self.dataset = None

    def register_dataset(self, dataset_type:type):
        if not type(self.dataset) == dataset_type:
            logger.info(f'Register dataset: {dataset_type}')
            self.dataset = dataset_type
        else:
            logger.warning(f'DataContainer will keep dataset instance: {dataset_type}')
    
    def get_analyzer_params(self, analyzer_name) -> dict:
        '''根据数据集和模型名不同, 获取所需的模型参数'''
        params = tools.GLOBAL_CONF['analyzers'][analyzer_name]
        return params.copy()