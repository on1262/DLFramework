import numpy as np
import tools
import os
from tools import logger as logger
from .container import DataContainer
from datasets.abstract_dataset import AbstractDataset


class Analyzer:
    def __init__(self, name_list:list, dataset:AbstractDataset) -> None:
        '''
        params: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        '''
        self.container = DataContainer()
        self.analyzer_dict = {
            "TemplateAnalyzer": None
        }
        self.dataset_dict = {
            "default": AbstractDataset
        }
        for name in name_list:
            for label in self.analyzer_dict.keys():
                if label in name: # analyzer@version
                    if label in self.dataset_dict.keys():
                        self.container.register_dataset(self.dataset_dict[label])
                    else:
                        self.container.register_dataset(self.dataset_dict["default"])
                    self.run_sub_analyzer(name, label)
                    break
        
    def run_sub_analyzer(self, analyzer_name, label):
        logger.info(f'Run Analyzer: {analyzer_name}')
        params = self.container.get_analyzer_params(analyzer_name)
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[label](params, self.container)
        sub_analyzer.run()
        # utils.create_final_result()