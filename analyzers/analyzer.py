import numpy as np
import tools
import os
import importlib
from os.path import join as osjoin
from tools import logger as logger
from .container import DataContainer
from datasets.abstract_dataset import AbstractDataset
import multiprocessing as mp

def run_analyzer(unique_idx:int, queue:mp.Queue, analyzer, params, container):
    try:
        analyzer(params, container).run()
    except Exception as e: # if exception happened
        queue.put((unique_idx, e))
        return
    queue.put((unique_idx, None))
    return

class Analyzer:
    def __init__(self, name_list:list) -> None:
        '''
        name_list: 启动脚本, 否则需要手动run_sub_analyzer, 可以是None
        '''
        self.container = DataContainer()
        self.analyzer_dict = self.auto_import_analyzers()
        self.dataset_dict = {
            "default": AbstractDataset
        }
        mp_launch = tools.GLOBAL_CONF['mp_launch']
        if mp_launch:
            logger.info('Enable multiprocessed launch')
            self.multiprocess_init(len(name_list))
        else:
            logger.info('Enable Sequential launch')
            self.launch_list = []
        for name in name_list:
            for label in self.analyzer_dict.keys():
                if label in name: # analyzer@version
                    container = DataContainer()
                    if label in self.dataset_dict.keys():
                        container.register_dataset(self.dataset_dict[label])
                    else:
                        container.register_dataset(self.dataset_dict["default"])
                    self.register_analyzer(name, label, container)
                    break
        if mp_launch:
            self.multiprocess_launch()
        else:
            self.sequential_launch()

    def auto_import_analyzers(self):
        result = {}
        for filename in os.listdir('analyzers'):
            if filename.endswith('_analyzer.py'):
                prefix = filename.split('.')[0]
                module_name = filename.split('_analyzer')[0].capitalize() + 'Analyzer' # for example: abc_analyzer.py -> AbcAnalyzer
                result[module_name] = getattr(importlib.import_module('.' + prefix, 'analyzers'), module_name)
        return result
    
    def sequential_launch(self):
        for launch_dict in self.launch_list:
            analyzer = launch_dict['analzer']
            params = launch_dict['params']
            container = launch_dict['container']
            analyzer(params, container).run()

    def multiprocess_init(self, n_max):
        self.launch_list = []
        self.register_devices = {}
        self.mp_queue = mp.Queue(maxsize=n_max)

    def multiprocess_launch(self):
        '''A simple scheduler for multiprocessing
        '''
        unique_idx = 0
        device_stats = np.asarray([True for _ in self.container.available_devices])
        p_container = []
        while(len(self.launch_list) > 0):
            launch_idx = []
            for l_idx, launch_dict in enumerate(self.launch_list):
                # find a suitable analyzer
                n_devices = launch_dict['n_devices']
                if n_devices <= sum(device_stats):
                    analyzer = launch_dict['analyzer']
                    params = launch_dict['params'].copy()
                    available_idx = [idx for idx,stat in enumerate(device_stats) if stat][:n_devices]
                    launch_dict['container'].available_devices = \
                        [d for i,d in enumerate(self.container.available_devices.copy()) if i in available_idx] # register devices
                    logger.info(f'Launch {analyzer} with unique_idx={unique_idx}')
                    p = mp.Process(target=run_analyzer, args=(unique_idx, self.mp_queue, analyzer, params, launch_dict['container']))
                    p_container.append(p)
                    # after launch
                    launch_idx.append(l_idx)
                    device_stats[available_idx] = False # update device availability
                    self.register_devices[unique_idx] = available_idx.copy()
                    unique_idx += 1
                    p.start()
            
            if len(launch_idx) > 0:
                self.launch_list = [d for i,d in enumerate(self.launch_list) if i not in launch_idx] # delete launched analyzers from launch list
                p_idx, e = self.mp_queue.get(block=True)
                if e is not None:
                    logger.error(f'Process {p_idx} cause Exception: {e}')
                device_stats[self.register_devices[p_idx]] = True # update device availability
            elif len(device_stats) == sum(device_stats): # no launch and can not launch rest analyzers
                logger.error(f'Some analyzers need more devices than available:{self.launch_list}')
                return
        # all analyzers launched, make sure main process will not exit before everything done
        logger.info('Multiprocesses: All analyzers launched')
        for p in p_container:
            if p.is_alive():
                p.join()
        logger.info('Multiprocesses: All processes done.')


    def register_analyzer(self, analyzer_name:str, label:str, container:DataContainer):
        '''use multiprocessing to avoid core process dump
        '''
        logger.info(f'Register Analyzer: {analyzer_name}')
        params = self.container.get_analyzer_params(analyzer_name)
        params['analyzer_name'] = analyzer_name
        sub_analyzer = self.analyzer_dict[label]
        self.launch_list.append({
            'analyzer': sub_analyzer, 
            'container': container,
            'params': params,
            'n_devices': params['n_devices']
        })
        # utils.create_final_result()
