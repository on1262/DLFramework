import torch
import torchinfo
from datasets.abstract_dataset import AbstractDataset
import tools
from tqdm import tqdm
from tools.colorful_logging import logger
import numpy as np


class TrainerTemplate():
    '''A template trainer'''
    def __init__(params:dict):
        self.params = params
        self.model = None
        self.dataset = None
        
        self.loss_logger = tools.LossLogger()
        self.registered_loss = {
            'train_out': [],
            'test_out': [],
        }
        pass
    
    def run_epochs(self, model, dataset:AbstractDataset, n_epochs, start_epoch=0, verbose=1):
        for epoch_idx in tqdm(range(start_epoch, start_epoch+n_epochs, 1)):
            train_loss_dict = self.run_one_epoch(epoch_idx, model, dataset, 'train', verbose=verbose)
            valid_loss_dict = self.run_one_epoch(epoch_idx, model, dataset, 'valid', verbose=verbose)
            # add loss dict to registered loss
            

    
    def run_one_epoch(self, epoch_idx, model, dataset:AbstractDataset, mode:str, verbose=2):
        # init registered loss
        reg_loss = {}
        for key in self.registered_loss.keys():
            if mode in key:
                reg_loss[key] = 0
        
        if mode != 'test':
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        dataloader = tools.FlexibleLoader(dataset=dataset, batch_size=self.params['batchsize'], sampler=sampler, collate_fn=None)

        sample_count = 0
        for batch_idx, data in enumerate(dataloader):
            sample_count += data['n_sample']
            # model inference
            loss_dict = self.model.forward(data)
            # registered loss
            for key in loss_dict:
                if key in reg_loss.keys():
                    reg_loss[key] += loss_dict[key] * data['n_sample']
            # print loss
            if verbose > 1 and sample_count * 5 % len(dataset) == 0 and sample_count > 0:
                loss_str = ', '.join([str(key) + '=%.5g' % val / sample_count for key,val in reg_loss.items()])
                logger.info(f"Mode: {mode}, sample={sample_count}/{len(dataset)}, loss={loss_str}")
        
        return reg_loss

    





