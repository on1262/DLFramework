import torch
from torch.utils.data import RandomSampler, SequentialSampler
import torchinfo
import datetime
from datasets.abstract_dataset import AbstractDataset
from models.abstract_model import AbstractModel
import tools
from tqdm import tqdm
from tools.colorful_logging import logger
from tools import GLOBAL_PATH, GLOBAL_CONF
import numpy as np
import datetime
import os

class TrainerTemplate():
    '''A template trainer'''
    def __init__(self, params:dict):
        self.params = params

        self.analyzer_name = params['analyzer_name']
        self.model_weights_dir = os.path.join(GLOBAL_PATH['saved_model'])
        if not os.path.exists(self.model_weights_dir):
            tools.reinit_dir(self.model_weights_dir, build=True)
        self.model = None
        self.opt_list = None
        self.dataset = None
        
        self.loss_logger = tools.LossLogger()
        self.registered_loss = { # mode@reg_name
            'train@out_loss': [],
            'test@out_loss': [],
        }
    
    def run_epochs(self, model, dataset:AbstractDataset, n_epochs, start_epoch=0, verbose=1):
        tq = tqdm(total=n_epochs)
        for epoch_idx in range(start_epoch, start_epoch+n_epochs, 1):
            train_loss_dict = self.run_one_epoch(epoch_idx, model, dataset, 'train', verbose=verbose)
            valid_loss_dict = self.run_one_epoch(epoch_idx, model, dataset, 'valid', verbose=verbose)
            # add loss dict to registered loss
            for key in self.registered_loss:
                if key in train_loss_dict:
                    self.registered_loss[key].append(train_loss_dict[key])
                if key in valid_loss_dict:
                    self.registered_loss[key].append(valid_loss_dict[key])
            # save model
            self.save(model, self.opt_list, epoch_idx, self.registered_loss, self.model_weights_dir)
            # update progress bar
            tq.update(1)
            tq.set_description(f"Epoch:{epoch_idx}")
            tq.set_postfix(tr_loss=train_loss_dict['train@out_loss'], val_loss=valid_loss_dict['valid@out_loss'])
    
    def run_one_epoch(self, epoch_idx:int, model:AbstractModel, dataset:AbstractDataset, mode:str, verbose=2):
        # init registered loss
        reg_loss = {}
        for key in self.registered_loss.keys():
            spl = key.split('@')
            if mode == spl[0]: 
                reg_loss[spl[1]] = 0
        
        if mode != 'test':
            sampler = SequentialSampler(dataset)
        else:
            sampler = RandomSampler(dataset)
        dataloader = tools.FlexibleLoader(dataset=dataset, batch_size=self.params['batchsize'], sampler=sampler, collate_fn=None)

        sample_count = 0
        for data in dataloader:
            sample_count += data['n_sample']
            # model inference
            loss_dict = model(data)
            # registered loss
            for key in loss_dict:
                if key in reg_loss.keys():
                    reg_loss[key] += loss_dict[key] * data['n_sample']
            # print loss
            if verbose > 1 and sample_count * 5 % len(dataset) == 0 and sample_count > 0:
                loss_str = ', '.join([str(key) + '=%.5g' % val / sample_count for key,val in reg_loss.items()])
                logger.info(f"Epoch:{epoch_idx}, Mode:{mode}, sample={sample_count}/{len(dataset)}, loss={loss_str}")
        
        return {mode + '@' + k:v / sample_count for k,v in reg_loss.items()}

    def save(self, model:AbstractModel, opt_list:list, epoch, registered_loss, out_dir):
        '''Save model/opts/params in a fixed structure'''
        formatted_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        result_dict = {
            'info': {
                'model_name': model.name(),
                'model_conf': model.model_conf,
                'epoch': epoch,
                'date': formatted_date,
                'params': self.params,
                'registered_loss': registered_loss
            },
            'model': model,
            'opt': opt_list
        }
        if not os.path.exists(out_dir):
            tools.reinit_dir(out_dir, build=True)
        save_path = os.path.join(out_dir, f'Epoch_{epoch}_{formatted_date}.pt')
        with open(save_path, 'wb') as f:
            torch.save(result_dict, f)
        logger.info(f'Model saved at {save_path}')

    def load(self, load_dir, load_option='latest', suffix='.pt'):
        '''
        Load model/opts/params
        load_option:
            'latest': load latest model
            'Epoch_X': load model in X epoch
        '''
        if load_option == 'latest':
            file_path = max([os.path.join(load_dir, fp) for fp in os.listdir(load_dir) if fp.endswith(suffix)], key=os.path.getctime)
        else:
            matching_files = [os.path.join(load_dir, f) for f in os.listdir(load_dir) if f.endswith(suffix) and f.startswith(load_option)]
            file_path = max(matching_files, key=os.path.getctime)
        logger.info(f'Loading model from {file_path}')
        with open(file_path, 'wb') as fp:
            result_dict = torch.load(fp)
        # load variables
        self.model = result_dict['model']
        

