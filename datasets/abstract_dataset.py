from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    __name = 'abstract_dataset'

    @classmethod
    def name(cls):
        return cls.__name

    def __init__(name:str):
        __name = name
    
    def mode(self, mode=['train', 'valid', 'test', 'all']):
        '''change dataset mode'''
        raise NotImplementedError
    
    def set_kf_index(self, kf_index):
        '''use data in one fold subset'''
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

