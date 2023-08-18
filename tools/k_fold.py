from datasets.abstract_dataset import AbstractDataset

class KFoldIterator:
    def __init__(self, dataset:AbstractDataset, k):
        self.current = -1
        self.k = k
        self.dataset = dataset

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.k:
            return self.dataset.set_kf_index(self.current)
        else:
            raise StopIteration
