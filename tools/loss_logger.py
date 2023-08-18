import numpy as np
from matplotlib import pyplot as plt

class LossLogger:
    def __init__(self) -> None:
        self.data = None

    def add_loss(self, data:dict):
        '''
        增加一条独立的model训练记录
        data: dict
            'train'/'valid'/'test': [n, epochs] or [epochs]
            'epochs': [epochs]
        '''
        assert('epochs' in data.keys())
        for key in data.keys():
            assert(isinstance(data[key], np.ndarray))
            if len(data[key].shape) == 1:
                data[key] = data[key][None, ...]
            assert(len(data[key].shape) <= 2)
        if self.data is None:
            self.data = data
        else:
            for key in data.keys():
                assert(key in self.data.keys())
                self.data[key] = np.concatenate([self.data[key], data[key]], axis=0)

    def clear(self):
        self.data = None
    
    def plot(self, std_bar=False, log_loss=False, title='Loss Logger', out_path:str=None):
        '''
        输出loss下降图
        std_bar: bool 是否作标准差(对于n>1)误差区间
        title: str
        out_path: str
        '''
        data = self.data.copy()
        for key in data.keys():
            assert(isinstance(data[key], np.ndarray))

        # create figure
        epoch_len = data['epochs'].shape[1]
        plt.figure(figsize = (round(min(12+epoch_len/50, 15)),6))
        plt.title(title)
        plt.xlabel('Epochs')
        if log_loss:
            plt.ylabel('Loss(log10)')
        else:
            plt.ylabel('Loss')

        for idx, key in enumerate(sorted(data.keys())):
            if key == 'epochs':
                continue
            n = data[key].shape[0]
            std_flag = (n > 1) and std_bar
            epochs = np.repeat(data['epochs'], n, axis=0)
            
            if log_loss:
                data[key] = np.log10(data[key])

            mean_loss = np.mean(data[key], axis=0)
            plt.plot(epochs.T, data[key].T, color=f'C{idx}', alpha=0.2)
            plt.plot(epochs[0, :], mean_loss, 'o-', color=f'C{idx}', label=f'{key} loss')
            

            if std_flag:
                std_loss = np.std(data[key], axis=0)
                plt.fill_between(epochs[0, :], mean_loss-std_loss, mean_loss+std_loss, alpha=0.2, color=f"C{idx}", edgecolor=None) # 绘制标准差区域

        plt.legend()
        if out_path is None:
            plt.show()
        else:
            plt.savefig(out_path)
        plt.close()

if __name__ == '__main__':
    logger = LossLogger()
    data = {
        'epochs': np.asarray(list(range(100))),
        'train': np.random.randn(5, 100),
        'valid': np.random.randn(5, 100)+2
    }
    logger.add_loss(data)
    logger.plot(std_bar=True, log_loss=False, title="No log loss", out_path='./test_losslogger.png')
