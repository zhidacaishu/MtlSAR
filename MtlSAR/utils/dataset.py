import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from models import const
from utils.sampler import *
import ast

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return self.sampler.data.shape[0]
    
    def __getitem__(self, index):
        return self.sampler.sample(index)

    def collate_batch(self, feed_dicts):
        result_dict = {}
        for key in feed_dicts[0].keys():
            if isinstance(feed_dicts[0][key], list):
                stack_val = list(torch.from_numpy(np.array(elem)) for elem in zip(*[d[key] for d in feed_dicts]))
                if len(stack_val) == 1:
                    stack_val = stack_val[0]
            else:
                continue
            result_dict[key] = stack_val
        result_dict['batch_size'] = len(feed_dicts)
        result_dict['search'] = feed_dicts[0]['src']
        return result_dict
    
class RecDataset(BaseDataset):
    def __init__(self, train, user_vocab):
        super().__init__()
        if train == 'train':
            self.sampler = Sampler(data_path=const.rec_train, search=False, user_vocab=user_vocab)
        elif train == 'val':
            self.sampler = Sampler(data_path=const.rec_val, search=False, user_vocab=user_vocab)
        elif train == 'test':
            self.sampler = Sampler(data_path=const.rec_test, search=False, user_vocab=user_vocab)

class SrcDataset(BaseDataset):
    def __init__(self, train, user_vocab):
        super().__init__()
        if train == 'train':
            self.sampler = Sampler(data_path=const.src_train, search=True, user_vocab=user_vocab)
        elif train == 'val':
            self.sampler = Sampler(data_path=const.src_val, search=True, user_vocab=user_vocab)
        elif train == 'test':
            self.sampler = Sampler(data_path=const.src_test, search=True, user_vocab=user_vocab)