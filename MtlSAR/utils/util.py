import numpy as np
import torch
import random
import os
import yaml
import logging
import pickle

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

GLOBAL_SEED = 1

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_WORKER_ID = None

def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

def load_pickle(path):
    return pickle.load(open(path, 'rb'))

def batch_to_gpu(batch, device):
    for c in batch:
        if isinstance(batch[c], torch.Tensor):
            batch[c] = batch[c].to(device)
        elif isinstance(batch[c], list):
            batch[c] = [[p.to(device) for p in k] if isinstance(k, list) else k.to(device) for k in batch[c]]
    return batch

def format_metric(result_dict):
    assert type(result_dict) == dict
    format_str = []
    metrics = np.unique([k.split('@')[0] for k in result_dict.keys()])
    topks = np.unique([int(k.split('@')[1]) for k in result_dict.keys()])
    for topk in np.sort(topks):
        for metric in np.sort(metrics):
            name = f'{metric}@{topk}'
            m = result_dict[name]
            if type(m) is float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append(f'{name}: {m:<.4f}')
            elif type(m) is int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append(f'{name}: {m}')
    return ', '.join(format_str)

def check_dir(file_name):
    dir_path = os.path.dirname(file_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_parameter(args):
    load_path = f'config/{args.model}_{args.data}.yaml'
    with open(load_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    
    for key, value in configs.items():
        setattr(args, key, value)
    return args

def set_logging(args):
    log_path = f'output/{args.data}/{args.model}/logs/{args.time}.log'
    check_dir(log_path)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename=log_path, filemode='w')