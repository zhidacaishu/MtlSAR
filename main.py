import argparse
import datetime
import logging
import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.join(PROJECT_ROOT, 'MtlSAR')
if PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, PACKAGE_ROOT)

from models import const
from models.MtlSAR import MtlSAR
from utils import runner, util

def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='Device used for training or inference, e.g., cuda:0 or cpu.')
    parser.add_argument('--random_seed', type=int, default=20251211,
                        help='Random seed used by Python, NumPy, and PyTorch.')
    parser.add_argument('--time', type=str, default='none',
                        help='Run name used by checkpoints and logs. A timestamp is used when set to none.')
    parser.add_argument('--train', type=int, default=1, choices=[0, 1],
                        help='Set to 1 for training and 0 for evaluation only.')
    parser.add_argument('--test_path', type=str, default='',
                        help='Checkpoint path used when --train 0.')
    parser.add_argument('--detect_anomaly', action='store_true',
                        help='Enable PyTorch autograd anomaly detection for debugging.')

    parser.add_argument('--data', type=str, default='JDsearch', choices=['JDsearch', 'KuaiSAR'],
                        help='Dataset setting to use.')
    parser.add_argument('--model', type=str, default='MtlSAR', choices=['MtlSAR'],
                        help='Model name used in output paths.')
    
    return parser

def init_dataset_setting(data_name):
    global_start_time = datetime.datetime.now()

    if data_name == 'KuaiSAR':
        const.init_setting_KuaiSAR()
    elif data_name == 'JDsearch':
        const.init_setting_JDsearch()
    else:
        raise ValueError(f'Unsupported dataset: {data_name}')
    return global_start_time

def main():
    parser = argparse.ArgumentParser(
        description='Train and evaluate MtlSAR for unified search and recommendation.'
    )
    parse_global_args(parser)
    MtlSAR.parse_model_args(parser)
    runner.SARRunner.parse_runner_args(parser)
    args = parser.parse_args()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Please use --device cpu or run on a CUDA-enabled machine.')

    if args.train == 0 and not args.test_path:
        raise ValueError('--test_path is required when --train 0.')

    global_start_time = init_dataset_setting(args.data)
    
    util.setup_seed(args.random_seed)

    if args.time == 'none':
        cur_time =  datetime.datetime.now()
        args.time = cur_time.strftime(r'%Y%m%d-%H%M%S')

    args.model_path = f'output/{args.data}/{args.model}/checkpoints/{args.time}'

    util.set_logging(args)
    print(args)
    for flag, value in sorted(args.__dict__.items(), key=lambda x: x[0]):
        logging.info(f'{flag}: {value}')
    
    model = MtlSAR(args)
    runer = runner.SARRunner(args)

    if args.train == 0:
        model.load_model(model_path=args.test_path)
        test_result, _ = runer.evaluate(model, 'test')
        logging.info('Test Result: ')
        logging.info(test_result)
        print(f'Test result: {test_result}')
    else:
        runer.train(model)
    
    global_end_time = datetime.datetime.now()
    logging.info(f'running used time: {global_end_time - global_start_time}')

if __name__ == '__main__':
    main()