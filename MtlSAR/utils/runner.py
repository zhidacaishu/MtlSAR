import time
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import const
from utils import util
from utils.dataset import *
import logging
import gc

class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=100, help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
        parser.add_argument('--patience', type=int, default=3, help='Number of epochs with no improvement after which learning rate will be reduced')
        parser.add_argument('--early_stop', type=int, default=5, help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimal learning rate.')
        parser.add_argument('--l2', type=float, default=1e-5, help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=1024, help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=512, help='Batch size during testing.')
        parser.add_argument('--num_workers', type=int, default=2, help='Number of processors when prepare batches in DataLoader.')

        return parser
    
    def __init__(self, args):
        self.epoch = args.epoch
        self.print_interval = 500

        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.patience = args.patience
        self.min_lr = args.min_lr
        self.l2 = args.l2
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.num_workers = args.num_workers

        self.topk = [1, 5, 10, 30, 50]
        self.metrics = ['NDCG', 'HR']
        self.main_metric = 'NDCG@5'

        self.optimizer = None
        self.datasetParaDict = {'user_vocab': None}
        self.query_vocab = None

    def _build_optimizer(self, model):
        self.optimizer = torch.optim.Adam(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=self.patience, min_lr=self.min_lr)

    def get_DataLoader(self, dataset, batch_size, shuffle):
        if self.num_workers > 0:
            prefetch_factor = batch_size // self.num_workers + 1
            persistent_workers = True
        else:
            prefetch_factor = None
            persistent_workers = False

        dataloader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, num_workers=self.num_workers, 
                                pin_memory=True, prefetch_factor=prefetch_factor, 
                                worker_init_fn=util.worker_init_fn, persistent_workers=persistent_workers, collate_fn=dataset.collate_batch)
        return dataloader
    
    def set_dataloader(self):
        raise NotImplementedError
    
    def train(self, model):
        self._build_optimizer(model)

        main_metric_results, dev_results = [], []
        for epoch in range(self.epoch):
            epoch_loss = self.train_epoch(epoch, model)
            logging.info(f'epoch {epoch} mean loss: {epoch_loss:.4f}')
            
            dev_result, main_result = self.evaluate(model, 'val')
            dev_results.append(dev_result)
            main_metric_results.append(main_result)
            logging.info('Dev result: ')
            logging.info(dev_result)
            print(f'Dev result: {dev_result}')
            self.scheduler.step(main_result)
            logging.info(f'Current Learning Rate: {self.scheduler.get_last_lr()}')

            if max(main_metric_results) == main_metric_results[-1]:
                model.save_model()
                test_result, _ = self.evaluate(model, 'test')
                logging.info('Test result: ')
                logging.info(test_result)
                print(f'Test result: {test_result}')

            if self.early_stop > 0 and self.eval_termination(main_metric_results):
                logging.info(f'Early stop at {epoch + 1} based on dev result.')
                break
        
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(' ')
        logging.info(f'Best dev result at epoch {best_epoch}')
        logging.info(dev_results[best_epoch])
        print(f'\nBest dev result at epoch {best_epoch}: {dev_results[best_epoch]}')
        model.load_model()

        test_result, _ = self.evaluate(model, 'test')
        logging.info(' ')
        logging.info('Test result: ')
        logging.info(test_result)
        print(f'\nTest result: {test_result}')
    
    def train_epoch(self, epoch, model):
        raise NotImplementedError
    
    def eval_termination(self, criterion):
        if len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False
    
    @staticmethod
    def evaluate_method(predictions, topk, metrics):
        evaluations = {}
        sort_idx = (-predictions).argsort(axis=1)
        gt_rank = np.argwhere(sort_idx == 0)[:, 1] + 1
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = f'{metric}@{k}'
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                else:
                    raise ValueError(f'Undefined evaluation metric: {metric}')
        return evaluations
    
    @staticmethod
    @torch.no_grad()
    def predict(model, test_loader):
        model.eval()
        predictions = []

        start = time.time()
        for step, batch in enumerate(test_loader):
            prediction = model.predict(util.batch_to_gpu(batch, model.device))
            predictions.extend(prediction.cpu().data.numpy())
            if (~torch.isfinite(prediction)).any():
                raise ValueError('There is a nan or inf!')
            mean_per_raw = prediction.mean(dim=-1)
            if (prediction[:, 0] == mean_per_raw).any() or ((prediction[:, 0] == prediction[:, 1]) & (prediction[:, 0] == prediction[:, -1])).any():
                raise ValueError('Model collapsed!')

        logging.info(f'model evaluate time used {time.time() - start}')
        predictions = np.array(predictions)

        return predictions
    
    def evaluate(self, model, mode):
        raise NotImplementedError
    
    def build_dataset(self):
        if self.datasetParaDict['user_vocab'] is None:
            self.datasetParaDict['user_vocab'] = util.load_pickle(const.user_vocab)
    
    def get_query_vocab(self):
        if self.query_vocab is None:
            self.query_vocab = util.load_pickle(const.query_vocab)

class SARRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--src_loss_weight', type=float, default=0.3)

        return BaseRunner.parse_runner_args(parser)
    
    def __init__(self, args):
        super().__init__(args)
        self.build_dataset()
        self.set_dataloader()

        self.src_loss_weight = args.src_loss_weight

    def set_dataloader(self):
        self.rec_train_loader = self.get_DataLoader(self.train_data['rec'], batch_size=self.batch_size, shuffle=True)
        self.rec_val_loader = self.get_DataLoader(self.val_data['rec'], batch_size=self.eval_batch_size, shuffle=False)
        self.rec_test_loader = self.get_DataLoader(self.test_data['rec'], batch_size=self.eval_batch_size, shuffle=False)

        src_train_batch_size = len(self.train_data['src']) // (len(self.train_data['rec']) // self.batch_size + 1) + 1
        logging.info(f'Search train batch size: {src_train_batch_size}')
        self.src_train_loader = self.get_DataLoader(self.train_data['src'], src_train_batch_size, shuffle=True)
        self.src_val_loader = self.get_DataLoader(self.val_data['src'], self.eval_batch_size, shuffle=False)
        self.src_test_loader = self.get_DataLoader(self.test_data['src'], self.eval_batch_size, shuffle=False)

    def build_dataset(self):
        super().build_dataset()

        self.train_data = {'rec': RecDataset(train='train', user_vocab=self.datasetParaDict['user_vocab']), 'src': SrcDataset(train='train', user_vocab=self.datasetParaDict['user_vocab'])}
        self.val_data = {'rec': RecDataset(train='val', user_vocab=self.datasetParaDict['user_vocab']), 'src': SrcDataset(train='val', user_vocab=self.datasetParaDict['user_vocab'])}
        self.test_data = {'rec': RecDataset(train='test', user_vocab=self.datasetParaDict['user_vocab']), 'src': SrcDataset(train='test', user_vocab=self.datasetParaDict['user_vocab'])}

    def train_epoch(self, epoch, model):
        model.train()
        logging.info(' ')
        logging.info(f'Epoch {epoch}')
        print(f'\nEpoch {epoch}')

        src_iterator = iter(self.src_train_loader)

        loss_list = []
        loss_dict = {'rec': {}, 'src': {}}
        start = time.time()
        for step, rec_batch in enumerate(tqdm(self.rec_train_loader)):
            try:
                src_batch = next(src_iterator)
            except StopIteration:
                src_iterator = iter(self.src_train_loader)
                src_batch = next(src_iterator)
            
            rec_loss = model.rec_loss(util.batch_to_gpu(rec_batch, model.device))
            src_loss = model.src_loss(util.batch_to_gpu(src_batch, model.device))

            for k in rec_loss.keys():
                if k in loss_dict['rec'].keys():
                    loss_dict['rec'][k].append(rec_loss[k].item())
                else:
                    loss_dict['rec'][k] = [rec_loss[k].item()]

            for k in src_loss.keys():
                if k in loss_dict['src'].keys():
                    loss_dict['src'][k].append(src_loss[k].item())
                else:
                    loss_dict['src'][k] = [src_loss[k].item()]

            total_loss = rec_loss['total_loss'] + src_loss['total_loss'] * self.src_loss_weight

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_list.append(total_loss.item())

            if step > 0 and step % self.print_interval == 0:
                logging.info(f"epoch {epoch} step {step} time {time.time() - start} |rec {' '.join([f'{k}: {np.mean(v).item()}' for k, v in loss_dict['rec'].items()])} | src {' '.join([f'{k}: {np.mean(v).item()}' for k, v in loss_dict['src'].items()])}")
        logging.info(f'total time: {time.time() - start}')

        torch.cuda.empty_cache()
        gc.collect()

        return np.mean(loss_list).item()
    
    def evaluate(self, model, mode):
        if mode == 'val':
            rec_predictions = self.predict(model, self.rec_val_loader)
            src_predictions = self.predict(model, self.src_val_loader)
        elif mode == 'test':
            rec_predictions = self.predict(model, self.rec_test_loader)
            src_predictions = self.predict(model, self.src_test_loader)
        else:
            raise ValueError('Test set error')
        rec_results = self.evaluate_method(rec_predictions, self.topk, self.metrics)
        src_results = self.evaluate_method(src_predictions, self.topk, self.metrics)

        results = {'rec': util.format_metric(rec_results), 'src': util.format_metric(src_results)}

        return results, (rec_results[self.main_metric] + src_results[self.main_metric]) / 2.0