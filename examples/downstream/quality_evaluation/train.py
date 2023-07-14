import os
import sys
sys.path.append(os.path.dirname(__file__))
import math
import numpy as np
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, MeanSquaredError, MeanSquaredLogError,R2Score,PearsonCorrCoef
import platform
from torch.utils.tensorboard import SummaryWriter


def my_cuda_tensor(items, device):
    for k, v in items.items():
        if isinstance(v, torch.Tensor):
            items[k] = v.to(device)
        elif isinstance(v, dict):
            items[k] = my_cuda_tensor(v, device)

    return items


class MyTrainer(object):
    def __init__(self, args, model, optimizer=None, scheduler=None, logger=None, tensorboard_writer=None, **kwargs):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.tensorboard_writer = tensorboard_writer

        self.classify_metric_collection = MetricCollection([
                                    Accuracy(task="multiclass", num_classes=3, average="micro"),
                                    # Precision(task="multiclass", num_classes=3, average="micro"),
                                    # Recall(task="multiclass", num_classes=3, average="micro")
                                ]).to(args.device)
        self.logits_metric_collection = MetricCollection([
                                    MeanSquaredError(),
                                    # MeanSquaredLogError(),
                                    R2Score(),
                                    PearsonCorrCoef()
                                ]).to(args.device)
    
    def train(self, train_dataloader, valid_dataloader):
        self._global_step = 0
        size = len(train_dataloader.dataset) // train_dataloader.batch_size

        best_valid_loss = 9999
        best_epoch = None
        for epoch in tqdm(range(self.args.epochs)):
            self.model.train()
            # train
            train_loss = 0
            self.logger.info(f"------ epoch {epoch} ------")
            for idx, batch in enumerate(train_dataloader):
                batch = my_cuda_tensor(batch, self.args.device)
                # Compute prediction error
                outputs = self.model(**batch)
                score_logits, label_logits = outputs.score_logits, outputs.label_logits
                
                pred_label = torch.argmax(label_logits, dim=1)
                self.classify_metric_collection.update(pred_label, batch["label"])
                self.logits_metric_collection.update(score_logits, batch["score"].float())
                
                self.tensorboard_writer.add_scalar("train_loss", outputs.loss.item(), self._global_step)
                self._global_step +=1
                train_loss += outputs.loss.item()

                # Backpropagation
                loss = outputs.loss / self.args.grad_accum
                loss.backward()
                # 梯度积累
                if (idx+1) % self.args.grad_accum == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            train_loss /= size

            # train_metric
            total_train_classify_metric = {k: v.item() for k,v in self.classify_metric_collection.compute().items()}
            total_train_logits_metric = {k: v.item() for k,v in self.logits_metric_collection.compute().items()}
            self.logger.info(f"train metric for epoch: {total_train_classify_metric},{total_train_logits_metric}")
            self.classify_metric_collection.reset()
            self.logits_metric_collection.reset()
            
            valid_loss, total_valid_classify_metric, total_valid_logits_metric = self.valid(valid_dataloader)
            # log
            for k, v in total_valid_classify_metric.items():
                self.tensorboard_writer.add_scalar(f"classify_metric/{k}", v, epoch)
            for k, v in total_valid_logits_metric.items():
                self.tensorboard_writer.add_scalar(f"logits_metric/{k}", v, epoch)                                                   
            self.tensorboard_writer.add_scalars(f"EpochLoss", {"TrainLoss": train_loss, "ValidLoss": valid_loss}, epoch)

            if valid_loss < best_valid_loss:
                self.model.save_pretrained(self.args.checkpoint_dir)
                best_valid_loss = valid_loss
                best_epoch = epoch
                self.logger.info(f"saving best model at epoch {epoch}...")
        
        self.logger.info(f"Finish training, best model is at epoch {best_epoch}...")

    def valid(self, valid_dataloader): 
        self.model.eval()
        size = len(valid_dataloader.dataset) // valid_dataloader.batch_size
        valid_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                batch = my_cuda_tensor(batch, self.args.device)
                # Compute prediction error
                outputs = self.model(**batch)
                score_logits, label_logits = outputs.score_logits, outputs.label_logits
                valid_loss += outputs.loss.item()

                pred_label = torch.argmax(label_logits, dim=1)
                self.classify_metric_collection.update(pred_label, batch["label"])
                self.logits_metric_collection.update(score_logits, batch["score"].float())

            valid_loss /= size
        
        total_valid_classify_metric = {k: v.item() for k,v in self.classify_metric_collection.compute().items()}
        total_valid_logits_metric = {k: v.item() for k,v in self.logits_metric_collection.compute().items()}
        self.logger.info(f"Validation metric: {total_valid_classify_metric},{total_valid_logits_metric}")
        self.classify_metric_collection.reset()
        self.logits_metric_collection.reset()

        return valid_loss, total_valid_classify_metric, total_valid_logits_metric
