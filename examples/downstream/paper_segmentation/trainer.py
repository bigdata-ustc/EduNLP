import os
import sys
# sys.path.append(os.path.dirname(__file__))
import torchmetrics
import math
import numpy as np
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score
from torchmetrics import MetricCollection
from utils import get_pk
from model import PaperSegModel

def my_cuda_tensor(items, device):
    for k, v in items.items():
        if isinstance(v, torch.Tensor):
            items[k] = v.to(device)
        elif isinstance(v, list):
            items[k] = my_cuda_document(v, device)
    return items

def my_cuda_document(v, device):
    if isinstance(v, torch.Tensor):
        v = v.to(device)
    elif isinstance(v, list):
        v = [my_cuda_document(x, device) for x in v]
    return v

class MyTrainer(object):
    def __init__(self, args, model, optimizer=None, scheduler=None, logger=None, tensorboard_writer=None, **kwargs):
        self.args = args
        self.model = model
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.tensorboard_writer = tensorboard_writer

        self.Metric = MetricCollection([
            BinaryPrecision(),
            BinaryRecall(),
            BinaryF1Score(),
        ]).to(args.device)
    
    def train(self, train_dataloader, eval_dataloader):
        self._global_step = 0
        size = len(train_dataloader.dataset) // train_dataloader.batch_size

        # best_val_metric = 
        best_valid_loss = 9999
        best_epoch = None
        
        for epoch in range(self.args.epochs):
            self.model.train()
            # train
            train_loss = 0
            self.logger.info(f"------ epoch {epoch} ------")
            for batch in train_dataloader:  # batch_size = 1
                batch = my_cuda_tensor(batch, self.args.device)

                # self.model.zero_grad()
                self.optimizer.zero_grad()
                
                outputs = self.model(**batch)
                
                self.tensorboard_writer.add_scalar("train_loss", outputs.loss.item(), self._global_step)
                self._global_step +=1
                train_loss += outputs.loss.item()

                # Backpropagation
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

            train_loss /= size
            self.logger.info(f'epoch {epoch:3} done, loss = {train_loss}')
            # validate
            valid_loss, valid_pk, valid_metrics = self.valid(eval_dataloader)
            
            for k, v in valid_metrics.items():
                self.tensorboard_writer.add_scalar(f"Metric/{k}", v, epoch)
            self.tensorboard_writer.add_scalar(f"Metric/pk", valid_pk, epoch)
            self.tensorboard_writer.add_scalars(f"EpochLoss", {"TrainLoss": train_loss, "ValidLoss": valid_loss}, epoch)

            # store best model
            # if valid_metrics["BinaryF1Score"] > best_val_metric:
            #     best_val_metric = valid_metrics["BinaryF1Score"]
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                try:
                    self.model.save_pretrained(self.args.checkpoint_dir)
                except Exception:
                    self.logger.info("[Warning] Model.save_pretrained Error !!!")
                
                with open(f'{self.args.checkpoint_dir}/best_model.bin', mode='wb') as f:
                    torch.save(self.model.state_dict(), f)
                with open(f'{self.args.checkpoint_dir}/best_model.obj.bin', mode='wb') as f:
                    torch.save(self.model, f)
                
                best_epoch = epoch
                self.logger.info(f"saving best model at epoch {epoch}...")
                
        self.logger.info(f"Finish training, best model is at epoch {best_epoch}...")

    def valid(self, valid_dataloader): 
        assert valid_dataloader.batch_size == 1
        self.model.eval()
        size = len(valid_dataloader.dataset) // valid_dataloader.batch_size
        valid_pk_list = []
        valid_loss = 0
        for batch in valid_dataloader:
            batch = my_cuda_tensor(batch, self.args.device)
            # documents, tags = batch
            # for document, tag in zip(documents, tags):
            if isinstance(self.model, PaperSegModel):
                outputs = self.model(**batch)
                flat_pred_tags = torch.argmax(torch.softmax(outputs.logits, 1), 1)
                valid_loss += outputs.loss.item()
            else:
                outputs = self.model(batch["documents"])
                flat_pred_tags = torch.argmax(torch.softmax(outputs, 1), 1)
                
                criterion = torch.nn.CrossEntropyLoss()
                loss = criterion(outputs, batch["tags"].view(-1))
                valid_loss += loss.item()
            
            """ compute Metric"""
            self.Metric.update(flat_pred_tags, batch["tags"].view(-1))
            """ compute pk"""
            tag = batch["tags"].detach().cpu().numpy()
            pred_tag = flat_pred_tags.detach().cpu().numpy()
            document = batch["documents"][0]
            """ only can compute each document at each time """
            k = max(math.ceil(len(document)/2), 2)
            pk = get_pk(pred_tag, tag, k)
            valid_pk_list.append(pk)

        valid_loss /= size
        valid_pk = np.mean(valid_pk_list)
        valid_metrics = {k: v.item() for k,v in self.Metric.compute().items()}
        self.logger.info(f"Validate: valid_loss= {valid_loss}, valid_pk= {valid_pk}, Validation metric: {valid_metrics}")
        self.Metric.reset()

        return valid_loss, valid_pk, valid_metrics