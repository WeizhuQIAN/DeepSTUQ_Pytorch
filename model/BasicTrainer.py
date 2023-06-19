import torch
import math
import os
import time
import copy
import numpy as np
#from lib.logger import get_logger
from lib.metrics import All_Metrics
from lib.utils import enable_dropout
import torch.nn.functional as F
from torch.autograd import grad
import torch.nn as nn

from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
#from model.train_methods import QuantileLoss


class Trainer(object):
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super().__init__()    
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        #self.quantile_loss = QuantileLoss()
        
    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_dataloader):
                data = data[..., :self.args.input_dim]
                label = target[..., :self.args.output_dim]
                if self.args.real_value:
                    label = self.scaler.inverse_transform(label)   
                    
                if self.args.model_name == "basic" or self.args.model_name=="dropout":
                    output = self.model(data, target, teacher_forcing_ratio=0.)
                    loss = self.loss(output.cuda(), label)
                
                elif self.args.model_name == "heter" or self.args.model_name=="combined":
                    output, _ = self.model(data, target, teacher_forcing_ratio=0.)
                    loss = self.loss(output.cuda(), label)
                
                elif self.args.model_name == "quantile": 
                    output = self.model(data, target, teacher_forcing_ratio=0.)                
                    loss = self.quantile_loss(output,label)
                
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data = data[..., :self.args.input_dim]
            label = target[..., :self.args.output_dim]  # (..., 1)
            
            #label = torch.log(label)
            
            self.optimizer.zero_grad()

            #teacher_forcing for RNN encoder-decoder model
            #if teacher_forcing_ratio = 1: use label as input in the decoder for all steps
            if self.args.teacher_forcing:
                global_step = (epoch - 1) * self.train_per_epoch + batch_idx
                teacher_forcing_ratio = self._compute_sampling_threshold(global_step, self.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #data and target shape: B, T, N, F; output shape: B, T, N, F
            
            if self.args.real_value:
                label = self.scaler.inverse_transform(label)
                
            if self.args.model_name == "basic" or self.args.model_name=="dropout":
                    index = torch.randperm(data.shape[1])
                    data = data[:,index,:,:]
                    
                    output = self.model(data, target, teacher_forcing_ratio=0.)
                    loss = self.loss(output.cuda(), label)
                    
            elif self.args.model_name == "heter" or self.args.model_name=="combined":    
                mu, log_var = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
                loss = torch.mean(torch.exp(-log_var)*(label-mu)**2 + log_var)
                loss = 0.1*loss + 0.9*self.loss(mu, label) 
                        
            elif self.args.model_name == "quantile": 
                output = self.model(data, target, teacher_forcing_ratio=teacher_forcing_ratio)  
                loss = self.quantile_loss(output,label)
 
            loss.backward()

            # add max grad clipping
            if self.args.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

            #log information
            if batch_idx % self.args.log_step == 0:
                print('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, self.train_per_epoch, loss.item()))
        train_epoch_loss = total_loss/self.train_per_epoch
        #self.logger.info('**********Train Epoch {}: averaged Loss: {:.6f}, tf_ratio: {:.6f}'.format(epoch, train_epoch_loss, teacher_forcing_ratio))

        #learning rate decay
        if self.args.lr_decay:
            self.lr_scheduler.step()
        return train_epoch_loss
    
    def train(self):
        
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        for epoch in range(1, self.args.epochs + 1):
            #epoch_time = time.time()
            train_epoch_loss = self.train_epoch(epoch)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                print('Gradient explosion detected. Ending...')
                break
            #if self.val_loader == None:
            #val_epoch_loss = train_epoch_loss
            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:
                    print("Validation performance didn\'t improve for {} epochs. "
                                    "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                print('*********************************Current best model saved!')
                best_model = copy.deepcopy(self.model.state_dict())
                    

        training_time = time.time() - start_time
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))

        #save the best model to file
        if not self.args.debug:
            torch.save(best_model, self.best_path)
            print("Saving current best model to " + self.best_path)

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        #self.test(self.model, self.args, self.test_loader, self.scaler, self.logger=None)    
    
    @staticmethod
    def test(model, args, data_loader, scaler, logger=None, path=None):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
            
        model.eval()
        #enable_dropout(model)
        
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                #output, z, mu, log_sigma = model.sample_code_(data)
                
                #label = torch.log(label)
                
                output = model(data, target, teacher_forcing_ratio=0)
                #output, *_ = model(data, target, teacher_forcing_ratio=0)
                y_true.append(label)
                y_pred.append(output)
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
        if args.real_value:
            y_pred = torch.cat(y_pred, dim=0)
        else:
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
        #np.save('./{}_true.npy'.format(args.dataset), y_true.cpu().numpy())
        #np.save('./{}_pred.npy'.format(args.dataset), y_pred.cpu().numpy())
        for t in range(y_true.shape[1]):
            mae, rmse, mape, _, _ = All_Metrics(y_pred[:, t, ...], y_true[:, t, ...],
                                                args.mae_thresh, args.mape_thresh)
            print("Horizon {:02d}, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape*100))
        mae, rmse, mape, _, _ = All_Metrics(y_pred, y_true, args.mae_thresh, args.mape_thresh)
        print("Average Horizon, MAE: {:.2f}, RMSE: {:.2f}, MAPE: {:.4f}%".format(
                    mae, rmse, mape*100))
    