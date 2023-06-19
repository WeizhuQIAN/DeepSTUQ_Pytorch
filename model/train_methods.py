import torch
import torch.nn as nn
from lib.utils import enable_dropout
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from tqdm import tqdm 


###====Adaptive Weight Averaging====###
def awa_train_combined(trainer,epoch_swa, regularizer=None, lr_schedule=None): 
    #total_loss = 0
    #criterion = torch.nn.L1Loss()#torch.nn.MSELoss()
    #loss_sum = 0.0
    num_iters = len(trainer.train_loader)
    
    lr1 = 0.003
    lr2 = lr1*0.01
    
    optimizer_swa = torch.optim.Adam(params=model.parameters(), lr=lr1, betas=(0.9,0.99),
                            weight_decay=1e-4, amsgrad=False)

    cycle = num_iters
    swa_c = 1
    swa_model = AveragedModel(trainer.model)
    #scheduler = CosineAnnealingLR(optimizer_swa, T_max=cycle,eta_min=0.0)
    scheduler_swa = CosineAnnealingWarmRestarts(optimizer_swa,T_0=num_iters,
                                                T_mult=1, eta_min=lr2, last_epoch=-1)
    
    lr_ls = []
    for epoch in tqdm(range(epoch_swa)):
        trainer.model.train()
        for iter, (data, target) in enumerate(trainer.train_loader):
            input = data[..., :trainer.args.input_dim]
            label = target[..., :trainer.args.output_dim]  # (..., 1)
            optimizer_swa.zero_grad()       

            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            if trainer.args.teacher_forcing:
                global_step = (epoch - 1) * trainer.train_per_epoch + batch_idx
                teacher_forcing_ratio = trainer._compute_sampling_threshold(global_step, trainer.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #output,log_var = trainer.model.forward_heter(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            mu,log_var = trainer.model.forward(data, target, teacher_forcing_ratio=0.5)
            if trainer.args.real_value:
                label = trainer.scaler.inverse_transform(label)
            loss = torch.mean(torch.exp(-log_var)*(label-mu)**2 + log_var)
            loss = 0.1*loss + 0.9*trainer.loss(mu, label)  
            #loss = trainer.loss(mu, label)
            loss.backward()
            optimizer_swa.step()
            if (epoch % 2 ==0) & (iter != num_iters-1):
                scheduler_swa.step()
            else:  
               optimizer_swa.param_groups[0]["lr"]=lr2
            #scheduler.step() 
        if (epoch+1) % 2 ==0:#) & (epoch !=0):
            swa_model.update_parameters(trainer.model)
            torch.optim.swa_utils.update_bn(trainer.train_loader, swa_model) 
            
        #swa_scheduler.step()
        #scheduler.step()   
    return swa_model    


def swa_train(trainer,epoch_swa, regularizer=None, lr_schedule=None): 

    num_iters = len(trainer.train_loader)
    
    lr1 = 0.003#0.1
    lr2 = lr1*0.01#0.001
    
    optimizer_swa = torch.optim.Adam(params=model.parameters(), lr=lr1, betas=(0.9,0.99),
                            weight_decay=1e-4, amsgrad=False)

    cycle = num_iters
    swa_c = 1
    swa_model = AveragedModel(trainer.model)
    #scheduler = CosineAnnealingLR(optimizer_swa, T_max=cycle,eta_min=0.0)
    scheduler_swa = CosineAnnealingWarmRestarts(optimizer_swa,T_0=num_iters,
                                                T_mult=1, eta_min=lr2, last_epoch=-1)
    lr_ls = []
    for epoch in tqdm(range(epoch_swa)):
        trainer.model.train()
        for iter, (data, target) in enumerate(trainer.train_loader):
            input = data[..., :trainer.args.input_dim]
            label = target[..., :trainer.args.output_dim]  # (..., 1)
            optimizer_swa.zero_grad()       

            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            if trainer.args.teacher_forcing:
                global_step = (epoch - 1) * trainer.train_per_epoch + batch_idx
                teacher_forcing_ratio = trainer._compute_sampling_threshold(global_step, trainer.args.tf_decay_steps)
            else:
                teacher_forcing_ratio = 1.
            #output,log_var = trainer.model.forward_heter(data, target, teacher_forcing_ratio=teacher_forcing_ratio)
            output = trainer.model.forward(data, target, teacher_forcing_ratio=0.5)
            if trainer.args.real_value:
                label = trainer.scaler.inverse_transform(label)
            loss = trainer.loss(output, label)  
            loss.backward()
            optimizer_swa.step()
            if (epoch % 2 ==0) & (iter != num_iters-1):
                scheduler_swa.step()
            else:  
               optimizer_swa.param_groups[0]["lr"]=lr2
            #scheduler.step() 
        if (epoch+1) % 2 ==0:
            swa_model.update_parameters(trainer.model)
            torch.optim.swa_utils.update_bn(trainer.train_loader, swa_model) 
            
    return swa_model    
    
###====Calibration====###    
class ModelCali(nn.Module):
    def __init__(self,args):
        super(ModelCali, self).__init__()
        #self.model = model
        #self.T = nn.Parameter(torch.ones(args.num_nodes)*1.5, requires_grad=True) 
        self.T = nn.Parameter(torch.ones(1)*1.5, requires_grad=True) 
        
              
def train_cali(model, args, data_loader, scaler, logger=None, path=None):
    model_cali = ModelCali(args).cuda()
    optimizer_cali = torch.optim.LBFGS(list(model_cali.parameters()), lr=0.02, max_iter=500)
    model.eval()
    #nll_fun = nn.GaussianNLLLoss()
    y_true = []
    mu_pred = []
    log_var_pred = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data[..., :args.input_dim]
            label = target[..., :args.output_dim]
            mu, log_var = model.forward(data, target, teacher_forcing_ratio=0)
            mu_pred.append(mu)
            log_var_pred.append(log_var)
            y_true.append(label)
        if args.real_value:
            mu_pred = torch.cat(mu_pred, dim=0)
        else:
            mu_pred = scaler.inverse_transform(torch.cat(mu_pred, dim=0))     
        log_var_pred = torch.cat(log_var_pred, dim=0)    
        y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))
           
    y_pred = mu_pred
    precision = torch.exp(-log_var_pred)    
    def eval_():
        optimizer_cali.zero_grad()
        #loss(input, target, var)
        temperature = torch.exp(model_cali.T) 
        #loss = nll_fun(y_pred,y_true,torch.exp(log_var_pred)/temperature) 
        loss = torch.mean(temperature*precision*(y_true-y_pred)**2 + log_var_pred-model_cali.T)
        #print(loss.item())
        loss.backward()
        return loss
    optimizer_cali.step(eval_) 
    print("Calibration finished!")
    return model_cali.T
    
    
def train_cali_mc(model,num_samples, args, data_loader, scaler, logger=None, path=None):
    model_cali = ModelCali(args).cuda()
    optimizer_cali = torch.optim.LBFGS(list(model_cali.parameters()), lr=0.02, max_iter=500)
    model.eval()
    enable_dropout(model)
    nll_fun = nn.GaussianNLLLoss()
    y_true = []
    with torch.no_grad():
        for batch_idx, (_, target) in enumerate(data_loader):
            label = target[..., :args.output_dim]
            y_true.append(label)
    y_true = scaler.inverse_transform(torch.cat(y_true, dim=0)).squeeze(3)
    
    mc_mus = torch.empty(0, y_true.size(0), y_true.size(1), y_true.size(2)).cuda()
    mc_log_vars = torch.empty(0, y_true.size(0),y_true.size(1), y_true.size(2)).cuda()
    
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            mu_pred = []
            log_var_pred = []
            for batch_idx, (data, target) in enumerate(data_loader):
                data = data[..., :args.input_dim]
                label = target[..., :args.output_dim]
                mu, log_var = model.forward(data, target, teacher_forcing_ratio=0)
                #print(mu.size())
                mu_pred.append(mu.squeeze(3))
                log_var_pred.append(log_var.squeeze(3))
        
            if args.real_value:
                mu_pred = torch.cat(mu_pred, dim=0)
            else:
                mu_pred = scaler.inverse_transform(torch.cat(mu_pred, dim=0))     
            log_var_pred = torch.cat(log_var_pred, dim=0)    

            #print(mc_mus.size(),mu_pred.size())    
            mc_mus = torch.vstack((mc_mus,mu_pred.unsqueeze(0)))   
            mc_log_vars = torch.vstack((mc_log_vars,log_var_pred.unsqueeze(0))) 
    
    y_pred = torch.mean(mc_mus, axis=0)
    #pred_std = torch.sqrt(torch.exp(torch.mean(mc_log_vars, axis=0)))
    #mc_std = torch.std(mc_mus, axis=0)  
    #total_std = mc_std+pred_std
    #total_var = total_std**2   
    #log_var_total = 2*torch.log(mc_std+pred_std)
    log_var_total = torch.exp(torch.mean(mc_log_vars, axis=0))
    #precision = (mc_std+pred_std)**2
    precision = torch.exp(-torch.mean(mc_log_vars, axis=0))
        
    def eval_():
        optimizer_cali.zero_grad()
        #loss(input, target, var)
        temperature = torch.exp(model_cali.T) 
        #loss = nll_fun(y_pred.ravel(),y_true.ravel(),torch.exp(log_var_total.ravel())/temperature) 
        #loss = torch.mean(temperature*precision*(y_true-y_pred)**2 + log_var_total-model_cali.T)
        loss = torch.mean(temperature*precision*(y_true-y_pred)**2-model_cali.T)
        #print(loss.item())
        loss.backward()
        return loss
    optimizer_cali.step(eval_)   
    print("Calibration finished!")
    return model_cali.T    
    
