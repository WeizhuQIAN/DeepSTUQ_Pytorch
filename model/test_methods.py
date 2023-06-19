import torch
import torch.nn as nn
from lib.utils import enable_dropout
from lib.metrics import All_Metrics
from tqdm import tqdm 

####======MC+Heter========####
def combined_test(model,num_samples,args, data_loader, scaler, T=torch.zeros(1).cuda(), logger=None, path=None):
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
            for batch_idx, (data, _) in enumerate(data_loader):
                data = data[..., :args.input_dim]
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
    
    temperature = torch.exp(T)     
    y_pred = torch.mean(mc_mus, axis=0)
    total_var = torch.var(mc_mus, axis=0)+torch.exp(torch.mean(mc_log_vars, axis=0))/temperature   
    total_std = total_var**0.5 
    
    mpiw = 2*1.96*torch.mean(total_std)    
    nll = nll_fun(y_pred.ravel(), y_true.ravel(), total_var.ravel())
    lower_bound = y_pred-1.96*total_std
    upper_bound = y_pred+1.96*total_std
    in_num = torch.sum((y_true >= lower_bound)&(y_true <= upper_bound ))
    picp = in_num/(y_true.size(0)*y_true.size(1)*y_true.size(2))
    
    
    print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%,  NLL: {:.4f}, \
PICP: {:.4f}%, MPIW: {:.4f}".format(mae, rmse, mape*100, nll, picp*100, mpiw))  
    