import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
from model.AGCRNCell import AGCRNCell

class AVWDCRNN(nn.Module):
    def __init__(self, model_name, p1,node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers=1):
        super(AVWDCRNN, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(model_name, p1,node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(model_name, p1,node_num, dim_out, dim_out, cheb_k, embed_dim))

    def forward(self, x, init_state, node_embeddings):
        #shape of x: (B, T, N, D)
        #shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      #(num_layers, B, N, hidden_dim)

    
###========Main model=========
class AGCRN_UQ(nn.Module):
    def __init__(self, args):
        super(AGCRN_UQ, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers
        
        ###
        self.model_name = args.model_name 
        self.p1= args.p1
        
        
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
    
 
        self.encoder = AVWDCRNN(self.model_name, self.p1, args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                args.embed_dim, args.num_layers)
    
            
        if self.model_name == "combined":
            self.get_mu = nn.Sequential(
             nn.Conv2d(1, 32, kernel_size=(1,1), bias=True),
             nn.Dropout(0.2),
             nn.ReLU(),
             nn.Conv2d(32, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True), 
             ) 

            self.get_log_var = nn.Sequential(
             nn.Conv2d(1, 32, kernel_size=(1,1), bias=True),
             nn.Dropout(0.2),
             nn.ReLU(),
             nn.Conv2d(32, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True), 
             ) 

        
    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        #supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        emb = self.node_embeddings   
        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, emb)      #B, T, N, hidden
        #print(output.shape)
        output = output[:, -1:, :, :]   

        #CNN based predictor
    
        if  self.model_name == "combined":  
            mu = self.get_mu((output))                         #B, T*C, N, 1
            mu = mu.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
            mu = mu.permute(0, 1, 3, 2)                             #B, T, N, C

            log_var = self.get_log_var((output))                         #B, T*C, N, 1
            log_var = log_var.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
            log_var = log_var.permute(0, 1, 3, 2)  
            return mu, log_var
        
    
    
    
    
    
    