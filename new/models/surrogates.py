import torch
import torch.nn as nn


class SurrogateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        
        self.lstm = nn.Sequential()
        self.lstm.add_module('lstm1', nn.LSTM(input_size, hidden_size, num_layers))
        self.ff1 = nn.Linear(hidden_size, 128)
        self.ff2 = nn.Linear(20, 128)
        
        self.act = nn.ReLU()
        
    def forward(self, x):
        #h0 = torch.autograd.Variable(torch.zeros((self.num_layers, x.size(1), self.hidden_size)))
        #c0 = torch.autograd.Variable(torch.zeros((self.num_layers, x.size(1), self.hidden_size)))
        output, (hn, cn) = self.lstm(x)#, (h0, c0))
        output = output[:, :, 0].unsqueeze(2)
        return output
    
class SurrogateFNN(nn.Module):
    def __init__(self, input_size, output_size, n_layers, n_nodes, act = nn.ReLU()):
        super().__init__()
        self.input_size = input_size
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.act = act
        
        self.FNN = nn.Sequential()
        self.FNN.add_module('fnn0', nn.Linear(input_size, n_nodes))
        self.FNN.add_module('act0', act)
        for i in range(n_layers - 1):
            self.FNN.add_module('fnn%d'% (i+1), nn.Linear(n_nodes, n_nodes))
            self.FNN.add_module('act%d'% (i+1), act)
            
        self.FNN.add_module('fnn_final', nn.Linear(n_nodes, output_size))
        self.FNN.add_module('relu_final', nn.Sigmoid())
        
    def forward(self, x):
        return self.FNN(x)
                                              
                                            
