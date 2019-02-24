import torch
from torch import nn
from torch.nn import Parameter
class Controller_SineWave(nn.Module):
    def __init__(self):
        super(Controller_SineWave, self).__init__()

        self.inp_dims = inp_dims
        self.num_hidden_cells = num_hidden_cells
        self.num_hidden_layer = self.num_hidden_layer

        self.LSTM = nn.LSTM(input_size=inp_dims,
                            hidden_size=num_hidden_cells,
                            num_layers=num_hidden_layers)

        self.h_bias = Parameter(torch.randn(self.num_hidden_layer, 1, self.num_hidden_cells) * 0.05)







