import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


np.random.seed(0)
torch.manual_seed(0)

class LSTM_Model(nn.Module):
    def __init__(self , input_len, hidden_len = 51, mem_col = 20, mem_row = 128, num_heads = 1):
        super(LSTM_Model, self).__init__()

        self.mem_row = mem_row
        self.mem_col = mem_col

        heads = nn.ModuleList([])
        memory = NTMMemory(mem_row, mem_col)

        self.num_heads = num_heads

        for i in range(num_heads):
            heads += [
                NTMReadHead(memory, hidden_len),
                NTMWriteHead(memory, hidden_len)
            ]

        self.memory = memory

        self.input_len = input_len
        self.lstm1 = nn.LSTMCell(self.input_len, self.hidden_len)
        self.lstm2 = nn.LSTMCell(self.hidden_len, self.hidden_len)

        self.linear = nn.Linear(self.hidden_len + (self.mem_col*self.num_heads) , 1)

        # Initialize the initial previous read values to random biases
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = torch.randn(1, self.M) * 0.01
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "At least 1 R-Head"

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 51, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            reads = []
            heads_states = []
            for head, prev_head_state in zip(self.heads, prev_heads_states):
                if head.is_read_head():
                    r, head_state = head(controller_outp, prev_head_state)
                    reads += [r]
                else:
                    head_state = head(controller_outp, prev_head_state)
                heads_states += [head_state]

            # Generate Output
            inp2 = torch.cat([controller_outp] + reads, dim=1)
            o = F.sigmoid(self.fc(inp2))

            output = self.linear(h_t2)
            outputs += [output]

        # for i in range(future):  # if we should predict the future
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #
        #     output = self.linear(h_t2)
        #     outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs