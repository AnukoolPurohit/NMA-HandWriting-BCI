from torch import nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.num_directions = int(self.bidirectional) + 1

        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        return

    def forward(self, input_sequences):
        rnn_outputs, (hidden_state, cell_state) = self.rnn(input_sequences)
        hidden_state = hidden_state.squeeze(0)
        hidden_state = F.relu(hidden_state)
        output = self.fc(hidden_state)
        return output
