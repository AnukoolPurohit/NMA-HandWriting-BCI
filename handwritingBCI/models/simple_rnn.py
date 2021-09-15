import torch
from torch import nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 1, bidirectional: bool = False,
                 dropout: float = 0.) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout

        self.num_directions = int(self.bidirectional) + 1

        self.rnn = nn.LSTM(input_size, hidden_size,
                           num_layers=self.num_layers,
                           bidirectional=self.bidirectional,
                           batch_first=True,
                           dropout=dropout)

        self.fc_input_size = self.num_directions * self.hidden_size * self.num_layers
        self.fc = nn.Linear(in_features=self.fc_input_size,
                            out_features=output_size)
        return

    @staticmethod
    def concat_hidden(hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = [hidden_state[i] for i in range(len(hidden_state))]
        hidden_state = torch.cat(hidden_state, dim=1)
        return hidden_state

    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        rnn_outputs, (hidden_state, cell_state) = self.rnn(input_sequences)
        hidden_state = self.concat_hidden(hidden_state)
        hidden_state = F.relu(hidden_state)
        output = self.fc(hidden_state)
        return output
