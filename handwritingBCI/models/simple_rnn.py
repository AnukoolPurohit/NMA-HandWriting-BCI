from torch import nn
import torch.nn.functional as F


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        return

    def forward(self, input_sequences):
        rnn_outputs, (hidden_state, cell_state) = self.rnn(input_sequences)
        hidden_state = hidden_state.squeeze(0)
        hidden_state = F.relu(hidden_state)
        output = self.fc(hidden_state)
        return output
