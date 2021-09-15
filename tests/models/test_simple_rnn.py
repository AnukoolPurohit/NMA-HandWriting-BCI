import torch
import pdb
import random
import pytest
from handwritingBCI.models.simple_rnn import SimpleRNN


@pytest.fixture()
def test_batch():
    batch_size = random.randint(8, 128)
    num_neurons = random.randint(10, 300)
    time_steps = random.randint(10, 300)
    return torch.rand((batch_size, time_steps, num_neurons))


def test_fixture(test_batch):
    assert len(test_batch.shape) == 3
    return


class TestSimpleRNN:

    def test_output_size(self, test_batch):
        output_categories = random.randint(10, 100)
        hidden_size = random.randint(8, 128)
        num_layers = random.randint(1, 10)
        bidirectional = random.choice([True, False])

        simple_rnn = SimpleRNN(input_size=test_batch.shape[2],
                               hidden_size=hidden_size,
                               output_size=output_categories,
                               num_layers=num_layers,
                               bidirectional=bidirectional
                               )
        output = simple_rnn(test_batch)

        assert output.shape == (test_batch.shape[0], output_categories)
        return

    def test_output_size_simple(self):
        sample_input = torch.rand((32, 201, 192))
        simple_rnn = SimpleRNN(input_size=192,
                               hidden_size=64,
                               output_size=31)
        output = simple_rnn(sample_input)

        assert output.shape == (32, 31)
        return

    def test_output_size_batch_size_one(self):
        sample_input = torch.rand((1, 201, 192))
        simple_rnn = SimpleRNN(input_size=192,
                               hidden_size=64,
                               output_size=31)
        output = simple_rnn(sample_input)

        assert output.shape == (1, 31)
        return
