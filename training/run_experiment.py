from torch import nn
from handwritingBCI.data.datasets import NeuroDataset
from handwritingBCI.data.utils.transforms import get_simple_transforms
from handwritingBCI.data.databunch import Databunch
from handwritingBCI.pathlib_extension import Path
from handwritingBCI.training.trainer import Trainer
from handwritingBCI.training.metrics import Loss, Accuracy
from handwritingBCI.models.simple_rnn import SimpleRNN


PATH = Path("/home/anukoolpurohit/Documents/AnukoolPurohit/Datasets/HandwritingBCI/handwriting-bci/handwritingBCIData")

neuro_dataset = NeuroDataset.from_path(PATH, get_transforms=get_simple_transforms)

batch_size = 32
test_size = 0.1

data = Databunch.from_neuro_dataset(neuro_dataset, test_size=test_size, batch_size=batch_size)
model = SimpleRNN(input_size=192, hidden_size=64, output_size=31, num_layers=1, bidirectional=False)

epochs = 100
lr = 1e-3
loss_func = nn.CrossEntropyLoss()

trainer = Trainer(data, model, loss_func=loss_func, lr=lr, metrics=[Loss, Accuracy])
trainer.tune(epochs=epochs)
