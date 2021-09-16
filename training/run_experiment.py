import argparse
from torch import nn
from handwritingBCI.data.datasets import NeuroDataset
from handwritingBCI.data.utils.transforms import get_simple_transforms
from handwritingBCI.data.databunch import Databunch
from handwritingBCI.pathlib_extension import Path
from handwritingBCI.training.trainer import Trainer
from handwritingBCI.training.metrics import Loss, Accuracy
from handwritingBCI.models.simple_rnn import SimpleRNN


PATH = Path("/home/anukoolpurohit/Documents/AnukoolPurohit/Datasets/HandwritingBCI/handwriting-bci/handwritingBCIData")
BATCH_SIZE = 32
VALID_SIZE = 0.1
EPOCHS = 10
LR = 1e-3


def main():
    parser = _setup_parser()
    args = parser.parse_args()

    neuro_dataset = NeuroDataset.from_path(PATH, get_transforms=get_simple_transforms)
    data = Databunch.from_neuro_dataset(neuro_dataset, test_size=args.valid_size, batch_size=args.batch_size)
    model = SimpleRNN(input_size=192, hidden_size=64, output_size=31, num_layers=1, bidirectional=False)
    loss_func = nn.CrossEntropyLoss()
    trainer = Trainer(data, model, loss_func=loss_func, lr=args.learning_rate, metrics=[Loss, Accuracy])
    trainer.tune(epochs=args.epochs)


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",
                        help=f"Number of epochs you want to train the network (default = {EPOCHS})",
                        type=int, default=EPOCHS)
    parser.add_argument("-bs", "--batch_size",
                        help=f"Batch size (default = {BATCH_SIZE})",
                        type=int, default=BATCH_SIZE)
    parser.add_argument("--valid_size",
                        help=f"The percentage of training set to kept out as validation set (default = {VALID_SIZE})",
                        type=float, default=VALID_SIZE)
    parser.add_argument("-lr", "--learning_rate",
                        help=f"learning rate (default = {LR})",
                        type=float, default=LR)
    return parser


if __name__ == "__main__":
    main()
