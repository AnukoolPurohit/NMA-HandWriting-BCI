import torch
from torch import nn
from tqdm.auto import tqdm
from handwritingBCI.training.logger import Logger
from handwritingBCI.utils import DEVICE
from handwritingBCI.data.databunch import Databunch


class Trainer:
    def __init__(self, data: Databunch,
                 model: nn.Module,
                 loss_func: nn.Module,
                 lr: float = 1e-3,
                 optimizer: str = "Adam",
                 device=DEVICE,
                 logger=None,
                 ) -> None:

        self.data = data
        self.model = model
        self.lr = lr
        self.loss_func = loss_func
        self.optimizer = getattr(torch.optim, optimizer)
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.lr)
        self.device = device
        if logger is None:
            self.logger = Logger(self)

        # Placeholders
        self.loss = 0.
        self.predictions = 0.
        self.xb: torch.Tensor = torch.tensor([0.])
        self.yb: torch.Tensor = torch.tensor([0.])
        return

    def tune(self, epochs: int = 5):
        self.model.to(self.device)
        self.loss_func.to(self.device)
        progress_bar = tqdm(range(epochs))
        for epoch in progress_bar:
            progress_bar.set_description(f"Epoch: {epoch}")
            self.train_mode()
            self.valid_mode()
            self.logger.epoch_complete()

    def train_mode(self):
        self.model.train()
        self.all_batches()
        return

    def valid_mode(self):
        self.model.eval()
        with torch.no_grad():
            self.all_batches()

    def all_batches(self):
        if self.model.training:
            dl = self.data.train_dl
        else:
            dl = self.data.valid_dl
        progress_bar = tqdm(dl, leave=False)
        for xb, yb in progress_bar:
            progress_bar.set_description(f"Loss: {self.loss:.2f}")
            self.xb, self.yb = xb, yb
            self.one_batch()
        return

    def one_batch(self):
        self.xb, self.yb = self.xb.to(self.device), self.yb.to(self.device)

        if self.model.training:
            self.optimizer.zero_grad()

        self.predictions = self.model(self.xb)
        self.loss = self.loss_func(self.predictions, self.yb)
        self.logger.update()

        if self.model.training:
            self.loss.backward()
            self.optimizer.step()

        return



