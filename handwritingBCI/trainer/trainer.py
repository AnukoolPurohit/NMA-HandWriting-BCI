import torch
from torch import nn
from tqdm.auto import tqdm
from handwritingBCI.trainer.logger import Logger
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
            self.logger = Logger()
        return

    def one_batch(self, xb: torch.Tensor, yb: torch.Tensor):
        xb, yb = xb.to(self.device), yb.to(self.device)

        if self.model.training:
            self.optimizer.zero_grad()

        predictions = self.model(xb)
        loss = self.loss_func(predictions, yb)
        self.logger.update_loss(self.model.training, loss.cpu().detach())

        if self.model.training:
            loss.backward()
            self.optimizer.step()

        return

    def all_batches(self):
        if self.model.training:
            dl = self.data.train_dl
        else:
            dl = self.data.valid_dl

        for xb, yb in tqdm(dl, leave=False):
            self.one_batch(xb, yb)
        return

    def train_mode(self):
        self.model.train()
        self.all_batches()
        return

    def valid_mode(self):
        self.model.eval()
        with torch.no_grad():
            self.all_batches()

    def tune(self, epochs: int = 5):
        self.model.to(self.device)
        self.loss_func.to(self.device)
        for epoch in tqdm(range(epochs)):
            self.train_mode()
            self.valid_mode()
            self.logger.epoch_complete()
