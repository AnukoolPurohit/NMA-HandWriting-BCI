import torch
import matplotlib.pyplot as plt


class Logger:
    def __init__(self):
        self.epoch = 0
        self.train_losses, self.valid_losses = [], []
        self.epoch_train_losses, self.epoch_valid_losses = [], []
        return

    def update_loss(self, is_training_loss: bool, loss):
        if is_training_loss:
            self.epoch_train_losses.append(loss)
        else:
            self.epoch_valid_losses.append(loss)
        return

    def reset_epoch_losses(self):
        self.epoch_train_losses, self.epoch_valid_losses = [], []
        return

    def epoch_complete(self):
        self.epoch += 1
        epoch_train_losses = torch.stack(self.epoch_train_losses)
        epoch_valid_losses = torch.stack(self.epoch_valid_losses)

        mean_train_loss = epoch_train_losses.mean()
        mean_valid_loss = epoch_valid_losses.mean()

        self.train_losses.append(mean_train_loss)
        self.valid_losses.append(mean_valid_loss)

        self.reset_epoch_losses()
        return

    def plot_loss(self, text_size=25, figure_size=(15, 15)):
        plt.figure(figsize=figure_size)
        plt.plot(self.train_losses)
        plt.plot(self.valid_losses)
        plt.grid()
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.title("Training vs Validation Loss", size=text_size)
        plt.xlabel("Epochs", size=text_size)
        plt.ylabel("Cross Validation Loss", size=text_size)
        plt.legend(["Training Loss", "Validation Loss"])
        plt.show()
