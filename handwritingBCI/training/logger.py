import matplotlib.pyplot as plt
from handwritingBCI.training.metrics import Metrics, Loss, Accuracy


class Logger:
    def __init__(self, trainer, metrics: list = [Loss, Accuracy]):
        self.trainer = trainer
        self.epoch = 0
        self.train_metrics = Metrics(self.trainer, metrics)
        self.valid_metrics = Metrics(self.trainer, metrics)
        return

    def update(self):
        if self.trainer.model.training:
            self.train_metrics.update()
        else:
            self.valid_metrics.update()
        return

    def reset_epoch_metrics(self):
        self.train_metrics.reset_epoch_metric()
        self.valid_metrics.reset_epoch_metric()
        return

    def epoch_complete(self):
        self.epoch += 1
        self.train_metrics.epoch_complete()
        self.valid_metrics.epoch_complete()
        self.reset_epoch_metrics()
        return

    def plot(self, metric="loss", text_size=25, figure_size=(15, 15)):
        plt.figure(figsize=figure_size)
        self.train_metrics.plot(metric)
        self.valid_metrics.plot(metric)
        plt.grid()
        plt.axhline(0, color="black")
        plt.axvline(0, color="black")
        plt.title(f"Training vs Validation {metric.title()}", size=text_size)
        plt.xlabel("Epochs", size=text_size)
        plt.ylabel("Cross Validation Loss", size=text_size)
        plt.legend([f"Training {metric.title()}", f"Validation {metric.title()}"])
        plt.show()
