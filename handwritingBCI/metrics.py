import torch
import matplotlib.pyplot as plt


class Metric:
    def __init__(self, trainer):
        self.trainer = trainer
        self.metric, self.epoch_metric = [], []
        return

    def update(self):
        raise NotImplemented

    def epoch_complete(self):
        self.calculate_epoch_metric()
        self.reset_epoch_metric()
        return

    def calculate_epoch_metric(self):
        raise NotImplemented

    def reset_epoch_metric(self):
        self.epoch_metric = []
        return


class Loss(Metric):
    def update(self):
        metric = self.trainer.loss
        if isinstance(metric, float):
            metric = torch.Tensor([metric])
        metric = metric.cpu().detach()
        self.epoch_metric.append(metric)
        return

    def calculate_epoch_metric(self):
        epoch_metric = torch.stack(self.epoch_metric)
        mean_epoch_loss = epoch_metric.mean()
        self.metric.append(mean_epoch_loss)
        return


class Accuracy(Metric):
    def update(self):
        predictions = self.trainer.predictions
        targets = self.trainer.yb.cpu().detach()
        if isinstance(predictions, float):
            pass
        predictions = predictions.cpu().detach()
        predictions = torch.argmax(predictions, dim=1)
        result = predictions == targets
        accuracy = result.sum()/len(result)
        self.epoch_metric.append(accuracy)

    def calculate_epoch_metric(self):
        epoch_metric = torch.stack(self.epoch_metric)
        mean_epoch_loss = epoch_metric.mean()
        self.metric.append(mean_epoch_loss)
        return


class Metrics:
    def __init__(self, trainer, metrics: list):
        self.trainer = trainer
        self.metrics = metrics
        self._add_attributes()
        return

    def _add_attributes(self):
        for metric in self.metrics:
            self.__setattr__(metric.__name__.lower(), metric(self.trainer))

    def _get_attribute(self, attribute:str):
        return getattr(self, attribute)

    def update(self):
        for metric in self.metrics:
            attribute = getattr(self, metric.__name__.lower())
            attribute.update()

    def epoch_complete(self):
        for metric in self.metrics:
            attribute = getattr(self, metric.__name__.lower())
            attribute.epoch_complete()

    def reset_epoch_metric(self):
        for metric in self.metrics:
            attribute = getattr(self, metric.__name__.lower())
            attribute.reset_epoch_metric()

    def plot(self, metric="loss"):
        metric = self._get_attribute(metric)
        plt.plot(metric.metric)


