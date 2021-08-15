from handwritingBCI.data.datasets import NeuroDataset


class Databunch:
    def __init__(self, train_dl, valid_dl, test_dl=None,
                 x_type=None, y_type=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.x_type = x_type
        self.y_type = y_type
        return

    def __repr__(self):
        message = f"train_dl length {len(self.train_dl)}"

        if self.x_type and self.y_type:
            message += f": Inputs of {self.x_type}"
            message += f" Outputs of {self.y_type}"
        message += f"\nvalid_dl length {len(self.valid_dl)}"

        if self.x_type and self.y_type:
            message += f": Inputs of {self.x_type}"
            message += f" Outputs of {self.y_type}"

        if self.test_dl:
            message += f"\ntest_dl length {len(self.test_dl)}"
            if self.x_type and self.y_type:
                message += f": Inputs of {self.x_type}"
                message += f" Outputs of {self.y_type}"
        return message

    @classmethod
    def from_neuro_dataset(cls, neuro_dataset: NeuroDataset,
                           test_size: float = 0.1, batch_size: int = 32,
                           generator=None) -> object:
        sample_x, sample_y = neuro_dataset[0]
        train_dl, valid_dl = neuro_dataset.get_dataloaders(test_size,
                                                           batch_size,
                                                           generator)
        return cls(train_dl, valid_dl, None, type(sample_x), type(sample_y))
