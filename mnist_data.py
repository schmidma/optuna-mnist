import os

import lightning
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST


class MnistDataModule(lightning.LightningDataModule):
    def __init__(
        self, data_dir=None, batch_size=128, train_val_split=0.8, num_workers=0
    ):
        super().__init__()
        self.data_dir = data_dir or os.getcwd()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_val_split = train_val_split
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == "fit" or stage == "validate":
            full = MNIST(
                self.data_dir,
                train=True,
                download=True,
                transform=self.transform,
            )
            train_set_size = int(len(full) * self.train_val_split)
            val_set_size = len(full) - train_set_size
            self.data_train, self.data_val = random_split(
                full, [train_set_size, val_set_size]
            )

        if stage == "test":
            self.data_test = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

        if stage == "predict":
            self.data_predict = MNIST(
                self.data_dir,
                train=False,
                download=True,
                transform=self.transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
