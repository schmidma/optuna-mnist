import lightning
import torch
import torch.nn as nn
import torchmetrics


class MnistModel(lightning.LightningModule):
    def __init__(self, hyperparameters):
        super().__init__()

        self.learning_rate = hyperparameters["learning_rate"]

        self.convolution = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
        )

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.convolution(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return nn.functional.softmax(x, dim=1)

    def cross_entropy_loss(self, logits, labels):
        return nn.functional.nll_loss(logits, labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.train_accuracy(logits, y)
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": self.train_accuracy,
            },
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.val_accuracy(logits, y)
        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": self.val_accuracy,
            },
            on_step=True,
            on_epoch=True,
        )
        return loss

    def test_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        self.test_accuracy(logits, y)
        self.log_dict(
            {
                "test_loss": loss,
                "test_accuracy": self.test_accuracy,
            },
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
