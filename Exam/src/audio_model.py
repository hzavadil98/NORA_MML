"""
===============================================================================
PyTorch Lightning Dataloaders and Model Definitions
===============================================================================

This module contains the implementation of PyTorch Lightning dataloaders and
model definitions. It provides reusable components for data loading and
model architecture, facilitating efficient training and evaluation workflows
using the PyTorch Lightning framework.

Classes and Functions:
----------------------
- Dataloaders for preparing and batching datasets.
- Model classes defining neural network architectures compatible with Lightning.

Usage:
------
Import the required dataloader or model class and integrate it into your
Lightning training pipeline.

Author: [Your Name]
Date: [Date]
"""

import math
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import Accuracy, MulticlassConfusionMatrix


class Audio_Dataset(Dataset):
    def __init__(self, data_tensor, labels_tensor, transform=None):
        super().__init__()
        self.data = data_tensor
        self.labels = labels_tensor
        self.transform = transform

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Load and return a sample from the dataset
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


class Audio_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        path_to_data: Path,
        batch_size=32,
        val_ratio=0.2,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        # Load the original audio data and labels
        train_data = torch.load(self.path_to_data / "training_audio.pth").to(
            dtype=torch.float32
        )
        train_labels = torch.load(self.path_to_data / "training_audio_labels.pth")

        test_data = torch.load(self.path_to_data / "test_audio.pth").to(
            dtype=torch.float32
        )
        test_labels = torch.load(self.path_to_data / "test_audio_labels.pth")

        # Split the training data into training and validation sets
        train_data, validation_data, train_labels, validation_labels = train_test_split(
            train_data,
            train_labels,
            test_size=val_ratio,
            random_state=42,
            stratify=train_labels,
        )

        # Create datasets for training, validation, and testing
        self.train_dataset = Audio_Dataset(
            train_data, train_labels, transform=self.transform
        )
        self.val_dataset = Audio_Dataset(
            validation_data, validation_labels, transform=None
        )
        self.test_dataset = Audio_Dataset(test_data, test_labels, transform=None)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


class Lambda(torch.nn.Module):
    def __init__(self, f):
        super(Lambda, self).__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Residual(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(Residual, self).__init__()

        self.bottleneck = torch.nn.Conv1d(
            in_channels=in_size,
            out_channels=4 * out_size,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=4 * out_size)

    def forward(self, x, y):
        y = y + self.batch_norm(self.bottleneck(x))
        y = torch.nn.functional.relu(y)
        return y


class Inception(torch.nn.Module):
    def __init__(self, in_size, inner_size, filters=[11, 21, 41], drop_rate=0.5):
        super(Inception, self).__init__()

        self.bottleneck1 = torch.nn.Conv1d(
            in_channels=in_size,
            out_channels=inner_size,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv1 = torch.nn.Conv1d(
            in_channels=inner_size,
            out_channels=inner_size,
            kernel_size=filters[0],
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv2 = torch.nn.Conv1d(
            in_channels=inner_size,
            out_channels=inner_size,
            kernel_size=filters[1],
            stride=1,
            padding="same",
            bias=False,
        )

        self.conv3 = torch.nn.Conv1d(
            in_channels=inner_size,
            out_channels=inner_size,
            kernel_size=filters[2],
            stride=1,
            padding="same",
            bias=False,
        )

        self.max_pool = torch.nn.MaxPool1d(
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.bottleneck2 = torch.nn.Conv1d(
            in_channels=in_size,
            out_channels=inner_size,
            kernel_size=1,
            stride=1,
            padding="same",
            bias=False,
        )

        self.batch_norm = torch.nn.BatchNorm1d(num_features=4 * inner_size)

        self.dropout = torch.nn.Dropout1d(p=drop_rate)

    def forward(self, x):
        x0 = self.bottleneck1(x)
        x1 = self.dropout(self.conv1(x0))
        x2 = self.dropout(self.conv2(x0))
        x3 = self.dropout(self.conv3(x0))
        x4 = self.dropout(self.bottleneck2(self.max_pool(x)))
        y = torch.concat([x1, x2, x3, x4], dim=1)
        y = torch.nn.functional.relu(self.batch_norm(y))
        return y


class Inception_Time(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        in_size,
        inner_size,
        depth,
        filters=[11, 21, 41],
        drop_rate=0.5,
    ):
        super(Inception_Time, self).__init__()
        self.in_size = in_size
        self.inner_size = inner_size
        self.depth = depth
        self.filters = filters
        self.drop_rate = drop_rate

        modules = OrderedDict()

        for f in filters:
            modules[f"conv_{f}"] = torch.nn.Conv1d(
                in_channels=in_size,
                out_channels=inner_size,
                kernel_size=f,
                stride=1,
                padding="same",
                bias=False,
            )

        for d in range(depth):
            modules[f"inception_{d}"] = Inception(
                in_size=in_size if d == 0 else 4 * inner_size,
                inner_size=inner_size,
                filters=filters,
                drop_rate=drop_rate,
            )
            if d % 3 == 2:
                modules[f"residual_{d}"] = Residual(
                    in_size=in_size if d == 2 else 4 * inner_size,
                    out_size=inner_size,
                )

        modules["avg_pool"] = Lambda(f=lambda x: torch.mean(x, dim=-1))

        self.featurizer = torch.nn.Sequential(modules)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=4 * inner_size, out_features=inner_size),
            torch.nn.Linear(in_features=inner_size, out_features=num_classes),
        )

    def forward(self, x):
        y = None
        for d in range(self.depth):
            y = self.featurizer.get_submodule(f"inception_{d}")(x if d == 0 else y)
            if d % 3 == 2:
                y = self.featurizer.get_submodule(f"residual_{d}")(x, y)
                x = y
        y = self.featurizer.get_submodule("avg_pool")(y)
        # y = self.model.get_submodule('linear')(y)
        y = self.classifier(y)
        return y

    def get_inner_features(self, x):
        y = None
        for d in range(self.depth):
            y = self.featurizer.get_submodule(f"inception_{d}")(x if d == 0 else y)
            if d % 3 == 2:
                y = self.featurizer.get_submodule(f"residual_{d}")(x, y)
                x = y
        y = self.featurizer.get_submodule("avg_pool")(y)
        # y = self.model.get_submodule('linear')(y)
        return y


class SamePadConv1d(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True
    ):
        super().__init__()
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        L_in = x.shape[-1]
        effective_kernel = self.dilation * (self.kernel_size - 1) + 1
        L_out = math.ceil(L_in / self.stride)
        padding_needed = max((L_out - 1) * self.stride + effective_kernel - L_in, 0)
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        x = torch.functional.F.pad(
            x, (pad_left, pad_right)
        )  # F.pad takes (left, right)
        return self.conv(x)


class Conv1Net(torch.nn.Module):
    def __init__(self, num_classes, drop_rate=0.5, inner_size=32):
        super().__init__()
        self.inner_size = inner_size
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        self.featurizer = torch.nn.Sequential(
            SamePadConv1d(1, self.inner_size, kernel_size=100, stride=4),
            torch.nn.ReLU(),
            SamePadConv1d(
                self.inner_size, self.inner_size * 2, kernel_size=25, stride=4
            ),
            torch.nn.Dropout(self.drop_rate),
            torch.nn.ReLU(),
        )

        self.fc1 = torch.nn.Linear(self.inner_size * 2 * 500, self.inner_size * 4)
        self.dropout2 = torch.nn.Dropout(self.drop_rate)
        self.relu3 = torch.nn.ReLU()

        self.fc2 = torch.nn.Linear(self.inner_size * 4, num_classes)

    def forward(self, x):
        x = self.featurizer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout2(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def get_inner_features(self, x):
        x = self.featurizer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.dropout2(self.relu3(self.fc1(x)))
        return x


class Audio_Model(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        inner_size=32,
        drop_rate=0.5,
        learning_rate=0.001,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(num_classes=num_classes, task="multiclass")
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        self.num_classes = num_classes
        self.inner_size = inner_size
        self.drop_rate = drop_rate

        self.save_hyperparameters()

        self.model = Conv1Net(
            num_classes=self.num_classes,
            inner_size=self.inner_size,
            drop_rate=self.drop_rate,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", self.accuracy(logits, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", self.accuracy(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.accuracy(logits, y), prog_bar=True)
        self.confusion_matrix.update(torch.argmax(logits, dim=1), y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @rank_zero_only
    def on_test_epoch_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_title("Confusion Matrix")

        # Log confusion matrix to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
