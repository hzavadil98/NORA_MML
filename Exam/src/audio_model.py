"""
===============================================================================
PyTorch Lightning Audio Classification Components
===============================================================================

This module implements PyTorch Lightning components for audio only classification task,
including data loading, model architecture, and training utilities. It provides
a complete pipeline for training and evaluating 1D convolutional neural networks
on time-series audio data with automatic train/validation/test splits.

Classes:
--------
- Audio_Dataset: Custom PyTorch Dataset for loading audio tensors and labels
- Audio_Dataloader: PyTorch Lightning DataModule for managing audio data splits
- SamePadConv1d: 1D convolution with same padding for maintaining sequence length
- Conv1Net: Simple 1D CNN architecture for audio classification
- Audio_Model: PyTorch Lightning Module wrapping Conv1Net with training logic

Features:
---------
- Automatic train/validation/test data splitting with stratification
- Configurable dropout, batch normalization, and model dimensions
- Built-in metrics tracking (accuracy, loss, confusion matrix)
- Weights & Biases integration for experiment logging
- Feature extraction capability for multimodal applications
- Automatic dtype conversion to float32 for audio data

Usage:
------
```python
from pathlib import Path
from audio_model import Audio_Dataloader, Audio_Model

# Initialize data module
data_module = Audio_Dataloader(
    path_to_data=Path("data/"),
    batch_size=32,
    val_ratio=0.2
)

# Initialize model
model = Audio_Model(
    num_classes=10,
    learning_rate=0.001,
    inner_size=32
)

# Train with PyTorch Lightning Trainer
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, data_module)
```
"""

import math
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
    """
    Custom PyTorch Dataset for loading audio tensors and their corresponding labels.

    This dataset class handles pre-loaded audio tensor data and provides optional
    transformations during data loading. It's designed to work with 1D audio tensors
    that have been saved as .pth files.

    Args:
        data_tensor (torch.Tensor): Tensor containing audio data (1D time series)
        labels_tensor (torch.Tensor): Tensor containing corresponding labels
        transform (callable, optional): Optional transform to be applied to samples

    Example:
        >>> audio_data = torch.load("audio.pth")
        >>> labels = torch.load("labels.pth")
        >>> dataset = Audio_Dataset(audio_data, labels, transform=some_transform)
        >>> sample, label = dataset[0]
    """

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
    """
    PyTorch Lightning DataModule for managing audio dataset loading and splitting.

    This DataModule automatically handles train/validation/test splits from the
    provided training data, loads test data separately, and creates appropriate
    DataLoaders for each split. It expects audio data to be stored as .pth tensor
    files and automatically converts data to float32 dtype for compatibility.

    Args:
        path_to_data (Path): Path to directory containing the .pth audio data files
        batch_size (int, optional): Batch size for all DataLoaders. Defaults to 32.
        val_ratio (float, optional): Fraction of training data to use for validation.
            Defaults to 0.2.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 4.
        transform (callable, optional): Transform to apply to training data only.
            Defaults to None.

    Expected Data Files:
        - training_audio.pth: Training audio tensors (1D time series)
        - training_audio_labels.pth: Training labels
        - test_audio.pth: Test audio tensors
        - test_audio_labels.pth: Test labels

    Example:
        >>> from pathlib import Path
        >>> data_module = Audio_Dataloader(
        ...     path_to_data=Path("data/"),
        ...     batch_size=64,
        ...     val_ratio=0.15
        ... )
        >>> trainer.fit(model, data_module)
    """

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


class SamePadConv1d(torch.nn.Module):
    """
    1D convolution with same padding to maintain sequence length.

    This module implements 1D convolution that automatically calculates and
    applies padding to maintain the same output sequence length as the input
    (necessary for stride > 1).

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Convolution kernel size
        stride (int, optional): Convolution stride. Defaults to 1.
        dilation (int, optional): Convolution dilation. Defaults to 1.
        bias (bool, optional): Whether to use bias. Defaults to True.

    Example:
        >>> conv = SamePadConv1d(1, 32, kernel_size=100, stride=4)
        >>> output = conv(audio_tensor)  # Maintains sequence proportions
    """

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
    """
    Simple 1D Convolutional Neural Network for audio classification.

    This network consists of two strided 1D convolutional layers with ReLU activations
    and same-padding convolutions, followed by two fully connected layers.
    Designed for processing 1D time-series audio data with parameters configured
    for series of length 8000.

    Architecture:
        - SamePadConv1d(1, inner_size, kernel_size=100, stride=4) + ReLU
        - SamePadConv1d(inner_size, inner_size*2, kernel_size=25, stride=4) + Dropout + ReLU
        - Linear(inner_size*2*500, inner_size*4) + Dropout + ReLU
        - Linear(inner_size*4, num_classes)

    Args:
        num_classes (int): Number of output classes
        drop_rate (float, optional): Dropout probability. Defaults to 0.5.
        inner_size (int, optional): Base number of filters/features. Defaults to 32.

    Example:
        >>> model = Conv1Net(num_classes=10, inner_size=64, drop_rate=0.3)
        >>> output = model(torch.randn(32, 1, 8000))  # batch_size=32, 8000 samples
        >>> features = model.get_inner_features(torch.randn(32, 1, 8000))
    """

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
    """
    PyTorch Lightning Module for audio classification using Conv1Net architecture.

    This Lightning module wraps the Conv1Net model and provides complete training,
    validation, and testing functionality for audio classification tasks. It includes
    automatic metric tracking, confusion matrix generation, and Weights & Biases
    logging integration.

    Features:
        - Cross-entropy loss for multi-class classification
        - Accuracy and confusion matrix metrics
        - Adam optimizer with configurable learning rate
        - Automatic confusion matrix visualization and logging
        - Hyperparameter saving for reproducibility
        - Feature extraction capability for multimodal applications

    Args:
        num_classes (int): Number of output classes
        inner_size (int, optional): Base number of filters/features for Conv1Net.
            Defaults to 32.
        drop_rate (float, optional): Dropout probability for the underlying Conv1Net.
            Defaults to 0.5.
        learning_rate (float, optional): Learning rate for Adam optimizer.
            Defaults to 0.001.

    Example:
        >>> model = Audio_Model(num_classes=10, learning_rate=0.001)
        >>> trainer = pl.Trainer(max_epochs=50)
        >>> trainer.fit(model, datamodule)
        >>> trainer.test(model, datamodule)
    """

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
