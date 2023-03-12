from typing import Any, Optional

import pytorch_lightning as pl
import torch.backends.mps
import torch.nn.functional as F
import torch.utils.data
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
from pytorch_lightning.callbacks import RichProgressBar, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torch import optim, nn
from torchvision import transforms


def cifar10_normalization():
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )
    return normalize


class CIFAR10Trainer(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model: nn.Module = model
        self.batch_size = 128

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self) -> Any:
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_dict}

    @property
    def train_image_transform(self):
        train_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomCrop(32, padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                cifar10_normalization(),
            ]
        )
        return train_transforms

    @property
    def test_image_transforms(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             cifar10_normalization()
             ])
        return transform

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        # transform = torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.CIFAR10)
        dataset = torchvision.datasets.CIFAR10(root='.', train=True, transform=self.train_image_transform,
                                               download=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        dataset = torchvision.datasets.CIFAR10(root='.', train=False, transform=self.test_image_transforms,
                                               download=True)
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
        )
        return loader

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        image, label = batch
        logits = self.model(image)
        loss = F.cross_entropy(logits, label)

        with torch.no_grad():
            labels_hat = torch.argmax(logits, dim=-1)
            train_acc = torch.sum(label == labels_hat).item() / (len(label) * 1.0)

        self.log_dict({'train_loss': loss, 'train_acc': train_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        image, label = batch
        with torch.no_grad():
            logits = self.model(image)
            labels_hat = torch.argmax(logits, dim=-1)
            loss = F.cross_entropy(logits, label)

            val_acc = torch.sum(label == labels_hat).item() / (len(label) * 1.0)
            # log the outputs!
            self.log_dict({'val_loss': loss, 'val_acc': val_acc}, prog_bar=True)

        return None

    def get_accelerator(self):
        if torch.cuda.is_available():
            accelerator = 'cuda'
        elif torch.backends.mps.is_available():
            accelerator = 'mps'
        else:
            accelerator = 'cpu'

        print(f'Accelerator {accelerator}')

        return accelerator

    def fit(self, max_epochs, batch_size=128):
        self.batch_size = batch_size
        trainer = pl.Trainer(max_epochs=max_epochs, accelerator=self.get_accelerator(), devices=1,
                             callbacks=[RichProgressBar(leave=True),
                                        EarlyStopping(monitor='val_loss', patience=5)],
                             logger=TensorBoardLogger(f'logs/{self.model.__class__.__name__}/'))
        trainer.fit(model=self, train_dataloaders=self.train_dataloader(), val_dataloaders=self.val_dataloader())
