import json
import math
import os

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class MockDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __len__(self):
        return 4500

    def __getitem__(self, index):
        return {
            "video_feature": torch.rand((300, 1024)),
            "text_feature": torch.rand((300, 768)),
            "audio_feature": torch.rand((300, 128)),
            "video_mask": torch.rand((300)),
            "text_mask": torch.rand((300)),
            "audio_mask": torch.rand((300)),
            "label": [0] * 82,
        }


class MockDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train = MockDataset()
            self.val = MockDataset()

        if stage == "test" or stage is None:
            self.test = MockDataset()

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.config["batch_size"], shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.config["batch_size"], shuffle=False)

    def test_dataloader(self):
        return DataLoader(
            self.test, batch_size=self.config["batch_size"], shuffle=False
        )
