import json
import math

import numpy as np
import pytorch_lightning as pl
import torch
from datamodule.main import VideoDataset
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import average_precision_score, recall_score
from torch.utils.data import DataLoader

from module.losses import cross_entropy_loss
from module.metrics import calculate_gap


class VideoClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # to save config in checkpoint
        self.save_hyperparameters()
        self.hparams.learning_rate = config["learning_rate"]
        self.hparams.batch_size = config["batch_size"]

        # --------------------
        # read label_dict
        # ---------------------
        label_id_path = "./dataset/label_id.txt"
        self.label_dict = {"label_name_to_idx": dict(), "label_idx_to_name": list()}
        with open(label_id_path, "r") as file:
            for row in file.readlines():
                label_name, label_id = row.split()

                self.label_dict["label_name_to_idx"][label_name] = int(label_id)
                self.label_dict["label_idx_to_name"].append(label_name)

        # basic class member
        self.config = config

        # define loss function
        self.loss = cross_entropy_loss

        # define callbacks
        self.checkpoint_callback = ModelCheckpoint(
            monitor="val_GAP",
            filename="ckpt_{step:05d}_{val_GAP:.4f}",
            save_top_k=1,
            mode="max",
        )

        self.early_stop_callback = EarlyStopping(
            monitor="val_GAP",
            min_delta=0.00,
            patience=self.config["early_stopping_patience"],
            verbose=False,
            mode="max",
        )

        self.lr_monitor = LearningRateMonitor(logging_interval="step")

        self.callbacks = [
            self.checkpoint_callback,
            self.early_stop_callback,
            self.lr_monitor,
        ]

    def on_train_start(self):
        # at the beginning of training
        # log files using comet logger
        self.logger.experiment.log_code(file_name="main.py")
        self.logger.experiment.log_code(folder="datamodule")
        self.logger.experiment.log_code(folder="model")
        self.logger.experiment.log_code(folder="module")

    def _metric_calculator(self, gt_label, predicted_label):
        temp_predicted_label = np.copy(predicted_label.cpu().detach().numpy())
        temp_predicted_label[temp_predicted_label < 0.5] = 0
        temp_predicted_label[temp_predicted_label >= 0.5] = 1

        try:
            precision = average_precision_score(
                gt_label.cpu().reshape((-1)), temp_predicted_label.reshape((-1))
            )
        except Exception as e:
            precision = 0.0

        try:
            recall = recall_score(gt_label.cpu(), temp_predicted_label, average="micro")
        except Exception as e:
            recall = 0.0

        try:
            gap = calculate_gap(
                predicted_label.cpu().detach().numpy(), gt_label.cpu().numpy()
            )
        except Exception as e:
            gap = 0

        return {"GAP": gap, "precision@0.5": precision, "recall@0.5": recall}

    def training_step(self, batch, batch_idx):
        result = self.forward(batch)
        predicted_label = result["predicted_label"]
        gt_label = batch["label"]
        gt_label = (
            torch.transpose(torch.stack(gt_label), 0, 1)
            .type(torch.float)
            .to(self.device)
        )

        loss_value = self.loss(gt_label, predicted_label)
        self.log("train_loss", loss_value, prog_bar=True)

        train_metrics = self._metric_calculator(gt_label, predicted_label)
        for item in train_metrics:
            self.log(f"train_{item}", train_metrics[item], prog_bar=True)

        if math.isnan(loss_value):
            return None

        return loss_value

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        predicted_label = result["predicted_label"]
        gt_label = batch["label"]
        gt_label = (
            torch.transpose(torch.stack(gt_label), 0, 1)
            .type(torch.float)
            .to(self.device)
        )

        return {"predicted_label": predicted_label, "gt_label": gt_label}

    def validation_epoch_end(self, validation_step_outputs):
        gt_label = torch.zeros((0, self.config["num_classes"]))
        predicted_label = torch.zeros((0, self.config["num_classes"]))

        for batch_result in validation_step_outputs:
            temp_gt_label = batch_result["gt_label"]
            gt_label = torch.cat([gt_label, temp_gt_label], dim=0)

            predicted_label = torch.cat(
                [predicted_label, batch_result["predicted_label"]], dim=0
            )

        val_metrics = self._metric_calculator(gt_label, predicted_label)
        for item in val_metrics:
            self.log(f"val_{item}", val_metrics[item])

        return {"predicted_label": predicted_label, "gt_label": gt_label}

    def test_step(self, batch, batch_idx):
        result = self.forward(batch)
        predicted_label = result["predicted_label"]
        video_name = batch["video_full_name"]

        return {"predicted_label": predicted_label, "video_name": video_name}

    def test_epoch_end(self, test_step_outputs):
        result = dict()

        predicted_label = torch.zeros((0, self.config["num_classes"]))
        video_name_list = list()

        for batch_result in test_step_outputs:
            predicted_label = torch.cat(
                [predicted_label, batch_result["predicted_label"]], dim=0
            )
            video_name_list += batch_result["video_name"]

        predicted_label = predicted_label.numpy()

        for video_name, labels in zip(video_name_list, predicted_label):
            temp_result = {"labels": list(), "scores": list()}
            temp_dict = dict()

            for idx, score in enumerate(labels):
                if score > 0:
                    temp_dict[self.label_dict["label_idx_to_name"][idx]] = score
            temp_dict = dict(
                sorted(temp_dict.items(), key=lambda item: item[1], reverse=True)
            )

            temp_result["labels"] = list(temp_dict.keys())[:20]
            temp_result["scores"] = list()
            for item_score in list(temp_dict.values())[:20]:
                temp_result["scores"].append(str(item_score))

            result[video_name] = dict()
            result[video_name]["result"] = list()
            result[video_name]["result"].append(temp_result)

        # write test result into json file
        ckpt_hash = self.config.restore_path.split("/")[-3]
        with open(
            f"run_test/tagging_5k_{ckpt_hash}.json", "w+", encoding="utf-8"
        ) as outfile:
            json.dump(result, outfile, ensure_ascii=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.config["weight_decay"],
        )

        if self.config.lr_scheduler is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config["lr_scheduler"]["mode"],
                factor=self.config["lr_scheduler"]["factor"],
                patience=self.config["lr_scheduler"]["patience"],
                min_lr=self.config["lr_scheduler"]["min_lr"],
            )
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer, T_max=150
            # )
            #             scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=1, eta_min=5e-7)

            # ? maybe we need SGD to optimize
            # optimizer = torch.optim.SGD(
            #     self.model.parameters(),
            #     lr=0.1,
            #     momentum=self.config['momentum'],
            #     weight_decay=self.config['weight_decay']
            # )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_GAP",
                    "interval": "step",
                    "frequency": self.config["trainer"]["val_check_interval"],
                    #                     "frequency": 1
                },
            }
        else:
            return optimizer

    # ------------------
    # for built-in data loader
    # ------------------
    def setup(self, stage=None):
        # called on every GPU
        self.data_loader_dict = dict()

        if stage == "fit" or stage is None:
            self.data_loader_dict["train"] = VideoDataset(
                dataset_type="train", config=self.config
            )
            self.data_loader_dict["val"] = VideoDataset(
                dataset_type="val", config=self.config
            )

        if stage == "test" or stage is None:
            self.data_loader_dict["test"] = VideoDataset(
                dataset_type="test", config=self.config
            )

        if stage == "val":
            self.data_loader_dict["val"] = VideoDataset(
                dataset_type="val", config=self.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_loader_dict["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_loader_dict["val"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=8,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_loader_dict["test"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=8,
        )
