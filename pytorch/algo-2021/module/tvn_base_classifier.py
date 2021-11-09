import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamodule.raw_video import RawVideoDataset
from model.tvn import TVN, TBlock
from model.utils import weights_init
from sklearn.metrics import average_precision_score, recall_score
from torch.utils.data import DataLoader

from module.base import VideoClassifier
from module.metrics import calculate_gap

tvn_config = [
    TBlock(
        "Block1",
        in_channels=3,
        out_channels=32,
        spatial_ksize=3,
        spatial_stride=1,
        temporal_ksize=2,
        temporal_stride=2,
        temporal_pool_type="max",
        cg_ksize=3,
        cg_stride=2,
    ),
    TBlock(
        "Block1",
        in_channels=32,
        out_channels=64,
        spatial_ksize=3,
        spatial_stride=1,
        temporal_ksize=3,
        temporal_stride=1,
        temporal_pool_type="avg",
        cg_ksize=3,
        cg_stride=2,
    ),
    TBlock(
        "Block2", in_channels=64, out_channels=128, temporal_ksize=3, temporal_stride=1
    ),
    TBlock("Block3", in_channels=128, spatial_ksize=3, temporal_ksize=3),
    TBlock("Block3", in_channels=128, spatial_ksize=3, temporal_ksize=3),
    TBlock("Block3", in_channels=128, spatial_ksize=3, temporal_ksize=3),
    TBlock(
        "Block4",
        in_channels=128,
        out_channels=256,
        temporal_ksize=3,
        temporal_stride=1,
        cg_ksize=3,
    ),
    TBlock(
        "Block4",
        in_channels=256,
        out_channels=512,
        temporal_ksize=3,
        temporal_stride=1,
        cg_ksize=3,
    ),
]


class TVNBaseClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.tvn = TVN(tvn_config)
        # self.bert = BertModel.from_pretrained(config["bert_model_name"])

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features=512, out_features=512),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=512, out_features=config["num_classes"]),
            nn.Sigmoid(),
        )

        self.classifier.apply(weights_init)

    def forward(self, inputs):
        frames = inputs["frames"]  # (bs, max_length, C, H, W)
        frames_mask = inputs["frames_mask"]  # (bs, max_length)

        batch_size = frames.shape[0]
        valid_frames_cnt = torch.sum((frames_mask == 0).type(torch.int), dim=1)

        assert batch_size == len(valid_frames_cnt)

        frames_list = list()
        input_frames_cnt = list()
        input_segments_cnt = list()

        # ------------------
        n_frames_one_group = self.config["n_frames_one_group"]
        for item in valid_frames_cnt:
            temp_segment_cnt = 0
            for _ in range(int(item / n_frames_one_group)):
                input_frames_cnt.append(n_frames_one_group)
                temp_segment_cnt += 1

            if item % n_frames_one_group <= (n_frames_one_group * 3 / 4):
                input_frames_cnt[-1] += item % n_frames_one_group
            else:
                input_frames_cnt.append(item % n_frames_one_group)
                temp_segment_cnt += 1

            input_segments_cnt.append(temp_segment_cnt)
        # -------------------

        for idx, cnt in enumerate(valid_frames_cnt):
            frames_list.append(frames[idx, :cnt, :, :, :].squeeze())

        # frames -> (total_length, C, H, W)
        frames = torch.cat(frames_list, dim=0)

        # feature -> (total_segment_cnt, feature_size(512))
        feature = self.tvn((frames, input_frames_cnt))

        assert feature.shape[0] == sum(input_segments_cnt)

        idx = 0
        batch_feature_list = list()
        for cnt in input_segments_cnt:
            temp_feature = feature[idx : idx + cnt]  # (segment_length, feature_dim)

            temp_feature = torch.mean(temp_feature, dim=0)  # (1, feature_dim)
            batch_feature_list.append(temp_feature)

        feature = torch.stack(batch_feature_list)
        assert feature.shape[0] == batch_size

        # --- finish extract video_feature, frame_feature -> (batch_size, max_length, 512)

        # feed into classifer
        result = self.classifier(feature)

        return {
            "predicted_label": result,
        }

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

        loss_value = self.loss(gt_label, predicted_label)
        self.log("train_loss", loss_value, prog_bar=True)

        train_metrics = self._metric_calculator(gt_label, predicted_label)
        for item in train_metrics:
            self.log(f"train_{item}", train_metrics[item], prog_bar=True)

        if math.isnan(loss_value):
            return None

        return loss_value

    def validation_step(self, batch, batch_idx):
        result = self.forward(batch)
        predicted_label = result["predicted_label"]
        gt_label = batch["label"]

        return {"predicted_label": predicted_label, "gt_label": gt_label}

    def setup(self, stage=None):
        # called on every GPU
        self.data_loader_dict = dict()

        if stage == "fit" or stage is None:
            self.data_loader_dict["train"] = RawVideoDataset(
                dataset_type="train", config=self.config
            )
            self.data_loader_dict["val"] = RawVideoDataset(
                dataset_type="val", config=self.config
            )

        if stage == "test" or stage is None:
            self.data_loader_dict["test"] = RawVideoDataset(
                dataset_type="test", config=self.config
            )

        if stage == "val":
            self.data_loader_dict["val"] = RawVideoDataset(
                dataset_type="val", config=self.config
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_loader_dict["train"],
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_loader_dict["val"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_loader_dict["test"],
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=4,
        )
