import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datamodule.raw_video import RawVideoDataset
from model import AttentionPooling, ContextGating, StackedCoAttention
from model.tvn import TVN, TBlock
from model.utils import weights_init
from torch.utils.data import DataLoader
from module.metrics import calculate_gap
from sklearn.metrics import average_precision_score, recall_score

from module.base import VideoClassifier

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


class TVNCoAttnPoolingClassifier(VideoClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.tvn = TVN(tvn_config, num_classes=config["num_classes"], prepare_seq=False)
        # self.bert = BertModel.from_pretrained(config["bert_model_name"])

        modal_list = ["video", "text", "audio"]
        self.modal_list = modal_list

        co_attention_dict = dict()
        for modal in modal_list:
            for other_modal in modal_list:
                if modal == other_modal:
                    continue

                co_attention_dict[f"{modal}_{other_modal}"] = StackedCoAttention(
                    embed_dim=self.config[f"{modal}_dim"],
                    num_heads=self.config["attention_head"],
                    dropout=0.3,
                    kdim=self.config[f"{other_modal}_dim"],
                    layer=self.config["co_attention_layer"],
                )

        self.co_attention_dict = nn.ModuleDict(co_attention_dict)

        self_attention_dict = dict()
        pooling_dict = dict()
        for modal in modal_list:
            self_attention_dict[modal] = StackedCoAttention(
                embed_dim=self.config[f"{modal}_dim"] * 2,
                num_heads=self.config["attention_head"],
                dropout=0.3,
                kdim=self.config[f"{modal}_dim"] * 2,
                layer=self.config["self_attention_layer"],
            )

            pooling_dict[modal] = AttentionPooling(
                embed_dim=self.config["pooling_dim"],
                num_heads=8,
                dropout=0.3,
                kdim=self.config[f"{modal}_dim"] * 2,
                num_classes=self.config["num_classes"],
            )

        self.self_attention_dict = nn.ModuleDict(self_attention_dict)
        self.pooling_dict = nn.ModuleDict(pooling_dict)

        classifiers = list()
        for _ in range(self.config["num_classes"]):
            classifiers.append(
                nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(
                        self.config["pooling_dim"] * 3, self.config["pooling_dim"]
                    ),
                    nn.GroupNorm(1, self.config["pooling_dim"]),
                    nn.ReLU(),
                    ContextGating(self.config["pooling_dim"]),
                    nn.Linear(self.config["pooling_dim"], 1),
                    nn.Sigmoid(),
                )
            )

        self.classifiers = nn.ModuleList(classifiers)
        for classifier in self.classifiers:
            classifier.apply(weights_init)

    def forward(self, inputs):
        feature_dict = dict()
        mask_dict = dict()

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
        feature = self.tvn((frames, input_frames_cnt), extract_feas=True)

        assert feature.shape[0] == sum(input_segments_cnt)

        # convert back into tensors with padding and mask
        idx = 0
        final_feature_list = list()
        final_mask_list = list()
        for cnt in input_segments_cnt:
            temp_feature = feature[idx : idx + cnt]
            temp_mask = torch.tensor([0.0] * cnt)

            need_to_pad_cnt = (
                int(self.config["frame_padding_size"] / n_frames_one_group) - cnt
            )

            temp_feature = F.pad(temp_feature, (0, 0, 0, need_to_pad_cnt))
            temp_mask = F.pad(temp_mask, (0, need_to_pad_cnt), value=1)

            final_feature_list.append(temp_feature)
            final_mask_list.append(temp_mask)
            idx += cnt

        feature_dict["video"] = (
            torch.transpose(torch.stack(final_feature_list), 0, 1)
            .type(torch.float)
            .cuda()
        )
        mask_dict["video"] = torch.stack(final_mask_list).cuda()

        del final_feature_list
        del final_mask_list

        # --- finish extract video_feature, frame_feature -> (batch_size, max_length, 512)

        # text_feature_dict = dict()
        # text_mask_dict = dict()

        # for token_type in ["ocr", "asr"]:
        #     temp_token = inputs[token_type]
        #     temp_token_mask = inputs[f"{token_type}_mask"]
        #     temp_token_single_mask = inputs[f"{token_type}_single_mask"]

        #     token_list = list()
        #     mask_list = list()
        #     cnt_list = list()
        #     for idx in range(batch_size):
        #         temp_mask = temp_token_mask[idx]
        #         valid_cnt = sum(temp_mask)
        #         cnt_list.append(valid_cnt)
        #         token_list.append(temp_token[idx, :valid_cnt, :])
        #         mask_list.append(temp_token_single_mask[idx, :valid_cnt, :])

        #     input_token = torch.cat(token_list, dim=0)
        #     input_mask = torch.cat(mask_list, dim=0)

        #     bert_result = self.bert(input_ids=input_token, attention_mask=input_mask)[
        #         "pooler_output"
        #     ]

        #     final_data_list = list()
        #     idx = 0
        #     for cnt in cnt_list:
        #         temp_data = bert_result[idx : idx + cnt]  # (length, 768)
        #         temp_data = F.pad(
        #             temp_data,
        #             (0, 0, 0, self.config["text_padding_size"] - temp_data.shape[0]),
        #         )
        #         final_data_list.append(temp_data)

        #     text_feature_dict[token_type] = torch.stack(
        #         final_data_list
        #     )  # (bs, max_length, 768)
        #     text_mask_dict[token_type] = temp_token_mask
        feature_dict["text"] = torch.transpose(inputs["text_feature"], 0, 1)
        mask_dict["text"] = inputs["text_mask"]

        # --- finish extract text_feature, text_feature -> (batch_size, max_length, 768)

        # read audio_feature -> (batch_size, max_length, 128)
        feature_dict["audio"] = torch.transpose(inputs["audio"], 0, 1).type(torch.float)
        mask_dict["audio"] = inputs["audio_mask"]

        # calculate co attention cross modal
        for modal in self.modal_list:
            for other_modal in self.modal_list:
                if modal == other_modal:
                    continue
                #                 print(f"{modal}: {feature_dict[modal].type()}")
                #                 print(f"{other_modal}: {feature_dict[other_modal].type()}")
                feature_name = f"{modal}_{other_modal}"
                feature_dict[feature_name] = self.co_attention_dict[feature_name](
                    feature_dict[modal],
                    feature_dict[other_modal],
                    mask_dict[modal],
                    mask_dict[other_modal],
                )

        for modal in self.modal_list:
            # concat co attention of the same modal
            temp_feature_list = list()
            need_to_pop_list = list()
            for feature_name in feature_dict:
                if f"{modal}_" in feature_name:
                    need_to_pop_list.append(feature_name)
                    temp_feature_list.append(feature_dict[feature_name])

            for feature_name in need_to_pop_list:
                feature_dict.pop(feature_name)

            feature_dict[modal] = torch.cat(temp_feature_list, dim=-1)

            # now calculate self attention of each modal
            feature_dict[modal] = self.self_attention_dict[modal](
                feature_dict[modal],
                feature_dict[modal],
                mask_dict[modal],
                mask_dict[modal],
            )

            # now pooling sequences of features with attn pooling
            feature_dict[modal] = self.pooling_dict[modal](
                feature_dict[modal], mask_dict[modal].type(torch.bool),
            )

        # now concat modal features and feed into classifiers
        # TODO: make sure no more keys are in the feature_dict, or this will be wrong
        feature = torch.cat(list(feature_dict.values()), dim=-1)

        result = list()
        for idx, classifier in enumerate(self.classifiers):
            temp_feature = feature[idx, :, :]
            temp_result = classifier(temp_feature)
            result.append(temp_result)

        result = torch.cat(result, dim=-1)

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
