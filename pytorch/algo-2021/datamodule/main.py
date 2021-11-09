import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    def __init__(
        self, config, dataset_type="train",
    ):
        super().__init__()
        assert config is not None
        assert dataset_type in ["train", "test", "val"]
        self.config = config
        self.is_test = dataset_type == "test"

        dataset_dir = config.dataset.dataset_dir

        self.label_dict: dict = json.load(
            open(os.path.join(dataset_dir, "label_dict.json"), "r", encoding="utf-8")
        )

        if not self.is_test:
            self.data_dir = os.path.join(dataset_dir, "train_5k")
            self.video_list = json.load(
                open(
                    os.path.join(self.data_dir, f"video_{dataset_type}.json"),
                    "r",
                    encoding="utf-8",
                )
            )

            self.video_label_dict = json.load(
                open(
                    os.path.join(self.data_dir, "video_name_label.json"),
                    "r",
                    encoding="utf-8",
                )
            )

        else:
            self.data_dir = os.path.join(dataset_dir, "test_5k_2nd")
            self.video_list = list()
            self.video_name_list = json.load(
                open(os.path.join(self.data_dir, "raw_name.json"), "r")
            )
            for item in self.video_name_list:
                self.video_list.append(item.split(".")[0])

        # calculate label weights
        # label_cnt_path = "./label_id_cnt.json"
        # label_cnt_dict = json.load(open(label_cnt_path))

        # self.label_weight_list = [0.0] * len(self.label_dict["label_idx_to_name"])
        # """
        # NOTICE:
        # current implementation is really simple
        # the more a label occurs, the lower its weight is
        # and this relationship is linear
        # """
        # beta = (len(self.label_dict["label_idx_to_name"]) - 1) / len(
        #     self.label_dict["label_idx_to_name"]
        # )
        # for label_id in label_cnt_dict:
        #     self.label_weight_list[int(label_id)] = (1 - beta) / (
        #         1 - math.pow(beta, label_cnt_dict[label_id])
        #     )

    def __len__(self):
        return len(self.video_list)

    def _load_feature(self, feature_name, video_name):
        try:
            temp_feature = np.load(
                os.path.join(
                    self.data_dir,
                    self.config.dataset[f"{feature_name}_feature"],
                    "%s.npy" % video_name,
                )
            )
        except Exception as e:
            temp_feature = np.zeros((0, self.config[f"{feature_name}_dim"]))
            print(
                f"[WARNING] Failed to find {os.path.join(self.data_dir, self.config.dataset[f'{feature_name}_feature'], '%s.npy' % video_name)}"
            )

        if temp_feature.shape[0] > self.config[f"{feature_name}_padding_size"]:
            temp_feature = temp_feature[: self.config[f"{feature_name}_padding_size"]]
            temp_mask = [0.0] * temp_feature.shape[0]
        else:
            temp_mask = [0.0] * temp_feature.shape[0] + [1.0] * (
                self.config[f"{feature_name}_padding_size"] - temp_feature.shape[0]
            )
            temp_feature = np.pad(
                temp_feature,
                [
                    (
                        0,
                        self.config[f"{feature_name}_padding_size"]
                        - temp_feature.shape[0],
                    ),
                    (0, 0),
                ],
                constant_values=1e-5,
            )

        return temp_feature, temp_mask

    def __getitem__(self, index):
        feature_dict = dict()

        for modal in self.config.modal_list:
            temp_feature, temp_mask = self._load_feature(modal, self.video_list[index])
            feature_dict[f"{modal}_feature"] = torch.tensor(temp_feature)
            feature_dict[f"{modal}_mask"] = torch.tensor(temp_mask)

        feature_dict["video_name"] = self.video_list[index]

        if not self.is_test:
            gt_label_one_hot = [0] * len(self.label_dict["label_idx_to_name"])
            for j in self.video_label_dict[self.video_list[index]]:
                gt_label_one_hot[j] = 1

            feature_dict["label"] = gt_label_one_hot
        else:
            feature_dict["label"] = torch.zeros(
                len(self.label_dict["label_idx_to_name"])
            )
            feature_dict["video_full_name"] = self.video_name_list[index]

        return feature_dict
