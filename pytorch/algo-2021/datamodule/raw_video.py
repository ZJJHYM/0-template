import json
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset
import json
from torchvision.io import read_image
from torchvision import transforms
import torch.nn.functional as F

from transformers import BertTokenizer


class RawVideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str = "./dataset",
        dataset_type: str = None,
        label_id_path: str = "./dataset/label_dict.json",
        frame_size=(224, 224),
        config=None,
    ):
        super().__init__()

        self.config = config
        self.frame_size = frame_size

        assert dataset_type in ["train", "val", "test"]

        self.is_test: bool = dataset_type == "test"

        # define image and text preprocessing
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(frame_size),
                transforms.Lambda(lambda x: x / 127.5 - 1),
                # ?? do we need to normalize
                # transforms.Normalize(mean=[], std=[]),
            ]
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        # read label_id
        self.label_dict: dict = json.load(open(label_id_path, "r", encoding="utf-8"))

        # read ground truth label for train dataset
        if dataset_type != "test":
            self.data_dir = os.path.join(dataset_dir, "train_5k")
            # TODO: or we do not need to split train and val in advance
            self.video_list = json.load(
                open(
                    os.path.join(self.data_dir, f"video_{dataset_type}.json"),
                    "r",
                    encoding="utf-8",
                )
            )

            # TODO: here is the code to load the whole train dataset, and split randomly later in dataloader
            # self.video_list = list()
            # self.video_name_list = json.load(open(os.path.join(self.data_dir, "raw_name.json"), "r"))
            # for item in self.video_name_list:
            #     self.video_list.append(item.split('.')[0])

            self.video_label_dict = json.load(
                open(
                    os.path.join(self.data_dir, "video_name_label.json"),
                    "r",
                    encoding="utf-8",
                )
            )

        # for test dataset, read the video_name
        else:
            self.data_dir = os.path.join(dataset_dir, "test_5k_2nd")
            self.video_list = list()
            self.video_name_list = json.load(
                open(os.path.join(self.data_dir, "raw_name.json"), "r")
            )
            for item in self.video_name_list:
                self.video_list.append(item.split(".")[0])

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        data_dict = dict()

        video_name = self.video_list[index]
        data_dict["video_name"] = video_name

        # raad video frames
        frame_dir = os.path.join(self.data_dir, "frames", video_name)
        frame_list = list()
        for item in os.listdir(frame_dir):
            temp_img = read_image(os.path.join(frame_dir, item))  # (C, H, W)
            temp_img = self.image_transforms(temp_img)
            frame_list.append(temp_img)

        data_dict["frames"] = torch.stack(frame_list)  # (n_frames, C, H, W)
        frames_cnt = data_dict["frames"].shape[0]
        data_dict["frames_mask"] = torch.zeros((frames_cnt))

        # !! pading operations might be a little confusing, the goal of padding is as follows
        # the frames count of every videos vary
        # the shape of frames data for one video is (frames_cnt, 3, 224, 224)
        # to stack the frames into (batch_size, max_length, 3, 224, 244)
        # we need to pad every video's frames tensor:
        # (frames_cnt, 3, 224, 224) -> (max_length, 3, 224, 224)
        # and `frames_mask` is to indicate which ones are padded

        # reference to pad function: https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html?highlight=pad#torch.nn.functional.pad
        need_to_pad_cnt = self.config["frame_padding_size"] - frames_cnt
        data_dict["frames"] = F.pad(
            data_dict["frames"], (0, 0, 0, 0, 0, 0, 0, need_to_pad_cnt)
        )
        data_dict["frames_mask"] = F.pad(
            data_dict["frames_mask"], (0, need_to_pad_cnt), value=1
        )

        # read video audios
        # ?? use extracted audio feature
        audio_file = os.path.join(self.data_dir, "audio_feature", f"{video_name}.npy")
        try:
            temp_audio_feature = torch.tensor(np.load(audio_file))
            audio_mask = torch.tensor([0.0] * temp_audio_feature.shape[0])
        except Exception as e:
            temp_audio_feature = torch.zeros((0, 128))
            audio_mask = torch.zeros((0))

        need_to_pad_cnt = (
            self.config["audio_padding_size"] - temp_audio_feature.shape[0]
        )
        data_dict["audio"] = F.pad(temp_audio_feature, (0, 0, 0, need_to_pad_cnt))
        data_dict["audio_mask"] = F.pad(audio_mask, (0, need_to_pad_cnt), value=1)

        """
        TODO: consider combine ocr and asr text
        - concat together
        - if the length is more than padding_size
        - dropout randomly, not clipping
        """
        # read text
        # text_file = os.path.join(self.data_dir, "text", f"{video_name}.txt")
        # text_json = json.load(open(text_file, "r"))
        # data_dict["ocr"] = text_json["video_ocr"].split("|")
        # data_dict["asr"] = text_json["video_asr"].split("|")

        temp_text_feature = np.load(
            os.path.join(self.data_dir, "text_feature", f"{video_name}.npy")
        )
        if temp_text_feature.shape[0] > self.config["text_padding_size"]:
            temp_text_feature = torch.tensor(temp_text_feature[:300, :])
            text_mask = [0.0] * self.config["text_padding_size"]
        else:
            temp_text_feature = torch.tensor(
                np.pad(
                    temp_text_feature,
                    [
                        (
                            0,
                            self.config["text_padding_size"]
                            - temp_text_feature.shape[0],
                        ),
                        (0, 0),
                    ],
                    constant_values=1e-5,
                )
            )
            text_mask = [0.0] * temp_text_feature.shape[0] + [1.0] * (
                self.config["text_padding_size"] - temp_text_feature.shape[0]
            )

        data_dict["text_feature"] = temp_text_feature
        data_dict["text_mask"] = torch.tensor(text_mask)

        """
        for item in ["ocr", "asr"]:
            data = data_dict[item]
            token_list = self.tokenizer(
                data,
                return_tensors="pt",
                padding=True,
                max_length=self.config["sentence_padding_size"],
            )
            # data_dict["orc" / "asr"] stores the tokenizations
            # data-dict["xxx_single_mask"] stores the mask of every tokenization (one sentence one mask)
            # data_dict["xxx_mask"] stores the text mask of every video (one video one mask)

            # data_dict["xxx"] -> (text_padding_size, sentence_padding_size)
            # data_dict["xxx_single_mask"] -> (text_padding_size, sentence_padding_size)
            # data_dict["xxx_mask"] -> (text_padding_size)

            # !! notice that `xxx_single_mask` use 0 to indicate mask -> as hugging face transformers does
            # !! but `xxx_mask` use 1 to indicate mask -> as pytorch attention layer does

            data_dict[item] = token_list["input_ids"]
            data_dict[f"{item}_single_mask"] = token_list["attention_mask"]
            data_dict[f"{item}_mask"] = torch.tensor(
                [0.0] * data_dict[item].shape[0]
                + [1.0] * (self.config["text_padding_size"] - data_dict[item].shape[0])
            )

            # padding all tokenizations and masks to max_length
            # !! padding operations will also be confusing just like the frames
            # for every sentence, we need to make sure the token count is `sentence_padding_size`
            # for every video, we need to make sure the sentence count is `padding_size`
            data_dict[item] = F.pad(
                data_dict[item],
                (
                    0,
                    self.config["sentence_padding_size"] - data_dict[item].shape[-1],
                    0,
                    self.config["text_padding_size"] - data_dict[item].shape[0],
                ),
            )

            single_mask_shape = data_dict[f"{item}_single_mask"].shape
            data_dict[f"{item}_single_mask"] = F.pad(
                data_dict[f"{item}_single_mask"],
                (
                    0,
                    self.config["sentence_padding_size"] - single_mask_shape[-1],
                    0,
                    self.config["text_padding_size"] - single_mask_shape[0],
                ),
            )
        """

        if not self.is_test:
            gt_label_one_hot = [0] * len(self.label_dict["label_idx_to_name"])
            for j in self.video_label_dict[video_name]:
                gt_label_one_hot[j] = 1

            data_dict["label"] = torch.tensor(gt_label_one_hot)

        else:
            data_dict["label"] = torch.zeros(
                (len(self.label_dict["label_idx_to_name"]))
            )

        return data_dict
