import os

import numpy as np
import torch
from torch.utils.data import Dataset


class CLaPPDataset(Dataset):
    def __init__(
        self,
        clapp_dataset_dir: str,
        split: str = "train",
        split_ratio: float = 0.9,
        suffix: str = "500k",
        shuffle_seed: int = 0,
    ):
        # Load the dataset
        self.clapp_dataset_dir = clapp_dataset_dir
        self.split = split
        self.split_ratio = split_ratio
        assert suffix == "500k", "Only 500k is supported for now"
        self.pose_features = np.load(os.path.join(clapp_dataset_dir, f"pose_features_{suffix}.npy"))
        self.text_features = np.load(os.path.join(clapp_dataset_dir, f"text_features_{suffix}.npy"))
        self.keys = np.load(os.path.join(clapp_dataset_dir, f"keys_{suffix}.npy"))
        total_num_examples = len(self.keys)

        # Shuffle the dataset
        indices = np.arange(len(self.keys))
        np.random.seed(shuffle_seed)
        np.random.shuffle(indices)
        self.pose_features = self.pose_features[indices]
        self.text_features = self.text_features[indices]
        self.keys = self.keys[indices]

        # Split the dataset
        if split == "train":
            self.num_examples = int(split_ratio * total_num_examples)
            self.pose_features = self.pose_features[: self.num_examples]
            self.text_features = self.text_features[: self.num_examples]
            self.keys = self.keys[: self.num_examples]
        elif split == "val":
            self.num_examples = int((1 - split_ratio) * total_num_examples)
            self.pose_features = self.pose_features[-self.num_examples :]
            self.text_features = self.text_features[-self.num_examples :]
            self.keys = self.keys[-self.num_examples :]
        else:
            raise ValueError(f"split must be either 'train' or 'val', got {split}")

    def __getitem__(self, idx):
        item = {
            "key": self.keys[idx],
            "pose_features": self.pose_features[idx],
            "text_features": self.text_features[idx],
        }
        return item

    def __len__(self):
        return len(self.keys)


if __name__ == "__main__":
    dataset = CLaPPDataset("clapp/data", split="train", split_ratio=1.0)
    print(len(dataset))
