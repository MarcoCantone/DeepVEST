from torch.utils.data import Dataset
import pandas as pd
from utilities import load3dfrom2dslice, path_seq_map, dcm_pos_orientation
import os
import nrrd
import numpy as np
import torch


class DukeMRI_Det_SS(Dataset):
    """
    Dataset class for duke, single-sequence with bbs targets
    you need to give images and targets in the same order

    """

    def __init__(self, image_root, images_list, targets_filepath, data_transform=None, target_transform=None):

        self.image_root = image_root

        with open(images_list, 'r') as f:
            self.image_dirs = [x[:-1] for x in f.readlines()]

        self.targets = pd.read_csv(targets_filepath)
        self.targets.index = self.targets["Patient ID"]
        self.targets.drop(self.targets.columns[0], axis=1, inplace=True)

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        img = load3dfrom2dslice(os.path.join(self.image_root,self.image_dirs[index]))
        target = self.targets.iloc[index].tolist()
        return img, target


class DukeMRI_Seg_MS_old(Dataset):
    """
    Dataset class for duke, multi-sequence with segmentation map

    """

    def __init__(self, images_root, seg_map_folder, path_seq_map_file, sequences=["post_1"], data_transform=None, target_transform=None, stack_sequences=False):

        self.images_root = images_root
        self.seg_map_folder = seg_map_folder
        self.sequences = sequences
        self.stack_sequences = stack_sequences

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.ids = sorted(os.listdir(seg_map_folder))

        self.path_seq = path_seq_map(path_seq_map_file)

        self.slice_order = dcm_pos_orientation(images_root)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        label = nrrd.read(os.path.join(self.seg_map_folder, self.ids[index], "Segmentation_" + self.ids[index] + "_Dense_and_Vessels.seg.nrrd"))
        if self.target_transform is not None:
            label = self.target_transform(label)

        series = {}
        for sequence_name in self.sequences:
            seq_dir = self.path_seq[(self.path_seq["classic_path"].str.contains(self.ids[index])) & (self.path_seq["sequence"] == sequence_name)]["classic_path"].item()
            series[sequence_name] = load3dfrom2dslice(os.path.join(self.images_root, seq_dir), flip_slice_dim=self.slice_order[seq_dir])

        if self.data_transform is not None:
            series = {x: self.data_transform(series[x]) for x in series}

        if self.stack_sequences:
            series = np.stack([series[x] for x in series])
            if len(self.sequences) == 1:
                series = series[0]

        return series, label


class DukeMRI_Seg_MS(Dataset):
    """
    Dataset class for duke, multi-sequence with segmentation map

    """

    def __new__(cls, root=None, metadata_filepath=None, data_transform=None, target_transform=None, stack_sequences=False):

        obj = super().__new__(cls)

        return obj

    def __init__(self, root, metadata_filepath, data_transform=None, target_transform=None, stack_sequences=False):

        self.root = root
        self.stack_sequences = stack_sequences

        self.data_transform = data_transform
        self.target_transform = target_transform

        self.data = torch.load(metadata_filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = nrrd.read(os.path.join(self.root, self.data.iloc[index]["label_path"]))
        if self.target_transform is not None:
            label = self.target_transform(label)

        series = {}
        for sequence_name in self.data.iloc[index]["sequence_paths"]:
            series[sequence_name] = load3dfrom2dslice(os.path.join(self.root, self.data.iloc[index]["sequence_paths"][sequence_name]), flip_slice_dim=self.data.iloc[index]["slice_ordering"])

        if self.data_transform is not None:
            series = {x: self.data_transform(series[x]) for x in series}

        if self.stack_sequences:
            series = torch.stack([series[x] for x in series])

            if len(series) == 1:
                series = series[0]

        return series, label

    def random_split(self, test_ratio=0.2, seed=42):

        test_data = self.data.sample(frac=test_ratio, random_state=seed)
        train_data = self.data.drop(test_data.index)

        train_dataset = DukeMRI_Seg_MS.__new__(DukeMRI_Seg_MS)
        test_dataset = DukeMRI_Seg_MS.__new__(DukeMRI_Seg_MS)

        train_dataset.root = self.root
        test_dataset.root = self.root

        train_dataset.data = train_data
        test_dataset.data = test_data

        train_dataset.stack_sequences = self.stack_sequences
        test_dataset.stack_sequences = self.stack_sequences

        train_dataset.data_transform = self.data_transform
        test_dataset.data_transform = self.data_transform

        train_dataset.target_transform = self.target_transform
        test_dataset.target_transform = self.target_transform

        return train_dataset, test_dataset
