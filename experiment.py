import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from utilities import model_rgb2gray, has_parameter
from train_functions import training
from config import load_config, write_config, create_object_from_dict, get_class_by_path

# COMMAND LINE ARGUMENT PARSER
# TODO: add the possibility to override params in cfg file
parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, default=None)
args = parser.parse_args()

config = load_config(args.config)

# CREATE TRANSFORMS
data_transform = None
if "DATA_TRANSFORM" in config.keys():
    transform_list = []
    for i in range(len(config["DATA_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["DATA_TRANSFORM"][i]))
    data_transform = transforms.Compose(transform_list)

target_transform = None
if "TARGET_TRANSFORM" in config.keys():
    transform_list = []
    for i in range(len(config["TARGET_TRANSFORM"])):
        transform_list.append(create_object_from_dict(config["TARGET_TRANSFORM"][i]))
    target_transform = transforms.Compose(transform_list)

# CREATE DATASET
data = get_class_by_path(config["DATASET"]["class"])(**config["DATASET"]["params"], data_transform=data_transform, target_transform=target_transform)

# SPLIT DATASET
# TODO: data have to define it own way to split. Add at least one assert
train_dataset, val_dataset = data.random_split(test_ratio=0.2, seed=42)

# DATALOADERS
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=config["TRAINING"]["train_batch_size"],
                              num_workers=config["TRAINING"]["num_workers"],
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=config["TRAINING"]["test_batch_size"],
                            num_workers=config["TRAINING"]["num_workers"],
                            shuffle=False)

# MODEL
# TODO: try to put backbone config inside the model
if "BACKBONE" in config.keys():
    backbone = create_object_from_dict(config["BACKBONE"]["model"])
    if config["BACKBONE"]["gray"]:
        model_rgb2gray(backbone)
    if config["BACKBONE"]["layerToDiscard"] > 0:
        backbone = nn.Sequential(*list(backbone.children())[:-config["BACKBONE"]["layerToDiscard"]])

model_class = get_class_by_path(config["MODEL"]["class"])
if has_parameter(model_class, "backbone"):
    model = model_class(**config["MODEL"]["params"], backbone=backbone)
else:
    model = model_class(**config["MODEL"]["params"])

if config["TRAINING"]["parallel"]:
    model = nn.DataParallel(model, device_ids=config["TRAINING"]["parallel"])
    device = config["TRAINING"]["parallel"][0]
else:
    device = torch.device(config["TRAINING"]["device"])

model = model.to(device)

# create loss, optimizer and scheduler
loss = create_object_from_dict(config["LOSS"])

optimizer = get_class_by_path(config["OPTIMIZER"]["class"])(**config["OPTIMIZER"]["params"], params=model.parameters())

if "SCHEDULER" in config.keys():
    scheduler = get_class_by_path(config["SCHEDULER"]["class"])(**config["SCHEDULER"]["params"], optimizer=optimizer)
else:
    scheduler = None

# write_config(config, config["TRAINING"]["workspace"]+"setup.yaml")
from train_functions import train_one_epoch



res = training(dataloader_train=train_dataloader,
               dataloader_valid=val_dataloader,
               model=model,
               criterion=loss,
               optimizer=optimizer,
               scheduler=scheduler,
               epochs=config["TRAINING"]["epochs"],
               device=device,
               resume=config["TRAINING"]["resume"],
               workspace=config["TRAINING"]["workspace"],
               print_per_epoch=20)
