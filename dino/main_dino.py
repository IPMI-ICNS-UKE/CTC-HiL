import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report
from tqdm import tqdm
import wandb
from sparsam.train import create_dino_gym

from hil.utils.model_zoo import xcit_gc_nano_12_p8_224_dist
from dino.data_processing.dataset import DinoImageSet
from dino.data_processing.data_augmentation import DataCropperDINO

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing randomness") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so results are reproducible") % 2 ** 32 - 1)

    # Path to the configuration file
    cfg_path = Path("/home/hhusseini/PycharmProjects/CTC/configs/dino_cfg.yml")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Load paths for unlabeled training data
    image_root_path_unlabeled_training = Path(config["general_paths"]["image_root_path_unlabeled_training"])
    df_unlabeled_train_paths = pd.read_csv(image_root_path_unlabeled_training)

    # Define root path and create unlabeled training paths
    root_path = Path(config["general_paths"]["root_path"])
    train_unlabeled_paths = [
        [root_path / Path(row["dapi_path"]),
         root_path / Path(row["ck_path"]),
         root_path / Path(row["cd45_path"])]
        for _, row in tqdm(df_unlabeled_train_paths.iterrows(), total=len(df_unlabeled_train_paths))
    ]
    print("Number of unlabeled samples:", len(train_unlabeled_paths))

    # Create dataset
    unlabeled_train_set = DinoImageSet(img_paths=train_unlabeled_paths)

    # Define data augmentation
    data_augmentation = DataCropperDINO(
        n_global_crops=2,
        n_local_crops=5,
        global_crops_scale=(0.5, 1),
        local_crops_scale=(0.1, 0.5),
        res=256,
    )

    # Set the device and backbone
    device = config["device"]
    backbone = xcit_gc_nano_12_p8_224_dist(**config["model_parameter"])
    backbone.to(torch.float32)
    backbone.to(device)

    # Define parameters for DataLoader and Optimizer
    data_loader_parameter = dict(
        batch_size=128,
        num_workers=3,
        shuffle=True,
        drop_last=True,
        persistent_workers=True
    )

    optimizer_parameter = dict(
        lr=0.001,
        weight_decay=0.04
    )

    save_path = config["DinoGym"]["save_path"]

    # Initialize Wandb if save path is specified
    if config["DinoGym"]["save_path"]:
        wandb.init(**config["wandb_parameter"], dir=save_path, config=config)
        logger = wandb
    else:
        logger = None

    metrics = [
        partial(classification_report, output_dict=True, zero_division=0),
    ]
    metrics_requires_probability = [False]

    # Create the Dino Gym for training
    gym = create_dino_gym(
        unalabeled_train_set=unlabeled_train_set,
        backbone_model=backbone,
        save_path=save_path,
        device=device,
        logger=logger,
        unlabeled_train_loader_parameters=data_loader_parameter,
        resume_training_from_checkpoint=False,
        projection_head_n_layers=3,
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_iterations=0,
        grad_clip_factor=3.0,
        teacher_momentum=0.995,
        final_lr=1.0e-6,
        optimizer_parameters=optimizer_parameter,
        data_augmentation=data_augmentation,
    )

    # Train the model
    student, teacher = gym.train()
