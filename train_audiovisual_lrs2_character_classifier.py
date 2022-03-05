import os

import torch
import numpy as np
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from generators.librispeech import LRS2AudioVisualPhonemeDataset, LRS2UnsupervisedLoader, LRS2AudioVisualCachedPhonemeDataset
from models.audiovisual_model import  FBAudioVisualCPCCharacterClassifierLightning
from util.pad import audiovisual_batch_collate, audiovisual_embedding_batch_collate

params={'batch_size': 8,
		'shuffle': True,
		'num_workers': 3}

val_params={'batch_size': 8,
		'shuffle': False,
		'num_workers': 3}

lrs2_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1"
src_ckpt="/home/analysis/Documents/studentHDD/chris/lightning_logs/version_0/checkpoints/epoch=10-step=63018.ckpt"

train_txt_path="/home/analysis/Documents/studentHDD/datasets/LRS2/txts/train.txt"
test_txt_path="/home/analysis/Documents/studentHDD/datasets/LRS2/txts/val.txt"

train_ids = LRS2UnsupervisedLoader(file_path=train_txt_path).load()
print(train_ids)

test_ids = LRS2UnsupervisedLoader(file_path=test_txt_path).load()
print(test_ids)