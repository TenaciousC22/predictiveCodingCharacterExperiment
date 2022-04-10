import os

import torch
import numpy as np
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from generators.librispeech import LibrispeechUnsupervisedDataset, LibrispeechUnsupervisedLoader,
    LRS2UnsupervisedLoader, LRS2UnsupervisedDataset
from models.audiovisual_model import FBAudioVisualCPCLightning

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 3}

val_params = {'batch_size': 8,
          'shuffle': False,
          'num_workers': 3}


base_dir = "/home/analysis/Documents/studentHDD/datasets/LRS2"

lrs2_path = os.path.join(base_dir, "mvlrs_v1")

train_txt_path = os.path.join(base_dir, "txts/pretrain.txt")
train_ids = LRS2UnsupervisedLoader(file_path=train_txt_path).load()+LRS2UnsupervisedLoader(file_path=train_txt_path.replace("pretrain", "train")).load()
training_set = LRS2UnsupervisedDataset(train_ids, lrs2_path, params['batch_size'])
training_generator = data.DataLoader(training_set, **params)

test_txt_path = os.path.join(base_dir, "txts/val.txt")
test_ids = LRS2UnsupervisedLoader(file_path=test_txt_path).load()
test_set = LRS2UnsupervisedDataset(test_ids, lrs2_path, val_params['batch_size'])
test_generator = data.DataLoader(test_set, **val_params)

network = FBAudioVisualCPCLightning(batch_size=params['batch_size'])

trainer = pl.Trainer(gpus=1, callbacks=[EarlyStopping(monitor='val_loss', patience=5)], default_root_dir="lrs2_audiovisual_predictor_lightning_logs")
trainer.fit(network, training_generator, test_generator)
