import os

import torch
import numpy as np
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from generators.librispeech import LRS2AudioVisualPhonemeDataset, LRS2UnsupervisedLoader, LRS2AudioVisualCachedCharacterDataset, LRS2UnsupervisedDataset, hybridNetworkDataset
from models.audiovisual_model import  FBAudioVisualCPCCharacterClassifierLightning, CPCCharacterClassifierLightningV4, embeddingDecoder
from util.pad import audiovisual_batch_collate, audiovisual_embedding_batch_collate

params={'batch_size': 8,
		'shuffle': True,
		'num_workers': 3}

val_params={'batch_size': 8,
		'shuffle': False,
		'num_workers': 3}

lrs2_path="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1"
src_ckpt='/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/base_model/epoch=50-step=906218.ckpt'
#src_ckpt="/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/one_layer_lstm/lightning_logs/version_1/checkpoints/epoch=10-step=63018.ckpt"

train_txt_path="/home/analysis/Documents/studentHDD/datasets/LRS2/txts/train.txt"

train_ids = LRS2UnsupervisedLoader(file_path=train_txt_path).load()
training_set = LRS2AudioVisualCachedCharacterDataset(train_ids, lrs2_path, params['batch_size'])
training_generator = data.DataLoader(training_set, collate_fn=audiovisual_embedding_batch_collate, **params)

test_txt_path="/home/analysis/Documents/studentHDD/datasets/LRS2/txts/val.txt"

test_ids = LRS2UnsupervisedLoader(file_path=test_txt_path).load()
test_set = LRS2AudioVisualCachedCharacterDataset(test_ids, lrs2_path, val_params['batch_size'])
test_generator = data.DataLoader(test_set, collate_fn=audiovisual_embedding_batch_collate, **val_params)

network = CPCCharacterClassifierLightningV4(src_checkpoint_path=src_ckpt, batch_size=params['batch_size'], LSTM=True, LSTMLayers=4)

trainer = pl.Trainer(gpus=1, callbacks=[EarlyStopping(monitor='train_loss', patience=10)], default_root_dir="4_layer_lstm")
trainer.fit(network, training_generator, test_generator)