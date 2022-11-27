import os

import librosa
import torch
import numpy as np
from tqdm import tqdm

from generators.librispeech import LibrispeechUnsupervisedDataset, LibrispeechUnsupervisedLoader
from models.audiovisual_model import FBAudioVisualCPCLightning
from models.audio_model import  FBAudioCPCLightning

# def download_state_dict(model_name):
#
#     base_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints"
#     return torch.hub.load_state_dict_from_url(f"{base_url}/{model_name}", map_location='cuda:0')
#
# state_dict = download_state_dict("60k_epoch4-d0f474de.pt")
#
# config = state_dict["config"]
# weights = state_dict["weights"]

model = FBAudioCPCLightning(batch_size=1).cuda()

checkpoint_url = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url,
														progress=False)

# checkpoint = torch.load('/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/base_model/60k_epoch4-d0f474de.pt')

#print(model.cpc_model.gEncoder.conv0.weight)

#model.load_from_checkpoint('fb_lightning_logs/lightning_logs/version_1/checkpoints/epoch=45-step=1617083.ckpt')
# model.load_state_dict(checkpoint['state_dict'])
#print(model.cpc_model.gEncoder.conv0.weight)

libri_path = "/home/analysis/Documents/studentHDD/chris/IDS-corpus-edited-ds"
dest_dir = "/home/analysis/Documents/studentHDD/chris/IDS-corpus-edited-ds-embed"

train_output = False

jobs = []

for file in os.listdir(libri_path):

	if not file.endswith('.wav'):
		continue

	jobs.append((libri_path, file))