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

# checkpoint_url = 'https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints/60k_epoch4-d0f474de.pt'
# checkpoint = torch.hub.load_state_dict_from_url(checkpoint_url,progress=False)

checkpoint = torch.load('/home/analysis/Dev/lukas/speech_cpc/lrs2_audio_only_lightning_logs/lightning_logs/version_0/checkpoints/epoch=60-step=1083908.ckpt')

#print(model.cpc_model.gEncoder.conv0.weight)

#model.load_from_checkpoint('fb_lightning_logs/lightning_logs/version_1/checkpoints/epoch=45-step=1617083.ckpt')
model.load_state_dict(checkpoint['state_dict'])
#print(model.cpc_model.gEncoder.conv0.weight)

libri_path = "/home/analysis/Documents/studentHDD/chris/IDS-corpus-edited-ds"
dest_dir = "/home/analysis/Documents/studentHDD/chris/IDS-corpus-edited-ds-embed/"

train_output = False

jobs = []

for file in os.listdir(libri_path):

	if not file.endswith('.wav'):
		continue

	jobs.append((dest_dir, file))

for folder, file in tqdm(jobs):
	if not train_output:
		dest = os.path.join(dest_dir, folder+'-'+file.replace('.wav', '.npy'))
	else:
		dest = os.path.join(dest_dir, folder, file.replace('.wav', '.npy'))

	# if os.path.exists(dest):
	#     continue

	if not os.path.exists(os.path.join(dest_dir, folder)):
		os.makedirs(os.path.join(dest_dir, folder))

	x, _ = librosa.load(os.path.join(libri_path, folder, file), sr=16000)

	x_tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).cuda()

	embedding = model.embedding(x_tensor, context=False, norm=True)

	embedding = torch.squeeze(embedding).detach().cpu().numpy()

	#print(embedding.shape)
	np.save(dest, embedding)