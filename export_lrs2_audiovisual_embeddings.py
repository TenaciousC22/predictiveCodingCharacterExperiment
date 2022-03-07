import os

import librosa
import torch
import numpy as np
from tqdm import tqdm

from generators.librispeech import LibrispeechUnsupervisedDataset, LibrispeechUnsupervisedLoader
from models.audiovisual_model import FBAudioVisualCPCLightning

# def download_state_dict(model_name):
#
#     base_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints"
#     return torch.hub.load_state_dict_from_url(f"{base_url}/{model_name}", map_location='cuda:0')
#
# state_dict = download_state_dict("60k_epoch4-d0f474de.pt")
#
# config = state_dict["config"]
# weights = state_dict["weights"]

model = FBAudioVisualCPCLightning(batch_size=1).cuda()

checkpoint = torch.load('/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/base_model/epoch=50-step=906218.ckpt')

#print(model.cpc_model.gEncoder.conv0.weight)

#model.load_from_checkpoint('fb_lightning_logs/lightning_logs/version_1/checkpoints/epoch=45-step=1617083.ckpt')
model.load_state_dict(checkpoint['state_dict'])
#print(model.cpc_model.gEncoder.conv0.weight)


libri_path = "/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1"
dest_dir = "/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1_embeddings"

train_output = True

jobs = []

for folder in os.listdir(libri_path):

    for file in os.listdir(os.path.join(libri_path, folder)):

            if not file.endswith('.wav'):
                continue

            jobs.append((folder, file))

print(jobs)

for folder, file in tqdm(jobs):
    if not train_output:
        dest = os.path.join(dest_dir, folder+'-'+file.replace('.wav', '.npy'))
    else:
        dest = os.path.join(dest_dir, folder, file.replace('.wav', '.npy'))

    # if os.path.exists(dest):
    #     continue

    if not os.path.exists(os.path.join(dest_dir, folder)):
        os.makedirs(os.path.join(dest_dir, folder))

    x_audio, _ = librosa.load(os.path.join(libri_path, folder, file), sr=16000)
    x_visual = np.load(os.path.join(libri_path, folder, file.replace('.wav', '.npy')))

    x_audio_tensor = torch.from_numpy(x_audio).unsqueeze(0).unsqueeze(0).cuda()
    x_visual_tensor = torch.from_numpy(x_visual.T).unsqueeze(0).cuda()

    embedding = model.embedding((x_audio_tensor, x_visual_tensor), context=True, audioVisual=True, norm=False)

    embedding = torch.squeeze(embedding).detach().cpu().numpy()
    print(embedding)

    #print(embedding.shape)
    np.save(dest, embedding)




