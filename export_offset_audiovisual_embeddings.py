import os

import librosa
import torch
import numpy as np
from tqdm import tqdm

from generators.librispeech import LibrispeechUnsupervisedDataset, LibrispeechUnsupervisedLoader
from models.audiovisual_model import FBAudioVisualCPCLightning

index2char={0:" ", 21:"'", 29:"1", 28:"0", 36:"3", 31:"2", 33:"5", 37:"4", 35:"7", 34:"6", 30:"9", 32:"8",
	4:"A", 16:"C", 19:"B", 1:"E", 11:"D", 15:"G", 18:"F", 5:"I", 8:"H", 23:"K", 24:"J", 17:"M",
	10:"L", 3:"O", 6:"N", 26:"Q", 20:"P", 7:"S", 9:"R", 12:"U", 2:"T", 14:"W", 22:"V", 13:"Y",
	25:"X", 27:"Z", 38:""}    #index to character reverse mapping

offsetMap={
	0:"I840",
	1:"I720",
	2:"I600",
	3:"I480",
	4:"I360",
	5:"I240",
	6:"I060",
	7:"base",
	8:"B060",
	9:"B240",
	10:"B360",
	11:"B480",
	12:"B600",
	13:"B720",
	14:"B840",
	15:"jumble"
}

def levenshtein(a, b):
	"Calculates the Levenshtein distance between a and b."
	n, m = len(a), len(b)
	if n > m:
		# Make sure n <= m, to use O(min(n,m)) space
		a, b = b, a
		n, m = m, n

	current = list(range(n+1))
	for i in range(1, m+1):
		previous, current = current, [i]+[0]*n
		for j in range(1, n+1):
			add, delete = previous[j]+1, current[j-1]+1
			change = previous[j-1]
			if a[j-1] != b[i-1]:
				change = change + 1
			current[j] = min(add, delete, change)

	return current[n]

def createDatasetPaths():
	paths=[]
	speakers=[]
	clips=[]

	for x in range(6):
		for y in range(28):
			for key in offsetMap:
				speakers.append(x+1)
				clips.append(y+1)
				paths.append("speaker"+str(x+1)+"clip"+str(y+1)+offsetMap[key]+".wav")

	return paths, speakers, clips

model = FBAudioVisualCPCLightning(batch_size=1).cuda()

checkpoint = torch.load('/home/analysis/Documents/studentHDD/chris/predictiveCodingCharacterExperiment/base_model/epoch=50-step=906218.ckpt')

#print(model.cpc_model.gEncoder.conv0.weight)

#model.load_from_checkpoint('fb_lightning_logs/lightning_logs/version_1/checkpoints/epoch=45-step=1617083.ckpt')
model.load_state_dict(checkpoint['state_dict'])
#print(model.cpc_model.gEncoder.conv0.weight)

libri_path = "/home/analysis/Documents/studentHDD/chris/monoSubclips/"
dest_dir = "/home/analysis/Documents/studentHDD/chris/monoSubclips_embeddings/"

train_output = True

jobs,dumb,test=createDatasetPaths()

for file in tqdm(jobs):
	dest = os.path.join(dest_dir, file.replace('.wav', '.npy'))

	# if not os.path.exists(os.path.join(dest_dir, folder)):
	# 	os.makedirs(os.path.join(dest_dir, folder))

	x_audio, _ = librosa.load(os.path.join(libri_path, file), sr=16000)
	x_visual = np.load(os.path.join(libri_path, file.replace('.wav', '.npy')))

	x_audio_tensor = torch.from_numpy(x_audio).unsqueeze(0).unsqueeze(0).cuda()
	x_visual_tensor = torch.from_numpy(x_visual.T).unsqueeze(0).cuda()

	embedding = model.embedding((x_audio_tensor, x_visual_tensor), context=True, audioVisual=True, norm=False)

	embedding = torch.squeeze(embedding).detach().cpu().numpy()

	print(dest)
	#np.save(dest, embedding)