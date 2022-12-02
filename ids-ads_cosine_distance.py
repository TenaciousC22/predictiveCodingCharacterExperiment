import os
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm

lib_path="/home/analysis/Documents/studentHDD/chris/IDS-corpus-edited-ds-embed"

files=[]
for file in os.listdir(lib_path):

	if not file.endswith('.npy'):
		continue

	files.append(file)

ads=[]
ids=[]
for file in files:
	if file[-5]=="B":
		ids.append(file)
	elif files[-5]=="D":
		ads.append(file)

ids.sort()
ads.sort()

for idsFile in tqdm(ids):
	idsNumpy=np.load(os.path.join(lib_path,idsFile))
	for adsFile in ads:
		adsNumpy=np.load(os.path.join(lib_path,adsFile))
		temp=distance.cosine(idsNumpy,adsNumpy)