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

files.sort()

ads=[]
ids=[]
for file in files:
	if file[-5]=="B":
		ids.append(file)
	if file[-5]=="D":
		ads.append(file)

ids.sort()
ads.sort()

vals=[]
for idsFile in tqdm(ids):
	idsNumpy=np.load(os.path.join(lib_path,idsFile))
	print(idsNumpy.shape)
	for adsFile in ads:
		adsNumpy=np.load(os.path.join(lib_path,adsFile))
		vals.append(distance.cosine(idsNumpy,adsNumpy))

print(len(vals))