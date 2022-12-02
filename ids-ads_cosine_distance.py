import os
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm
import random

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
for x in tqdm(range(10)):
	idsNumpy=np.load(os.path.join(lib_path,ids[random.randint(0,len(ids)-1)]))
	# print(idsNumpy.shape)
	adsNumpy=np.load(os.path.join(lib_path,ads[random.randint(0,len(ads)-1)]))
	vals.append(distance.cdist(idsNumpy,adsNumpy,'cosine'))

for val in vals:
	print(val)