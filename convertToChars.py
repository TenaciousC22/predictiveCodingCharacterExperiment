import os
import numpy as np
from tqdm import tqdm

def splitString(base):
	split=base[7:]
	print(split)

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

dirs=os.listdir(base_dir)

x=0
while x < len(dirs):
	if ".wav" in dirs[x]:
		dirs.pop(x)
	else:
		x+=1

for entry in tqdm(dirs):
	path=base_dir+"/"+entry
	dirWalk=os.walk(path)
	for walkEntry in dirWalk:
		files=walkEntry[2]
	
	x=0
	while x < len(files):
		if ".txt" not in files[x]:
			files.pop(x)
		else:
			x+=1

	for file in files:
		openable=path+"/"+file
		f=open(openable,"r")
		text=splitString(f.readline())
		#print(f.read())