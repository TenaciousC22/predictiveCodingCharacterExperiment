import os
import numpy as np
from tqdm import tqdm

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

dirs=os.listdir(base_dir)

for entry in dirs:
	path=base_dir+"/"+entry
	print(path)
	dirWalk=os.walk(path)
	for walkEntry in dirWalk:
		files=walkEntry[2]
	
	x=0
	while x < len(files):
		if ".txt" not in files[x]:
			files.pop(x)
		else:
			x+=1

	print(files)

	for file in files:
		openable=path+"/"+file
		print(openable)
		f=open(openable,"r")
		print(f.read(0))