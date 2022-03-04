import os
import numpy as np
from tqdm import tqdm

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

walk=os.walk("/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main")

dirs=[]
for entry in walk:
	dirs.append(entry[0])

for entry in dirs:
	walk=os.walk(entry)
	for walkEntry in walk:
		files=walkEntry[2]
	
	x=0
	while x < len(files):
		if ".txt" not in files[x]:
			files.pop(x)
		else:
			x+=1

	for file in files:
		openable=entry+"/"+file
		print(openable)
		f=open(openable,"r")
		print(f.read(0))