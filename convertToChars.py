import os
import numpy as np
from tqdm import tqdm

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

walk=os.walk("/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main")

dirs=[]
for x in range(len(walk)):
	dirs.append(walk[x][0])
	#print(entry[0])

for x in range(len(dirs)):
	print(dirs)

# for entry in dirs:
# 	print(entry)
# 	dirWalk=os.walk(entry)
# 	for walkEntry in dirWalk:
# 		files=walkEntry[2]
	
# 	x=0
# 	while x < len(files):
# 		if ".txt" not in files[x]:
# 			files.pop(x)
# 		else:
# 			x+=1

# 	for file in files:
# 		openable=entry+"/"+file
# 		print(openable)
# 		f=open(openable,"r")
# 		print(f.read(0))