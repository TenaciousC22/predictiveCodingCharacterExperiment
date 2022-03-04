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
	print(walk)
	# for entry in walk:
	# 	files=entry
	# print(files)