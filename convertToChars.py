import os
import numpy as np
from tqdm import tqdm

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main"

dirs=os.walk("/home/analysis/Documents/studentHDD/datasets/LRS2/mvlrs_v1/main")

for entry in dirs:
	print(entry)