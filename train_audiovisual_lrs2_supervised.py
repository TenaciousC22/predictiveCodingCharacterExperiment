import os

import torch
import numpy as np
from torch.utils import data
import pytorch_lightning as pl

params={'batch_size': 8,
		'shuffle': True,
		'num_workers': 3}

val_params={'batch_size': 8,
		'shuffle': False,
		'num_workers': 3}

base_dir="/home/analysis/Documents/studentHDD/datasets/LRS2"