import csv
import math
import os
import wave
from torch.utils import data
import librosa
import torch
import numpy as np
from progressbar import progressbar
from tqdm import tqdm
from torch.multiprocessing import Lock, Manager
from generators.librispeech import LRS2UnsupervisedLoader, LRS2AudioVisualPhonemeDataset
from models.audiovisual_model import  FBAudioVisualCPCPhonemeClassifierLightning
from util.pad import audiovisual_batch_collate
from util.seq_alignment import beam_search