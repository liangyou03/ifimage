import importlib
import ifimage_tools
import matplotlib.pyplot as plt
import importlib
import ifimage_tools
import matplotlib.pyplot as plt
importlib.reload(ifimage_tools)
from ifimage_tools import *
import pandas as pd
import os
from itertools import islice
import numpy as np
from stardist.matching import matching
import cv2
import pickle
# Example usage of IfImageDataset
# Make sure you have the following directory structure:
ds = IfImageDataset(image_dir="00_dataset", nuclei_masks_dir="nuc_masks", cell_masks_dir="cell_masks")
ds.load_data()   # ← warnings will fire here if anything is missing/empty
ds.summary(table=True)
