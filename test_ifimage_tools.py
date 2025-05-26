from unittest import TestCase
from ifimage_tools import *
import os
import numpy as np
from stardist.matching import matching
import ifimage_tools

image_dir = "Reorgnized Ground Truth"
masks_dir = "Reorgnized Ground Truth/mask"
dataset = ifimage_tools.IfImageDataset(image_dir, masks_dir, {})
dataset.load_data()
METHODS = ["cyto3", "watershed", "cell_expansion"]

iou_thresholds = np.arange(0.5, 1.0, 0.05)
save_dir = "pre_iou"
os.makedirs(save_dir, exist_ok=True)

class TestCentroidAnalyzer(TestCase):
    def test_find_centers(self):
        self.fail()

    def test_match_cells(self):
        self.fail()

    def test_plot_centroids(self):
        self.fail()

    def test_plot_centroids_with_cyto(self):
        self.fail()

    def test_analyze_image_sample(self):
        self.fail()

    def test_evaluation(self):
        self.fail()
