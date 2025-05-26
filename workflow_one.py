# input a image of dapi and a image of marker
# output a mask of nuclei and mask of cell bodies
import glob
import scipy.ndimage as ndimage
import skimage.io as skio
from cellpose import models
from imagecodecs.imagecodecs import none_check
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
from stardist.matching import matching

from preprocessing import *
from ifimage_tools import *

dapi_path = "/Users/macbookair/PycharmProjects/ifimage/dapi.tiff"
marker_path = "/Users/macbookair/PycharmProjects/ifimage/marker.tiff"

dapi = safe_read(dapi_path)
marker = safe_read(marker_path)

model_cyto3 = models.Cellpose(gpu=True, model_type='cyto3')
nuclei, _, _, _ = model_cyto3.eval(dapi, diameter=None, channels=[0,0])
cell_bodies=FindMarker.cellpose2channel(marker,nuclei)

mask_to_rois_zip(marker,"testing.zip")
mask_to_rois_zip(dapi,"testing.zip")

# After the processing we can adjust the rois manually in ImageJ

