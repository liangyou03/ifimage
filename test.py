
import matplotlib.pyplot as plt
import numpy as np




img = np.load('/ihome/jbwang/liy121/ifimage/11_fulldata/segmentation_test/Snap-6257_mask_marker.npy')
print(img)
print(img.max(), img.min())

plt.imshow(img)
plt.imsave('test.png', img)

#/ihome/jbwang/liy121/ifimage/test.py