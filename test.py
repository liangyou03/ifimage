
import matplotlib.pyplot as plt
import numpy as np

img = np.load('/ihome/jbwang/liy121/ifimage/heart/benchmark_results/omnipose_predictions/RA/RA1_dapi_pred.npy')
print(img)
print(img.max(), img.min())

plt.imshow(img)
plt.imsave('/ihome/jbwang/liy121/ifimage/test3.png', img)

#python /ihome/jbwang/liy121/ifimage/test.py