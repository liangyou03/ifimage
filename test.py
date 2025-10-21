
import matplotlib.pyplot as plt
import numpy as np




img = np.load('/ihome/jbwang/liy121/ifimage/012_microsam_benchmark/markeronly_prediction/gfap_3527.npy')
print(img)
print(img.max(), img.min())

plt.imshow(img)
plt.imsave('gfap_3527.png', img)

#/ihome/jbwang/liy121/ifimage/test.py