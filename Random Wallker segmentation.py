# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 02:47:06 2021

@author: abc
"""

from skimage import io , img_as_float
import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma

img = img_as_float(io.imread("C:/Users/abc/Desktop/image/random walker.jpg"))



sigma_est = np.mean(estimate_sigma(img, multichannel=True))
print(f"estimated noise standard deviation = {sigma_est}")

patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel=True)

# slow algorithm
denoise_img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                           **patch_kw)


from skimage import exposure
eq_img = exposure.equalize_adapthist(denoise_img)


plt.imshow(eq_img,cmap='gray')
#plt.hist(denoise_img.flat, bins=100, range=(0,1))

markers = np.zeros(img.shape,dtype=np.uint)
markers[(eq_img < 0.6) & (eq_img > 0.3)] = 1
markers[(eq_img > 0.8) & (eq_img < 0.99)] = 2

plt.imshow(markers)


#random walker  


from skimage.segmentation import random_walker
labels = random_walker(eq_img,markers,beta=10,mode='bf')


plt.imshow(labels)

#segmetation in image 

seg1 = (labels == 1)
seg2 = (labels == 2)

all_segments = np.zeros((eq_img.shape[0],eq_img.shape[1],3))
all_segments[seg1] = (1,0,0)

all_segments[seg2] = (0,1,0)

plt.imshow(all_segments)
















