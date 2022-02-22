import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2

img = cv2.imread('./test/01998_60year_47th_slice.png')
mask = cv2.imread('./test/result4/01998_60year_47th_slice_OUT.png')

plt.imshow(img)
plt.imshow(mask, cmap = 'seismic', alpha = 0.3, vmin = -255, vmax = 255)
plt.rcParams["figure.figsize"] = (13, 15)
plt.axis('off')
plt.show()