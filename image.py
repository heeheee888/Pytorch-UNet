import cv2
import numpy as np

# f = cv2.imread('./data/imgs/00001_51year_44th_slice.png')
# f1 = cv2.imread('./data/masks/00001_51year_44th_slice.png') /255
# #print(f.shape[1])
# a = np.unique(f1)
# print(a)


x = np.arange(12).reshape(3,4)
print(x.shape)
print(x.reshape(-1))