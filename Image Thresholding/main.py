import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('...png',0)
#Negative image conversion
negative_img = np.max(img) - img
#Default thresholding
thresh_val, threshold_img = cv2.threshold(img,25,255,cv2.THRESH_BINARY)
#Otsu thresholding
thresh_val, threshold_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#Log transform of image
log_output = np.log(1+img).astype(np.uint8)
gama = 0.4
power_law_output = np.power(img,gama).astype(np.uint8)

