#!/usr/bin/python
import argparse
import sys
import cv2
import numpy as np

center_val = 32768
width_val = 32768

def on_trackbar_center(val):
    global center_val
    center_val = val
    show_image()

def on_trackbar_width(val):
    global width_val
    width_val = val
    show_image()

def show_image():
  mask = cv2.inRange(depth_data, center_val - width_val, center_val + width_val)
  masked_img = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
  cv2.imshow("Image window", masked_img)
  cv2.imshow("Depth window", mask)
  
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-c", "--colorize", required= False, action="store_true", help="Colorize the depth image")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if(args["colorize"]):
  clipped = np.clip(depth_data, 0, 2000).astype(np.uint8)
  normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  depth_image = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
else:
  depth_image = depth_data
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Center", "Image window" , 0, 2000, on_trackbar_center)
cv2.createTrackbar("Width", "Image window" , 0, 2000, on_trackbar_width)

#cv2.imshow("Depth window", depth_image)
cv2.imshow("Image window", rgb_image)

while True:
  k = cv2.waitKey(10)
  if k == 27:
    break
cv2.destroyAllWindows()
