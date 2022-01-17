
#!/usr/bin/python
# checkout https://learnopencv.com/color-spaces-in-opencv-cpp-python/
import argparse
import sys
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-c", "--colorspace", required = True, help = "Colorspace to apply")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
colorspace = args["colorspace"]
width = int(rgb_image.shape[1] * 0.5)
height = int(rgb_image.shape[0] * 0.5)
dsize = (width, height)
rgb_image = cv2.resize(rgb_image, dsize)
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Channel 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Channel 2", cv2.WINDOW_NORMAL)
cv2.namedWindow("Channel 3", cv2.WINDOW_NORMAL)
""" Internally RGB images are byte order reversed, i.e. BGR"""
if(colorspace == "rgb"):
  image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
if(colorspace == "hsv"):
  image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
if(colorspace == "lab"):
  image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
if(colorspace == "ycrcb"):
  image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2YCrCb)
im1, im2, im3 = cv2.split(image)
cv2.imshow("Image window",image)
cv2.imshow("Channel 1",im1)
cv2.imshow("Channel 2",im2)
cv2.imshow("Channel 3",im3)
while True:
  k = cv2.waitKey(0)
  if k == 27:
    break    
cv2.destroyAllWindows()
