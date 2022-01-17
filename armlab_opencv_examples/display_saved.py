
#!/usr/bin/python
import argparse
import numpy as np
import sys
import cv2
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-c", "--colorize", required= False, action="store_true", help="Colorize the depth image")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
depth_image = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if(args["colorize"]):
  clipped = np.clip(depth_image, 0, 2000).astype(np.uint8)
  normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
  depth_image = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
cv2.imshow("Depth window", depth_image)
cv2.imshow("Image window", rgb_image)
while True:
  k = cv2.waitKey(0)
  if k == 27:
    break    
cv2.destroyAllWindows()
