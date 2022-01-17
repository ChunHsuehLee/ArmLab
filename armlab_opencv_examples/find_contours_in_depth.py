
#!/usr/bin/python
""" Example: 

python find_contours_in_depth.py -i image_blocks.png -d depth_blocks.png -l 905 -u 973

"""
import argparse
import sys
import cv2
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-l", "--lower", required = True, help = "lower depth value for threshold")
ap.add_argument("-u", "--upper", required = True, help = "upper depth value for threshold")
args = vars(ap.parse_args())
lower = int(args["lower"])
upper = int(args["upper"])
rgb_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
"""mask out arm & outside board"""
mask = np.zeros_like(depth_data, dtype=np.uint8)
cv2.rectangle(mask, (275,120),(1100,720), 255, cv2.FILLED)
cv2.rectangle(mask, (575,414),(723,720), 0, cv2.FILLED)
cv2.rectangle(rgb_image, (275,120),(1100,720), (255, 0, 0), 2)
cv2.rectangle(rgb_image, (575,414),(723,720), (255, 0, 0), 2)
thresh = cv2.bitwise_and(cv2.inRange(depth_data, lower, upper), mask)
# depending on your version of OpenCV, the following line could be:
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(rgb_image, contours, -1, (0,255,255), 3)
cv2.imshow("Threshold window", thresh)
cv2.imshow("Image window", rgb_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
