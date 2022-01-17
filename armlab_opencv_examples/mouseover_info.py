
#!/usr/bin/python
import argparse
import numpy as np
import sys
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

def mouse_callback(event, x, y, flags, param):
  r = rgb_image[y][x][2]
  g = rgb_image[y][x][1]
  b = rgb_image[y][x][0]
  h = hsv_image[y][x][0]
  s = hsv_image[y][x][1]
  v = hsv_image[y][x][2]
  if(avg_radius == 0):
    d = depth_data[y][x]
  else:
    n=0.0
    d=0.0
    for i in range(-avg_radius, avg_radius, 1):
      for j in range(-avg_radius, avg_radius, 1):
        d += depth_data[y+j][x+i]
        n += 1
    d = d/n
  output_uvd = "u:%d, v:%d, d:%.2f" % (x, y, d)
  output_rgb = "r:%d, g:%d, b:%d" % (r, g, b)
  output_hsv = "h:%d, s:%d, v:%d" % (h, s, v)
  rgb_tmp = rgb_image.copy()
  cv2.putText(rgb_tmp, output_uvd, (10, 20), font, 0.5, (0,0,0))
  cv2.putText(rgb_tmp, output_rgb, (10, 40), font, 0.5, (0,0,0))
  cv2.putText(rgb_tmp, output_hsv, (10, 60), font, 0.5, (0,0,0))
  cv2.imshow("Image window", rgb_tmp)
  if event == cv2.EVENT_LBUTTONDOWN:
    print("[%d, %d, %0.2f]," % (x, y, d))
    #print("rgb: [%d, %d, %d]" % (r, g, b))
    #print("hsv: [%d, %d, %d]" % (h, s, v))

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-c", "--colorize", required= False, action="store_true", help="Colorize the depth image")
ap.add_argument("-a", "--avg_depth", required= True, help="radius of pixels to average for depth measurement")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
if(args["colorize"]):
    clipped = np.clip(depth_data, 0, 2000).astype(np.uint8)
    normed = cv2.normalize(clipped, clipped, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_image = cv2.applyColorMap(normed, cv2.COLORMAP_JET)
else:
  depth_image = depth_data
avg_radius = int(args["avg_depth"])
cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
#cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
#cv2.imshow("Depth window", depth_image)
cv2.imshow("Image window", rgb_image)
cv2.setMouseCallback("Image window", mouse_callback)

while True:
  k = cv2.waitKey(10)
  if k == 27:
    break
cv2.destroyAllWindows()
