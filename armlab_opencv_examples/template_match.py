
#!/usr/bin/python
import argparse
import sys
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-t", "--template", required = True, help = "Path to the depthtemplate image")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
template = cv2.imread(args["template"])
gs_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
template = template[5:-5, 5:-5]
h, w = template.shape[:2]
gs_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
output_image = cv2.matchTemplate(gs_image, gs_template, cv2.TM_CCOEFF_NORMED)
ret, thresh_image = cv2.threshold(output_image, 0.9, 1.0, cv2.THRESH_BINARY)

# go through and find all of the matches above a certain threshold
# loop:
#   find max
#   append to list
#   make the current max black
#   draw a circle on the image
# delete the first element of the initial array
threshold = 0.8
max_val = 1
points = np.array([[0, 0]])
tmp_img = output_image.copy()
while max_val > threshold:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(tmp_img)
    points = np.append(points, [[max_loc[0] + 20, max_loc[1] + 20]], axis=0)
    if max_val > threshold:
        tmp_img[max_loc[1] - h // 2:max_loc[1] + h // 2 + 1,
            max_loc[0] - w // 2:max_loc[0] + w // 2 + 1] = 0
        rgb_image = cv2.circle(rgb_image, (max_loc[0] + 20, max_loc[1] + 20), 10,
                           (255, 255, 0), 1)
points = np.delete(points, 0)
print(points.shape)

cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Output window", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold window", cv2.WINDOW_NORMAL)
cv2.imshow("Threshold window", thresh_image)
cv2.imshow("Output window", output_image)
cv2.imshow("Image window", rgb_image)
k = cv2.waitKey(0)
if k == 27:
    cv2.destroyAllWindows()
