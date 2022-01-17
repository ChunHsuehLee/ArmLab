
#!/usr/bin/python
import argparse
import sys
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def mouse_callback(event, u, v, flags, param):
  if(avg_radius == 0):
    d = depth_data[v][u]
  else:
    n=0.0
    d=0.0
    for i in range(-avg_radius, avg_radius, 1):
      for j in range(-avg_radius, avg_radius, 1):
        d += depth_data[v+j][u+i]
        n += 1
    d = d/n
  pt_c = d * np.matmul(np.linalg.inv(K), np.transpose(np.array([u,v,1])))
  pt_c = np.append(pt_c, 1.0)
  pt_w = np.asarray(np.dot(np.linalg.inv(EXT), pt_c))[0]
  output_xyz = "x:%d, y:%d, z:%.2f" % (pt_w[0], pt_w[1], pt_w[2])
  rgb_tmp = rgb_image.copy()
  cv2.putText(rgb_tmp, output_xyz, (10, 20), font, 0.5, (0,0,0))
  cv2.imshow("Image window", rgb_tmp)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the rgb image")
ap.add_argument("-d", "--depth", required = True, help = "Path to the depth image")
ap.add_argument("-a", "--avg_depth", required= True, help="radius of pixels to average for depth measurement")
args = vars(ap.parse_args())
rgb_image = cv2.imread(args["image"])
depth_data = cv2.imread(args["depth"], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
avg_radius = int(args["avg_depth"])
""" Make array of gridpoints """
ypos = 50.0 * np.arange(-2.5, 9.5, 1.0)
xpos = 50.0 * np.arange(-9.0, 10.0, 1.0)
xloc, yloc = np.meshgrid(xpos, ypos)
board_points = np.array(np.meshgrid(xpos, ypos)).T.reshape(-1, 2)
board_points_3D = np.column_stack(
    (board_points, np.zeros(board_points.shape[0])))
board_points_homogenous = np.column_stack(
    (board_points_3D, np.ones(board_points.shape[0])))

""" Camera Parameters measured with checkerboard"""
# K = np.array([[971.3251488347465, 0.0, 678.0257445284223],
#               [0.0, 978.1977572681769, 374.998256815977], [0.0, 0.0, 1.0]])

# dist_coeffs = np.array([
#     0.13974450710810923, -0.1911712119896019, 0.004157844196335278,
#     0.0013002028638032788, 0.0
# ])
''' camera intrinsic matrix & inverse from camera_info ros message '''
K = np.array([[918.3599853515625, 0.0, 661.1923217773438],
              [0.0, 919.1538696289062, 356.59722900390625], [0.0, 0.0, 1.0]])
D = np.array([
    0.15564486384391785, -0.48568257689476013, -0.0019681642297655344,
    0.0007267732871696353, 0.44230175018310547
])

P = np.column_stack((K, [0.0,0.0,0.0]))

""" Naive extrinsic matrix"""
""" I hand tuned this for OK results... """
EXT = np.matrix([[1., 0, 0, -15], [0, -1., 0, 220.],
                             [0, 0, -1., 985.], [0, 0, 0, 1.]])

""" SVD extrinsic matrix 2D"""
# EXT = np.matrix([[ 9.99837123e-01,  1.36355743e-04, -1.80473803e-02, -1.45519497e+01],
#  [ 3.02219941e-04, -9.99957743e-01,  9.18807789e-03,  2.18233033e+02],
#  [-1.80453648e-02, -9.19203564e-03, -9.99794915e-01,  9.88635957e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

""" PnP Ransac """
EXT = np.matrix([[ 9.99889651e-01,  8.52434847e-04,  1.48310077e-02, -1.42681074e+01],
 [ 6.31088030e-04, -9.99888448e-01,  1.49228812e-02,  2.17248105e+02],
 [ 1.48420741e-02, -1.49118748e-02, -9.99778650e-01,  9.87952581e+02],
 [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

""" PnP 3D points"""
# EXT = np.matrix([[ 9.99878526e-01, -7.25187904e-05, -1.55861179e-02, -1.38625409e+01],
#  [-3.23161635e-04, -9.99870668e-01, -1.60792424e-02,  2.16780582e+02],
#  [-1.55829361e-02,  1.60823260e-02, -9.99749234e-01,  9.78215745e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

""" 3D Affine """
# EXT = np.matrix([[ 1.01746643e+00, -1.01215122e-02, -2.54505091e-02, -9.81018859e+00],
#  [ 5.38050557e-03, -1.00326372e+00,  4.05055844e-02,  2.16268596e+02],
#  [-2.51555556e-02, -3.95600000e-02, -1.02073270e+00,  9.99183000e+02],
#  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

""" Project gridpoints to image """
pixel_locations = np.transpose(
    np.matmul(
        P,
        1 / EXT[2, 3] * np.matmul(EXT, np.transpose(board_points_homogenous))))

#rgb_image = cv2.undistort(rgb_image, K, distCoeffs=D)

for element in pixel_locations:
    rgb_image = cv2.circle(rgb_image, (int(element[0, 0]), int(element[0, 1])), 5,
                       (0, 0, 255), 1)

cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
cv2.imshow("Image window", rgb_image)
cv2.setMouseCallback("Image window", mouse_callback)

while True:
  k = cv2.waitKey(0)
  if k == 27:
    break    
cv2.destroyAllWindows()
