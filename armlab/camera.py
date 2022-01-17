"""!
Class to represent the camera.
"""

import cv2
import time
import numpy as np
from PyQt4.QtGui import QImage
from PyQt4.QtCore import QThread, pyqtSignal, QTimer
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from apriltag_ros.msg import *
from cv_bridge import CvBridge, CvBridgeError
import copy

class Camera():
    """!
    @brief      This class describes a camera.
    """
    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        """plane info"""
        self.virtual_plane_depth = 990
        self.size_threshold = 1000
        self.VideoFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.CntFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.rgb_image = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.HSVFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720, 1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.array([])
        self.DepthEmptyFrame = cv2.imread("depth_new_empty.png", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        # mouse clicks & calibration variables
        self.cameraCalibrated = False
        self.intrinsic_matrix = np.array([])
        self.extrinsic_matrix = np.array([])
        self.last_click = np.array([0, 0])
        self.new_click = False
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.tag_detections = np.array([])
        self.tag_detections_imageFrame = np.zeros((4, 3)) # camera frame
        self.tag_locations = [[-250, -25, 0], [250, -25, 0], [250, 275, 0], [-250, 275, 0]]
        """ block info """
        self.block_contours = np.array([])
        self.block_detections_big = []
        self.block_detections_small = []
        # self.block_detections_big = np.array([])
        # self.block_detections_small = np.array([])

        """label block info"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        # self.colors = list((
        #     {'id': 'red', 'color': (19, 10, 120)},
        #     {'id': 'orange', 'color': (0, 74, 176)},
        #     {'id': 'yellow', 'color': (14, 163, 204)},
        #     {'id': 'green', 'color': (75, 175, 22)},
        #     {'id': 'blue', 'color': (138, 83, 0)},
        #     {'id': 'violet', 'color': (100, 40, 85)})
        # )
        self.colors = list((
            {'id': 'red', 'color': (175, 5)},
            {'id': 'orange', 'color': (10, 10)},
            {'id': 'yellow', 'color': (20, 20)},
            {'id': 'green', 'color': (75, 75)},
            {'id': 'blue', 'color': (100, 100)},
            {'id': 'violet', 'color': (120, 120)})
        )  
        self.lower = 600
        self.upper = 975

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None
    
    def convertQtContourFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.rgb_image, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        self.intrinsic_matrix = [[922.357834, 0, 649.483591], [0, 921.135476, 338.516031], [0, 0, 1]]
        self.extrinsic_matrix = [[1, 0, 0, 0], [0, -1, 0, 0.175], [0, 0, 1, 0.98]]
        

    def blockDetector(self, upper = 975):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        # pass
        a = 20
        intrinsic = np.array(self.intrinsic_matrix)
        extrinsic = np.array(self.extrinsic_matrix)
        self.block_detections_big = []
        self.block_detections_small = []
        self.rgb_image = copy.deepcopy(self.CntFrame)
        depth_data = self.DepthFrameRaw + self.virtual_plane_depth - self.DepthEmptyFrame
        hsv_image = self.HSVFrame
        # print(hsv_image)
        mask = np.zeros_like(depth_data, dtype=np.uint8)
        cv2.rectangle(mask, (240, 150),(1100,720), 255, cv2.FILLED)
        cv2.rectangle(mask, (575,414),(740,730), 0, cv2.FILLED)
        cv2.rectangle(self.rgb_image, (240, 150),(1100,720), (255, 0, 0), 2)
        cv2.rectangle(self.rgb_image, (575,414),(740,730), (255, 0, 0), 2)
        thresh = cv2.bitwise_and(cv2.inRange(depth_data, self.lower, upper), mask)
        
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(self.rgb_image, contours, -1, (0,255,255), thickness=1)
        for contour in contours:
            mean, color = self.retrieve_area_color(hsv_image, contour)
            theta = cv2.minAreaRect(contour)[2]
            M = cv2.moments(contour)
            if(M['m00'] <= 50):
                continue
            # try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            max_x = 0
            max_y = 0
            max_z = 10000
            for i in range(cx-a, cx+a):
                for j in range(cy-a, cy+a):
                    tmp_z = self.DepthFrameRaw[j][i]
                    if(tmp_z < max_z):
                        p_pixel_h_tmp = np.array([[i, j, 1]])
                        p_camera_h_tmp = np.linalg.inv(intrinsic).dot(tmp_z*np.transpose(p_pixel_h_tmp))
                        p_camera_h_tmp = np.vstack((p_camera_h_tmp, [[1]]))
                        # print("p_camera_h_tmp", p_camera_h_tmp)
                        # print("extrinsic" ,extrinsic)
                        p_world_h_tmp = np.linalg.inv(extrinsic).dot(p_camera_h_tmp)
                        if (p_world_h_tmp[2] < 930):
                            max_x = i
                            max_y = j
                            max_z = tmp_z
                        


            cv2.putText(self.rgb_image, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
            cv2.putText(self.rgb_image, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
            # cv2.putText(self.rgb_image, str(int(mean)), (cx-50, cy+60), self.font, 0.5, (255,255,255), thickness=2)

            z = self.DepthFrameRaw[cy][cx]
            p_pixel = np.array([[cx, cy, 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))
            p_world = np.linalg.inv(extrinsic).dot(p_camera)


            z_h = self.DepthFrameRaw[max_y][max_x]
            p_pixel_h = np.array([[max_x, max_y, 1]])
            p_camera_h = np.linalg.inv(intrinsic).dot(z_h*np.transpose(p_pixel_h))
            p_camera_h = np.vstack((p_camera_h, [[1]]))
            p_world_h = np.linalg.inv(extrinsic).dot(p_camera_h)
            # print([p_world])
            if(M['m00'] > self.size_threshold):
                self.block_detections_big.append((p_world[0:3], int(theta), color, p_world_h[2]))
                size = "big"
                # tmp = np.array()
                # self.block_detections_big = np.append(self.block_detections_big, np.array([[p_world[0:3], int(theta), color]]))
            else:
                self.block_detections_small.append((p_world[0:3], int(theta), color, p_world_h[2]))
                size = "small"
                # self.block_detections_small = np.append(self.block_detections_small, np.array([[p_world[0:3], int(theta), color]]))
            cv2.putText(self.rgb_image, size, (cx-50, cy+60), self.font, 0.5, (255,255,255), thickness=2)
            #print("highest point: ", p_world_h)
            # except:
            #     print("except")
            #     pass

    # def blockDetector(self):
    #     """!
    #     @brief      Detect blocks from rgb

    #                 TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
    #                 locations in self.block_detections
    #     """
    #     # pass
    #     intrinsic = np.array(self.intrinsic_matrix)
    #     extrinsic = np.array(self.extrinsic_matrix)
    #     self.block_detections_big = []
    #     self.block_detections_small = []
    #     self.rgb_image = copy.deepcopy(self.CntFrame)
    #     depth_data = self.DepthFrameRaw + self.virtual_plane_depth - self.DepthEmptyFrame
    #     hsv_image = self.HSVFrame
    #     # print(hsv_image)
    #     mask = np.zeros_like(depth_data, dtype=np.uint8)
    #     cv2.rectangle(mask, (240, 150),(1100,720), 255, cv2.FILLED)
    #     cv2.rectangle(mask, (575,414),(740,730), 0, cv2.FILLED)
    #     cv2.rectangle(self.rgb_image, (240, 150),(1100,720), (255, 0, 0), 2)
    #     cv2.rectangle(self.rgb_image, (575,414),(740,730), (255, 0, 0), 2)
    #     thresh = cv2.bitwise_and(cv2.inRange(depth_data, self.lower, self.upper), mask)
        
    #     _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(self.rgb_image, contours, -1, (0,255,255), thickness=1)
    #     for contour in contours:
    #         mean, color = self.retrieve_area_color(hsv_image, contour)
    #         theta = cv2.minAreaRect(contour)[2]
    #         M = cv2.moments(contour)
    #         if(M['m00'] == 0):
    #             continue
    #         # try:
    #         cx = int(M['m10']/M['m00'])
    #         cy = int(M['m01']/M['m00'])
    #         cv2.putText(self.rgb_image, color, (cx-30, cy+40), self.font, 1.0, (0,0,0), thickness=2)
    #         cv2.putText(self.rgb_image, str(int(theta)), (cx, cy), self.font, 0.5, (255,255,255), thickness=2)
    #         cv2.putText(self.rgb_image, str(int(mean)), (cx-50, cy+60), self.font, 0.5, (255,255,255), thickness=2)
    #         # print(color, int(theta), cx, cy)

    #         # p_camera = [[cx, cy, 1]]
    #         # p_camera = np.transpose(p_camera)
            
    #         # print("intrinsic", intrinsic)
    #         z = self.DepthFrameRaw[cy][cx]
    #         # z = self.DepthFrameRaw[cy][cx] + self.virtual_plane_depth - self.DepthEmptyFrame[cy][cx]

    #         p_pixel = np.array([[cx, cy, 1]])
    #         p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
    #         p_camera = np.vstack((p_camera, [[1]]))
    #         p_world = np.linalg.inv(extrinsic).dot(p_camera)
    #         # print([p_world])
    #         if(M['m00'] > self.size_threshold):
    #             self.block_detections_big.append((p_world[0:3], int(theta), color))
    #             # tmp = np.array()
    #             # self.block_detections_big = np.append(self.block_detections_big, np.array([[p_world[0:3], int(theta), color]]))
    #         else:
    #             self.block_detections_small.append((p_world[0:3], int(theta), color))
    #             # self.block_detections_small = np.append(self.block_detections_small, np.array([[p_world[0:3], int(theta), color]]))
    #         # except:
    #         #     print("except")
    #         #     pass

    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass


    # def retrieve_area_color(self, data, contour):
    #     labels = self.colors
    #     mask = np.zeros(data.shape[:2], dtype="uint8")
    #     cv2.drawContours(mask, [contour], -1, 255, -1)
    #     mean = cv2.mean(data, mask=mask)[:3]
    #     mean = np.asarray(mean)
    #     # print(mean)
    #     tmp = mean[0]
    #     mean[0] = mean[2]
    #     mean[2] = tmp
    #     min_dist = (np.inf, None)
    #     for label in labels:
    #         d = np.linalg.norm(label["color"] - np.array(mean))
    #         if d < min_dist[0]:
    #             min_dist = (d, label["id"])
    #     return mean, min_dist[1] 

    def retrieve_area_color(self, data, contour):
        labels = self.colors
        mask = np.zeros(data.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [contour], -1, 255, -1)
        data_sin = np.sin(np.deg2rad(2 * (data[:, :, 0]).astype(np.double)))
        data_cos = np.cos(np.deg2rad(2 * (data[:, :, 0]).astype(np.double)))
        # print(data_sin)
        mean_sin = cv2.mean(data_sin, mask=mask)[0]
        mean_cos = cv2.mean(data_cos, mask=mask)[0]
        # print("mean sin", mean_sin)
        # print("mean cos", mean_cos)
        mean = np.arctan2(mean_sin, mean_cos)
        mean = np.rad2deg(mean)
        if mean < 0:
            mean = mean + 360
        mean = mean / 2
        # print("mean", mean)
        # print(mean)
        # while(1):
        #     pass
        min_dist = (np.inf, None)
        for label in labels:
            for i in range(len(label["color"])):
                # print(label["color"][i])
                d = abs(label["color"][i] - np.array(mean))
                if d < min_dist[0]:
                    min_dist = (d, label["id"])
        return mean, min_dist[1]


class ImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        self.camera.CntFrame = copy.deepcopy(cv_image)
        self.camera.HSVFrame = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)



class TagImageListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.TagImageFrame = cv_image


class TagDetectionListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, AprilTagDetectionArray,
                                        self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.tag_detections = data
        for detection in data.detections:
            id = detection.id[0]
            self.camera.tag_detections_imageFrame[id-1][0] = detection.pose.pose.pose.position.x
            self.camera.tag_detections_imageFrame[id-1][1] = detection.pose.pose.pose.position.y
            self.camera.tag_detections_imageFrame[id-1][2] = detection.pose.pose.pose.position.z


class CameraInfoListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.tag_sub = rospy.Subscriber(topic, CameraInfo, self.callback)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.K, (3, 3))
        #print(self.camera.intrinsic_matrix)


class DepthListener:
    def __init__(self, topic, camera):
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            #cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        #self.camera.DepthFrameRaw = self.camera.DepthFrameRaw/2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_image_topic = "/tag_detections_image"
        tag_detection_topic = "/tag_detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        tag_image_listener = TagImageListener(tag_image_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        while True:
            rgb_frame = self.camera.convertQtVideoFrame()
            depth_frame = self.camera.convertQtDepthFrame()
            tag_frame = self.camera.convertQtTagImageFrame()
            cnt_frame = self.camera.convertQtContourFrame()
            if ((rgb_frame != None) & (depth_frame != None)):
                self.updateFrame.emit(rgb_frame, depth_frame, tag_frame, cnt_frame)
            time.sleep(0.03)
            if __name__ == '__main__':
                cv2.imshow(
                    "Image window",
                    cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                cv2.imshow(
                    "Labeled window",
                    cv2.cvtColor(self.camera.rgb_image, cv2.COLOR_RGB2BGR))
                cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                cv2.imshow(
                    "Tag window",
                    cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(3)
                time.sleep(0.03)


if __name__ == '__main__':
    camera = Camera()
    videoThread = VideoThread(camera)
    videoThread.start()
    rospy.init_node('realsense_viewer', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()
