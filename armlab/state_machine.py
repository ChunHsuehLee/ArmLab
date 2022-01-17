"""!
TEAM 210
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
import rospy
import copy
import cv2
from kinematics import *
from rxarm import RXArm

class StateMachine():

    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """
    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.dist_threshold = 50
        # self.teach_trajectory = np.array([])
        self.teach_trajectory = []
        self.extrinsic_matrix = np.array([[1, 0, 0, 0], [0, -1, 0, 190.5], [0, 0, -1, 980], [0, 0, 0, 1]])
        self.First = True
        self.currenttime = 0
        self.starttime = 0
        self.theta1 = []
        self.theta2 = []
        self.theta3 = []
        self.theta4 = []
        self.theta5 = []
        self.timeList = []
        self.j = 1
        
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]

        pid_values = [[640, 100, 3600],
                        [800, 100, 0],
                        [800, 100 ,0],
                        [800, 100 ,0],
                        [640, 100, 3600],
                        [640, 100, 3600]]
        pid_values = [[640, 0, 3600],
                        [800, 0, 0],
                        [800, 0 ,0],
                        [800, 0 ,0],
                        [640, 0, 3600],
                        [640, 0, 3600]]
        for i in range(len(self.rxarm.joint_names)):
            # print("joint name: ", name)
            self.rxarm.set_joint_position_pid_params(self.rxarm.joint_names[i], pid_values[i])
            # print(self.rxarm.get_joint_position_pid_params(self.rxarm.joint_names[i]))


    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "teach":
            self.teach()

        if self.next_state == "repeat":
            self.repeat()

        if self.next_state == "clear":
            self.clear()

        if self.next_state == "pick&place":
            self.pickAndPlace()
        
        if self.next_state == "pickNsort":
            self.pickNsort()

        if self.next_state == "pickNstack":
            self.pickNstack()

        if self.next_state == "lineMup":
            self.lineMup()

        if self.next_state == "stackMhigh":
            self.stackMhigh()

        if self.next_state == "sky":
            self.sky()

        if self.next_state == "unstack":
            self.unstack()

        if self.next_state == "stack":
            self.stack()

        if self.next_state == "get_data":
            self.get_data()

    """Functions run for each state"""
    

    def get_data(self):
        # p0 = self.camera.tag_locations[0]
        # p1 = self.camera.tag_locations[1]
        # p2 = self.camera.tag_locations[2]
        # p3 = self.camera.tag_locations[3]
        # print(p0, p1, p2, p3)
        tag_location_image = [[414, 308], [885, 310], [411, 594], [887, 596]]
        # tag_location_image = [[419, 308], [819, 317], [412, 594], [889, 603]]
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        x4 = []
        y4 = []

        with open('joints.txt', 'w') as f:
            print >> f, self.timeList
            print >> f, self.theta1
            print >> f, self.theta2
            print >> f, self.theta3
            print >> f, self.theta4
            print >> f, self.theta5




        # for item in my_list:
            # f.write("%s\n" % item)
        for i in range(50):

            self.calibrate()
            intrinsic = self.camera.intrinsic_matrix
            extrinsic = self.camera.extrinsic_matrix

            cx = tag_location_image[0][0]
            cy = tag_location_image[0][1]
            z = self.camera.DepthFrameRaw[cy][cx]
            p_pixel = np.array([[cx, cy, 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))
            p_world = np.linalg.inv(extrinsic).dot(p_camera)
            x1.append(list(p_world[0])[0])
            y1.append(list(p_world[1])[0])

            cx = tag_location_image[1][0]
            cy = tag_location_image[1][1]
            z = self.camera.DepthFrameRaw[cy][cx]
            p_pixel = np.array([[cx, cy, 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))
            p_world = np.linalg.inv(extrinsic).dot(p_camera)
            x2.append(list(p_world[0])[0])
            y2.append(list(p_world[1])[0])

            cx = tag_location_image[2][0]
            cy = tag_location_image[2][1]
            z = self.camera.DepthFrameRaw[cy][cx]
            p_pixel = np.array([[cx, cy, 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))
            p_world = np.linalg.inv(extrinsic).dot(p_camera)
            x3.append(list(p_world[0])[0])
            y3.append(list(p_world[1])[0])

            cx = tag_location_image[3][0]
            cy = tag_location_image[3][1]
            z = self.camera.DepthFrameRaw[cy][cx]
            p_pixel = np.array([[cx, cy, 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))
            p_world = np.linalg.inv(extrinsic).dot(p_camera)
            x4.append(list(p_world[0])[0])
            y4.append(list(p_world[1])[0])
            rospy.sleep(0.2)


        print("x1", x1)
        print("y1", y1)
        print("x2", x2)
        print("y2", y2)
        print("x3", x3)
        print("y3", y3)
        print("x4", x4)
        print("y4", y4)
        with open('heatmap.txt', 'w') as f:
            print >> f, x1
            print >> f, y1
            print >> f, x2
            print >> f, y2
            print >> f, x3
            print >> f, y3
            print >> f, x4
            print >> f, y4
            # break





    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.current_state = "execute"
        self.status_message = "State: Execute - Executing motion plan"

        for position in self.waypoints:
            self.rxarm.set_joint_positions(position, 2.0, 0.5, True)
            rospy.sleep(2)
        
        self.next_state = "idle"
    def calibrate(self):
        """
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        # self.camera.intrinsic_matrix = [[922.357834, 0, 649.483591], [0, 921.135476, 338.516031], [0, 0, 1]] # 3x3
        # factory 
        intrinsic = np.array(self.camera.intrinsic_matrix)

        tags_camera_frame = intrinsic.dot(self.camera.tag_detections_imageFrame.T) # 3x3 @ 3x4
        # print(self.camera.tag_detections_imageFrame[:, -1])
        # self.camera.virtual_plane_depth = np.mean(self.camera.tag_detections_imageFrame[:, -1]) * 1000
        # print(self.camera.virtual_plane_depth)
        # dist_mat = np.array([0.118240, -0.216326, -0.000765, 0.007087, 0.000000])
        dist_mat = np.array([0.1536298245191574, -0.4935448169708252, -0.0008808146812953055, 0.0008218809380196035, 0.4401721954345703])
        for i in range(len(tags_camera_frame)):
            for j in range(len(tags_camera_frame[0])):
                tags_camera_frame[i][j] = tags_camera_frame[i][j] / tags_camera_frame[-1][j]
            
        tag_locations = np.asarray(self.camera.tag_locations)
        tags_camera_frame = np.asarray(tags_camera_frame).T[:, :2]
        # print(tags_camera_frame)

        tags_camera_frame = tags_camera_frame.astype('double')
        tag_locations = tag_locations.astype('double')
        found, rvec, tvec = cv2.solvePnP(tag_locations, tags_camera_frame, intrinsic, dist_mat, flags=0)
        rmat, _ = cv2.Rodrigues(rvec, np.zeros((3, 3)))
        self.extrinsic_matrix = np.hstack((rmat, tvec))
        self.camera.extrinsic_matrix = np.vstack((self.extrinsic_matrix, np.array([0, 0, 0, 1])))
        # self.camera.extrinsic_matrix = self.extrinsic_matrix
        self.extrinsic_matrix = self.camera.extrinsic_matrix
        # print(self.extrinsic_matrix)
        self.status_message = "Calibration - Completed Calibration"


    

    """ TODO """

    def unstack(self):
        self.current_state = "unstack"
        self.calibrate()
        height_dic = {0: 910, 1: 935, 2: 960, 3: 975}
        ######################################################################
        #### for big blocks, position 1 and 6 are too close to each other ####
        ######################################################################
        bigDic = {
            9: [150, -5], 8: [220, -5], 7: [290, -5],
            6: [360, -5], 5: [110, -75], 4: [180, -75],
            3: [250, -75], 2: [320, -75], 1: [380, -75]
        }
        # bigDic = {
        #     9: [150, -15], 8: [220, -15], 7: [290, -15],
        #     6: [360, -15], 5: [110, -65], 4: [180, -65],
        #     3: [250, -65], 2: [320, -65], 1: [380, -65]
        # }
        smallDic = {
            9: [-130, -15], 8: [-210, -15], 7: [-290, -15],
            6: [-130, -65], 1: [-380, -65], 2: [-330, -65],
            3: [-280, -65], 4: [-230, -65], 5: [-180, -65]
        }

        gridOffsetRight = 45
        gridOffsetLeft = 28

        complete = False
        offsetBig = 1
        zOffsetBig = 0
        offsetSmall = 1
        zOffsetSmall = 0

        depth_dix = 0
        while(not complete):
            for block in self.camera.block_detections_big:
                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                if y > 5:
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [150]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [50 + zOffsetBig]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [200]]), 0.5)
                    offsetBig = offsetBig + 1
                    moved = True
                    if (offsetBig > 9):
                        offsetBig = 1
                        zOffsetBig = zOffsetBig + 50

            for block in self.camera.block_detections_small:
                print("small blocks")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                print(block, "block")
                if y > 5:
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    if(x > 50):
                        offset = gridOffsetRight
                    else:
                        offset = gridOffsetLeft
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 30]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [150]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [50 + zOffsetSmall]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [200]]), 0.5)
                    offsetSmall = offsetSmall + 1
                    moved = True
                    if (offsetSmall > 9):
                        offsetSmall = 1
                        zOffsetSmall = zOffsetSmall + 25
                #depthVal = depthVal + 20
                        
            complete = True
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
            rospy.sleep(2)
            if(depth_dix > 3):
                depth_dix = 3
            
            self.camera.blockDetector(height_dic[depth_dix])
            print(height_dic[depth_dix])

            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            
            if(depth_dix != 3):
                complete = False
            for block in self.camera.block_detections_big:
                if block[0][1][0] > 5:
                    print(block, "bad block")
                    complete = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] > 5:
                    print(block, "bad block")
                    complete = False
                    break
                
            depth_dix += 1
                    
        print("done unstack")
        self.next_state = "idle"
        


    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.current_state = "detect"
        # 910 -> 2 small on 1 big
        # 935 -> 2 big or 1 big 1 small
        # 950 -> 2 small

        test_value = 960
        self.camera.blockDetector()
        self.camera.blockDetector()
        # self.camera.blockDetector(test_value)
        # self.camera.blockDetector(test_value)


        self.next_state = "idle"
        # rospy.sleep(1)


    def pickNsort2(self):
        """!
        @brief      
        """
        self.current_state = "pickNsort"
        bigDic = {
            9: [300, 70], 8: [300, 150], 7: [300, -10],
            6: [400, -50], 5: [400, 30], 4: [400, 110],
            3: [400, 190], 2: [400, 270], 1: [400, 350]
        }
        smallDic = {
            9: [-130, -15], 8: [-210, -15], 7: [-290, -15],
            6: [-130, -65], 1: [-380, -65], 2: [-330, -65],
            3: [-280, -65], 4: [-230, -65], 5: [-180, -65]
        }

        complete = False
        offsetBig = 1
        zOffsetBig = 0
        offsetSmall = 1
        zOffsetSmall = 0

        while(not complete):
            
            for block in self.camera.block_detections_big:
                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                if x < 200 and y > 5:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [150]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [50 + zOffsetBig]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [200]]), 0.5)
                    offsetBig = offsetBig + 1
                    if (offsetBig > 9):
                        offsetBig = 1
                        zOffsetBig = zOffsetBig + 50

            for block in self.camera.block_detections_small:
                print("small blocks")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                print(block, "block")
                if y > 5 and x < 200:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 35]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [150]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [50 + zOffsetSmall]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [200]]), 0.5)
                    offsetSmall = offsetSmall + 1
                    if (offsetSmall > 9):
                        offsetSmall = 1
                        zOffsetSmall = zOffsetSmall + 25
                    
            complete = True
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            
            for block in self.camera.block_detections_big:
                if block[0][1][0] > 0 and block[0][0][0] < 200:
                    print(block, "bad block")
                    complete = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] > 0 and block[0][0][0] < 200:
                    print(block, "bad block")
                    complete = False
                    break
                    
        print("done event 1")
        self.next_state = "idle"




    def pickNsort(self):
        """!
        @brief      
        """
        self.current_state = "pickNsort"
        gridOffsetRight = 45
        gridOffsetLeft = 28
        bigDic = {
            9: [150, -15], 8: [220, -15], 7: [290, -15],
            6: [360, -15], 5: [110, -65], 4: [180, -65],
            3: [250, -65], 2: [320, -65], 1: [380, -65]
        }
        smallDic = {
            9: [-130, -15], 8: [-210, -15], 7: [-290, -15],
            6: [-130, -65], 1: [-380, -65], 2: [-330, -65],
            3: [-280, -65], 4: [-230, -65], 5: [-180, -65]
        }
        depthVal = 850
        complete = False
        offsetBig = 1
        zOffsetBig = 0
        offsetSmall = 1
        zOffsetSmall = 0

        moved = True

        while(not complete):
            # while(depthVal > 0):

            #     if(moved):

            #         self.rxarm.go_to_home_pose(1.2, 0.5, True)
            #         self.rxarm.go_to_sleep_pose(1.2, 0.5, False)
            #         rospy.sleep(2)
                    
            #         self.camera.blockDetector(depthVal)

            #     else:
            #         rospy.sleep(2)
                    
            #         self.camera.blockDetector(depthVal)

            #     moved = False
                
            for block in self.camera.block_detections_big:
                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                if y > 5:
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 10]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [150]]), 0.5)
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [50 + zOffsetBig]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [200]]), 0.5)
                    offsetBig = offsetBig + 1
                    moved = True
                    if (offsetBig > 9):
                        offsetBig = 1
                        zOffsetBig = zOffsetBig + 50

            for block in self.camera.block_detections_small:
                print("small blocks")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                print(block, "block")
                if y > 5:
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    if(x > 50):
                        offset = gridOffsetRight
                    else:
                        offset = gridOffsetLeft
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [150]]), 0.5)
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [50 + zOffsetSmall]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [200]]), 0.5)
                    offsetSmall = offsetSmall + 1
                    moved = True
                    if (offsetSmall > 9):
                        offsetSmall = 1
                        zOffsetSmall = zOffsetSmall + 25
                #depthVal = depthVal + 20
                        
            complete = True
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            
            for block in self.camera.block_detections_big:
                if block[0][1][0] > 0:
                    print(block, "bad block")
                    complete = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] > 0:
                    print(block, "bad block")
                    complete = False
                    break
                
                    
        print("done event 1")
        self.next_state = "idle"







    def get_lowest_position(self):
        dic = {
            1: [350, -100], 2: [250, -100], 3: [150, -100]
        }

        pos = np.array([0, 0, 0])
        block_detections = copy.deepcopy(self.camera.block_detections_big)
        block_detections = block_detections + self.camera.block_detections_small
        for block in block_detections:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            z_h = block[-1]
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[1])) < self.dist_threshold):
                pos[0] = z_h
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[2])) < self.dist_threshold):
                pos[1] = z_h
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[3])) < self.dist_threshold):
                pos[2] = z_h
        tower = np.argmin(pos) + 1
        towerDepth = pos[tower - 1]
        print(pos, "psssssssssss")

        
        return tower, towerDepth

    def get_lowest_index(self):
        dic = {
            0: [205, 325], 1: [-100, 225], 2: [480, 550]
        }

        pos = np.array([0, 0, 0])
        block_detections = copy.deepcopy(self.camera.block_detections_big)
        block_detections = block_detections + self.camera.block_detections_small
        for block in block_detections:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            z_h = block[-1]
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[1])) < self.dist_threshold):
                pos[0] = z_h
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[2])) < self.dist_threshold):
                pos[1] = z_h
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[3])) < self.dist_threshold):
                pos[2] = z_h
        idx = np.argmin(pos) + 1
        depth = pos[idx - 1]

        
        return idx, depth

    def getTowerDepth(self, tower):
        dic = {
            1: [350, -100], 2: [250, -100], 3: [150, -100]
        }

        block_detections = copy.deepcopy(self.camera.block_detections_big)
        block_detections = block_detections + self.camera.block_detections_small
        towerDepth = 0
        for block in block_detections:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            if(np.linalg.norm(np.array([x, y]) - np.array(dic[tower])) < self.dist_threshold):
                towerDepth = z


        return towerDepth

    def moveblocks(self):
        self.getDetection()
        complete = False

        while (not complete):

            for block in self.camera.block_detections_big:
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                if block[0][1][0] < 0:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[x], [y+200], [200]]), 0.5)
                    #print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[x], [y+200], [65]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[x], [y+200], [200]]), 0.5)

            for block in self.camera.block_detections_small:
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                if block[0][1][0] < 0:

                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[x], [y+200], [200]]), 0.5)
                    #print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[x], [y+200], [65]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[x], [y+200], [200]]), 0.5)

            self.getDetection()
            complete = True

            for block in self.camera.block_detections_big:
                if block[0][1][0] < 0:
                    print(block, "bad block")
                    complete = False
                    noBigBlocks = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] < 0:
                    print(block, "bad block")
                    complete = False
                    break






    def stack(self):
        self.current_state = "stack"
        dic = {
            1: [350, -100], 2: [250, -100], 3: [150, -100]
        }
        # self.calibrate()
        self.unstack2()

        #self.moveblocks()
        placed = [0, 0, 0]
        first = True

        complete = False
        noBigBlocks = False
        threshold = 95



        while(not complete):

            if (placed[0] < 3 and placed[1] < 3 and placed[2] < 3):
            
                for block in self.camera.block_detections_big:
                    y = block[0][1][0]
                    x = block[0][0][0]
                    if y < threshold and x < -40:
                        break

                if (not first):
                    self.getDetection()
                    first = False
                tower, towerDepth = self.get_lowest_position()
                tower = np.argmin(placed) + 1
                towerDepth = self.getTowerDepth(tower) + 25
                print(towerDepth, "tower depth")


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]
                gridOffset = 0
                if (abs(x) < 250):
                    gridOffset = 6
                else:
                    gridOffset = -8

                if placed[tower - 1] == 0:
                    gridOffset = 25

                print(gridOffset, "offset")
                if y < threshold and x < -40:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [50 - gridOffset + towerDepth]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    placed[tower-1] = placed[tower-1] + 1



                if(noBigBlocks):
                    for block in self.camera.block_detections_small:
                        y = block[0][1][0]
                        x = block[0][0][0]
                        if y < threshold and x < -40:
                            break

                    print("small blocks")
                    x = block[0][0][0]
                    y = block[0][1][0]
                    z = block[0][2][0]


                    print(block, "block")
                    if y < threshold and x < -40:
                        self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                        self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                        self.rxarm.close_gripper()
                        self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        # change z from 35 to 45
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [50 - gridOffset + towerDepth]]), 0.5)
                        self.rxarm.open_gripper()
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        placed[tower-1] = placed[tower-1] + 1
                print(placed, "placed")

            else:
                for block in self.camera.block_detections_big:
                    y = block[0][1][0]
                    x = block[0][0][0]
                    if y < threshold and x < -40:
                        break

                if (not first):
                    self.getDetection()
                    first = False
                tower, towerDepth = self.get_lowest_position()
                # tower = np.argmin(placed) + 1
                # towerDepth = self.getTowerDepth(tower) + 25
                print(towerDepth, "tower depth")


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]
                gridOffset = 0
                if (abs(x) < 250):
                    gridOffset = 6
                else:
                    gridOffset = -8




                if y < threshold and x < -40:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [50 - gridOffset + towerDepth]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)




                if(noBigBlocks):
                    for block in self.camera.block_detections_small:
                        y = block[0][1][0]
                        x = block[0][0][0]
                        if y < threshold and x < -40:
                            break

                    print("small blocks")
                    x = block[0][0][0]
                    y = block[0][1][0]
                    z = block[0][2][0]


                    print(block, "block")
                    if y < threshold and x < -40:
                        self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                        self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                        self.rxarm.close_gripper()
                        self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [50 - gridOffset + towerDepth]]), 0.5)
                        self.rxarm.open_gripper()
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)


                
            complete = True
            noBigBlocks = True
            self.getDetection()
            
            for block in self.camera.block_detections_big:
                y = block[0][1][0]
                x = block[0][0][0]
                if y < threshold and x < -40:
                    print(block, "bad block")
                    complete = False
                    noBigBlocks = False
                    break
            for block in self.camera.block_detections_small:
                y = block[0][1][0]
                x = block[0][0][0]
                if y < threshold and x < -40:
                    print(block, "bad block small")
                    complete = False
                    break
                    
        print("done")
        self.next_state = "idle"  

        


    def pickNstack(self):
        """!
        @brief      
        """
        self.current_state = "pickNstack"

        self.calibrate()
        self.detect()

        dic = {
            1: [-350, -50], 2: [-140, -50], 3: [175, -50]
        }
        placed = [0, 0, 0]
        first = True

        complete = False
        noBigBlocks = False




        while(not complete):

            if (placed[0] < 3 and placed[1] < 3 and placed[2] < 3):
            
                for block in self.camera.block_detections_big:
                    y = block[0][1][0]
                    if y > -15:
                        break

                if (not first):
                    self.getDetection()
                    first = False
                #tower, towerDepth = self.get_lowest_position()
                tower = np.argmin(placed) + 1
                towerDepth = self.getTowerDepth(tower) + 25
                print(towerDepth, "tower depth")


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]
                gridOffset = 0
                if (abs(x) < 250):
                    gridOffset = 6
                else:
                    gridOffset = -8

                if placed[tower - 1] == 0:
                    gridOffset = 25

                print(gridOffset, "offset")
                if y > -15:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [55 - gridOffset + towerDepth]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    placed[tower-1] = placed[tower-1] + 1



                if(noBigBlocks):
                    for block in self.camera.block_detections_small:
                        y = block[0][1][0]
                        if y > -15:
                            break

                    print("small blocks")
                    x = block[0][0][0]
                    y = block[0][1][0]
                    z = block[0][2][0]


                    print(block, "block")
                    if y > -15:
                        self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                        self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                        self.rxarm.close_gripper()
                        self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        # change z from 35 to 45
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [45 - gridOffset + towerDepth]]), 0.5)
                        self.rxarm.open_gripper()
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        placed[tower-1] = placed[tower-1] + 1
                print(placed, "placed")

            else:
                for block in self.camera.block_detections_big:
                    y = block[0][1][0]
                    if y > -15:
                        break

                if (not first):
                    self.getDetection()
                    first = False
                tower, towerDepth = self.get_lowest_position()
                # tower = np.argmin(placed) + 1
                # towerDepth = self.getTowerDepth(tower) + 25
                print(towerDepth, "tower depth")


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]
                gridOffset = 0
                if (abs(x) < 250):
                    gridOffset = 6
                else:
                    gridOffset = -8




                if y > -15:
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                    print([dic[tower][0], dic[tower][1]], "went")
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [55 - gridOffset + towerDepth]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)




                if(noBigBlocks):
                    for block in self.camera.block_detections_small:
                        y = block[0][1][0]
                        if y > -15:
                            break

                    print("small blocks")
                    x = block[0][0][0]
                    y = block[0][1][0]
                    z = block[0][2][0]


                    print(block, "block")
                    if y > -15:
                        self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                        self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                        self.rxarm.close_gripper()
                        self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [55 - gridOffset + towerDepth]]), 0.5)
                        self.rxarm.open_gripper()
                        self.goto(np.array([[dic[tower][0]], [dic[tower][1]], [200]]), 0.5)


                
            complete = True
            noBigBlocks = True
            self.getDetection()
            
            for block in self.camera.block_detections_big:
                if block[0][1][0] > 0:
                    print(block, "bad block")
                    complete = False
                    noBigBlocks = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] > 0:
                    print(block, "bad block")
                    complete = False
                    break
                    
        print("done")
        self.next_state = "idle"        




    def getDetection(self):
        self.rxarm.go_to_home_pose(1.2, 0.5, True)
        self.rxarm.go_to_sleep_pose(1.2, 0.5, False)
        rospy.sleep(2)
        
        self.camera.blockDetector()
        self.rxarm.go_to_home_pose(1.2, 0.5, True)





    def lineMup(self):
        """!
        @brief      
        """
        self.current_state = "lineMup"
        self.calibrate()

        bigDic = {
            1: [-125, 300], 2: [-75, 300], 3: [-25, 300],
            4: [25, 300], 5: [75, 300], 6: [125, 300]
        }
        smallDic = {
            1: [-25, 200], 2: [15, 200], 3: [55, 200],
            4: [95, 200], 5: [135, 200], 6: [175, 200]
        }

        # colorDic = {
        #     'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'violet': 6
        # }
        colorDic = {
            1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'violet'
        }

        placeDicBig = {
            1: [-125, 300], 2: [-75, 300], 3: [-25, 300], 4: [25, 300], 5: [75, 300], 6: [125, 300]
        }
        placeDicSmall = {
            1: [-25, 200], 2: [15, 200], 3: [55, 200], 4: [95, 200], 5: [135, 200], 6: [175, 200]
        }

        


        complete = False
        bigLine = 1
        smallLine = 1
        zOffsetBig = 0
        zOffsetSmall = 0
        colorIdxBig = 1
        colorIdxSmall = 1

        # clear = self.clearTheCourt(-380, 380, 330, 125, self.camera.block_detections_big, self.camera.block_detections_small, bigLine, smallLine)
        # while(not clear):
        #     self.rxarm.go_to_home_pose(2.0, 0.5, True)
        #     self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
        #     rospy.sleep(2)
            
        #     self.camera.blockDetector()
        #     self.rxarm.go_to_home_pose(2.0, 0.5, True)
        #     clear = self.clearTheCourt(-400, 370, 300, 175, bigLine, smallLine)

        # self.pickNsort2()
        self.unstack()
        
        

        while(not complete):



            found = False
            for block in self.camera.block_detections_big:
                print(block, "for loop block")
                if block[2] == colorDic[colorIdxBig] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    found = True
                    print("found")
                    break
                    
            if(found):


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                self.rxarm.go_to_home_pose(1.2, 0.5, True)
                self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                self.goto(np.array([[x], [y], [z + 30]]), 1.2)
                self.rxarm.close_gripper()
                self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                self.gotoRot(np.array([[placeDicBig[colorIdxBig][0]], [placeDicBig[colorIdxBig][1]], [150]]), 1.48, 0.5)
                self.gotoRot(np.array([[placeDicBig[colorIdxBig][0]], [placeDicBig[colorIdxBig][1]], [50 + zOffsetBig]]), 1.48, 0.5)
                self.rxarm.open_gripper()
                self.gotoRot(np.array([[placeDicBig[colorIdxBig][0]], [placeDicBig[colorIdxBig][1]], [200]]), 1.48, 0.5)
                bigLine = bigLine + 1

            colorIdxBig = colorIdxBig + 1
            if (colorIdxBig == 7):
                colorIdxBig = 1
            if (bigLine == 10):
                bigLine = 1

        
        
            found = False
            for block in self.camera.block_detections_small:
                if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    found = True
                    break
                    
            if(found):

                print("small blocks")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                print(block, "block")
                self.rxarm.go_to_home_pose(1.2, 0.5, True)
                self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                self.rxarm.close_gripper()
                self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [150]]), 1.48, 0.5)
                self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [50 + zOffsetSmall]]), 1.48, 0.5)
                self.rxarm.open_gripper()
                self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [200]]), 1.48, 0.5)
                smallLine = smallLine + 1
                
            colorIdxSmall = colorIdxSmall + 1
            if (colorIdxSmall == 7):
                colorIdxSmall = 1
            if (smallLine == 10):
                smallLine = 1


            self.rxarm.go_to_home_pose(1.2, 0.5, True)
            self.rxarm.go_to_sleep_pose(1.2, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()

            complete = True
            for block in self.camera.block_detections_big:
                if block[0][1][0] < 20 or block[0][0][0] > 200:
                    complete = False
            for block in self.camera.block_detections_small:
                if block[0][1][0] < 20 or block[0][0][0] > 200:
                    complete = False  

            
            # for block in self.camera.block_detections_big:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         noBigBlocks = False
            #         break
            # for block in self.camera.block_detections_small:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         break
                    
        print("done event 3")
        self.next_state = "idle"  



    # def clearTheCourt(self, x1, x2, y1, y2, offsetBig, offsetSmall):
    #     clear = True
    #     zOffsetBig = 0
    #     zOffsetSmall = 0
    #     bigDic = {
    #         9: [160, -35], 8: [260, -35], 7: [360, -35],
    #         6: [460, -35], 5: [160, -100], 4: [240, -100],
    #         3: [320, -100], 2: [400, -100], 1: [480, -100]
    #     }
    #     smallDic = {
    #         9: [-110, -35], 8: [-220, -35], 7: [-320, -35],
    #         6: [-420, -35], 1: [-430, -100], 2: [-350, -100],
    #         3: [-270, -100], 4: [-190, -100], 5: [-110, -100]
    #     }
    #     for block in self.camera.block_detections_big:
    #         x = block[0][0][0]
    #         y = block[0][1][0]
    #         z = block[0][2][0]
    #         if (x > x1 and x < x2) and (y > y1 and y < y2):
    #                 clear = False
    #                 self.goto(np.array([[x], [y], [z + 150]]), 0.5)
    #                 self.goto(np.array([[x], [y], [z + 20]]), 1.2)
    #                 self.rxarm.close_gripper()
    #                 self.goto(np.array([[x], [y], [z + 200]]), 0.5)
    #                 self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [150]]), 0.5)
    #                 self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [50 + zOffsetBig]]), 0.5)
    #                 self.rxarm.open_gripper()
    #                 self.goto(np.array([[bigDic[offsetBig][0]], [bigDic[offsetBig][1]], [200]]), 0.5)
    #                 offsetBig = offsetBig + 1
    #                 if (offsetBig > 9):
    #                     offsetBig = 1
    #                     zOffsetBig = zOffsetBig + 50
    #     for block in self.camera.block_detections_small:
    #         x = block[0][0][0]
    #         y = block[0][1][0]
    #         z = block[0][2][0]
    #         if (x > x1 and x < x2) and (y > y1 and y < y2):
    #                 clear = False
    #                 self.goto(np.array([[x], [y], [z + 150]]), 0.5)
    #                 self.goto(np.array([[x], [y], [z + 20]]), 1.2)
    #                 self.rxarm.close_gripper()
    #                 self.goto(np.array([[x], [y], [z + 200]]), 0.5)
    #                 self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [150]]), 0.5)
    #                 self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [50 + zOffsetSmall]]), 0.5)
    #                 self.rxarm.open_gripper()
    #                 self.goto(np.array([[smallDic[offsetSmall][0]], [smallDic[offsetSmall][1]], [200]]), 0.5)
    #                 offsetSmall = offsetSmall + 1
    #                 if (offsetSmall > 9):
    #                     offsetSmall = 1
    #                     zOffsetSmall = zOffsetSmall + 25

    #     return clear










    def stackMhigh(self):
        """!
        @brief      
        """
        self.current_state = "stackMhigh"
        self.calibrate()

        bigDic = {
            1: [-125, 300], 2: [-75, 300], 3: [-25, 300],
            4: [25, 300], 5: [75, 300], 6: [125, 300]
        }
        smallDic = {
            1: [-25, 200], 2: [15, 200], 3: [55, 200],
            4: [95, 200], 5: [135, 200], 6: [175, 200]
        }

        # colorDic = {
        #     'red': 1, 'orange': 2, 'yellow': 3, 'green': 4, 'blue': 5, 'violet': 6
        # }
        colorDic = {
            1: 'red', 2: 'orange', 3: 'yellow', 4: 'green', 5: 'blue', 6: 'violet'
        }

        placeDicBig = {
            1: [-125, 300], 2: [-75, 300], 3: [-25, 300], 4: [25, 300], 5: [75, 300], 6: [125, 300]
        }
        placeDicSmall = {
            1: [-25, 200], 2: [15, 200], 3: [55, 200], 4: [95, 200], 5: [135, 200], 6: [175, 200]
        }

        smallTower = [-330, 350]
        bigTower = [350, 250]

        


        complete = False
        bigLine = 1
        smallLine = 1
        zOffsetBig = 0
        zOffsetSmall = 0
        colorIdxBig = 1
        colorIdxSmall = 1

        # clear = self.clearTheCourt(-380, 380, 330, 125, self.camera.block_detections_big, self.camera.block_detections_small, bigLine, smallLine)
        # while(not clear):
        #     self.rxarm.go_to_home_pose(2.0, 0.5, True)
        #     self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
        #     rospy.sleep(2)
            
        #     self.camera.blockDetector()
        #     self.rxarm.go_to_home_pose(2.0, 0.5, True)
        #     clear = self.clearTheCourt(-400, 370, 300, 175, bigLine, smallLine)

        # self.pickNsort2()
        self.unstack()
        
        
        z_target_big = 0
        big_off = 65
        miss_big = False
        i = 1
        while(not complete):

            z_temp_big = z_target_big
            if (i >= 4 and miss_big):
                z_target_big = z_temp_big + 42
            else:
                bigTower[0], bigTower[1], z_target_big = self.getTargetZ(bigTower)

            if(i == 1):
                z_target_big = -10
            


            found = False
            for block in self.camera.block_detections_big:
                print(block, "for loop block")
                if block[2] == colorDic[colorIdxBig] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    found = True
                    print("found")
                    break
                    
            if(found):


                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                self.rxarm.go_to_home_pose(1.2, 0.5, True)
                self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                self.goto(np.array([[x], [y], [z + 30]]), 1.2)
                self.rxarm.close_gripper()
                self.goto(np.array([[x], [y], [z + 100 + z_target_big]]), 0.5)
                self.goto(np.array([[bigTower[0]], [bigTower[1]], [450]]), 0.5)
                print("going to place now at", bigTower[0], bigTower[1], big_off+z_target_big)
                self.gotoSlow(np.array([[bigTower[0]], [bigTower[1]], [big_off + z_target_big]]), 0.5)
                self.rxarm.open_gripper()
                self.gotoSlow(np.array([[bigTower[0] - 10], [bigTower[1] - 6], [big_off + z_target_big]]), 0.5)
                print(big_off + z_target_big, "offset")
                self.goto(np.array([[bigTower[0] - 5], [bigTower[1] - 3], [450]]), 0.5)

            miss_big = False
            for block in self.camera.block_detections_big:
                if block[2] == colorDic[colorIdxBig] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    miss_big = True
                    break 
            if (not miss_big):

                colorIdxBig = colorIdxBig + 1
                if (colorIdxBig == 7):
                    colorIdxBig = 1
                i = i + 1
                
                    
            if (bigLine == 10):
                bigLine = 1

        
        
            # found = False
            # for block in self.camera.block_detections_small:
            #     if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
            #         found = True
            #         break
                    
            # if(found):

            #     print("small blocks")
            #     x = block[0][0][0]
            #     y = block[0][1][0]
            #     z = block[0][2][0]


            #     print(block, "block")
            #     self.rxarm.go_to_home_pose(1.2, 0.5, True)
            #     self.goto(np.array([[x], [y], [z + 150]]), 0.5)
            #     self.goto(np.array([[x], [y], [z + 20]]), 1.2)
            #     self.rxarm.close_gripper()
            #     self.goto(np.array([[x], [y], [z + 200]]), 0.5)
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [150]]), 1.48, 0.5)
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [50 + zOffsetSmall]]), 1.48, 0.5)
            #     self.rxarm.open_gripper()
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [200]]), 1.48, 0.5)
            #     smallLine = smallLine + 1


            # miss = False
            # for block in self.camera.block_detections_small:
            #     if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
            #         miss = True
            #         break 
            # if (not miss):

            #     colorIdxSmall = colorIdxSmall + 1
            #     if (colorIdxSmall == 7):
            #         colorIdxSmall = 1

            # if (smallLine == 10):
            #     smallLine = 1


            self.rxarm.go_to_home_pose(1.2, 0.5, True)
            self.rxarm.go_to_sleep_pose(1.2, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()

            complete = True
            for block in self.camera.block_detections_big:
                if block[0][1][0] < 20 or block[0][0][0] > 200:
                    complete = False
            # for block in self.camera.block_detections_small:
            #     if block[0][1][0] < 20 or block[0][0][0] > 200:
            #         complete = False  

            
            # for block in self.camera.block_detections_big:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         noBigBlocks = False
            #         break
            # for block in self.camera.block_detections_small:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         break
        complete = False
        z_target_small = 0
        small_off = 30
        miss_small = False
        i = 1
        first = True
        while(not complete):
            if (first):
                small_off = 30
                first = False
            else:
                small_off = 45


            z_temp_small = z_target_small
            if (i >= 4 and miss_small):
                z_target_small = z_temp_small + 42
            else:
                smallTower[0], smallTower[1], z_target_small = self.getTargetZSmall(smallTower)


            found = False
            for block in self.camera.block_detections_small:
                print(block, "for loop block")
                if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    found = True
                    print("found")
                    break
                    
            if(found):


                print("small blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                self.rxarm.go_to_home_pose(1.2, 0.5, True)
                self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                self.goto(np.array([[x], [y], [z + 30]]), 1.2)
                self.rxarm.close_gripper()
                self.goto(np.array([[x], [y], [z + 100 + z_target_small]]), 0.5)
                self.goto(np.array([[smallTower[0]], [smallTower[1]], [450]]), 0.5)
                print("going to place now at", smallTower[0], smallTower[1], small_off+z_target_small)
                self.gotoSlow(np.array([[smallTower[0]], [smallTower[1]], [small_off + z_target_small]]), 0.5)
                self.rxarm.open_gripper()
                #self.gotoSlow(np.array([[smallTower[0] - 30], [smallTower[1] - 18], [small_off + z_target_small]]), 0.5)
                print(small_off + z_target_small, "offset")
                self.goto(np.array([[smallTower[0] - 5], [smallTower[1] - 3], [450]]), 0.5)

            miss_small = False
            for block in self.camera.block_detections_big:
                if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
                    miss_small = True
                    break 
            if (not miss_small):

                colorIdxSmall = colorIdxSmall + 1
                if (colorIdxSmall == 7):
                    colorIdxSmall = 1
                i = i + 1
                
                    
            if (smallLine == 10):
                smallLine = 1

        
        
            # found = False
            # for block in self.camera.block_detections_small:
            #     if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
            #         found = True
            #         break
                    
            # if(found):

            #     print("small blocks")
            #     x = block[0][0][0]
            #     y = block[0][1][0]
            #     z = block[0][2][0]


            #     print(block, "block")
            #     self.rxarm.go_to_home_pose(1.2, 0.5, True)
            #     self.goto(np.array([[x], [y], [z + 150]]), 0.5)
            #     self.goto(np.array([[x], [y], [z + 20]]), 1.2)
            #     self.rxarm.close_gripper()
            #     self.goto(np.array([[x], [y], [z + 200]]), 0.5)
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [150]]), 1.48, 0.5)
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [50 + zOffsetSmall]]), 1.48, 0.5)
            #     self.rxarm.open_gripper()
            #     self.gotoRot(np.array([[placeDicSmall[colorIdxSmall][0]], [placeDicSmall[colorIdxSmall][1]], [200]]), 1.48, 0.5)
            #     smallLine = smallLine + 1


            # miss = False
            # for block in self.camera.block_detections_small:
            #     if block[2] == colorDic[colorIdxSmall] and (block[0][0][0] > 200 or block[0][1][0] < 20):
            #         miss = True
            #         break 
            # if (not miss):

            #     colorIdxSmall = colorIdxSmall + 1
            #     if (colorIdxSmall == 7):
            #         colorIdxSmall = 1

            # if (smallLine == 10):
            #     smallLine = 1


            self.rxarm.go_to_home_pose(1.2, 0.5, True)
            self.rxarm.go_to_sleep_pose(1.2, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()

            complete = True
            for block in self.camera.block_detections_big:
                if block[0][1][0] < 20 or block[0][0][0] > 200:
                    complete = False
            for block in self.camera.block_detections_small:
                if block[0][1][0] < 20 or block[0][0][0] > 200:
                    complete = False  

            
            # for block in self.camera.block_detections_big:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         noBigBlocks = False
            #         break
            # for block in self.camera.block_detections_small:
            #     if block[0][1][0] > 0:
            #         print(block, "bad block")
            #         complete = False
            #         break
        print("done event 4")


        self.next_state = "idle"

    def sky(self):
        """!
        @brief      
        """
        self.current_state = "sky"

        loc = [-250, 100]
        skyDistTh = 20
        radiusAroundTarget = 10
        ogLoc = True
        gridOffsetLeft = 30
        gridOffsetRight = 18


        for block in self.camera.block_detections_big:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            z_h = block[-1]
            if(np.linalg.norm(np.array([x, y]) - np.array(loc)) < skyDistTh):

                loc = [x, y]
                ogLoc = False

                break
        i = 1
        z_target = 0
        while(1):

            

            self.rxarm.go_to_sleep_pose(1.5, 0.5, False)
            rospy.sleep(2)
            
            self.camera.blockDetector()
            
            z_temp = z_target
            if (i >= 4 and grabbedBlock):
                z_target = z_temp + 42
            else:
                loc[0], loc[1], z_target = self.getTargetZ(loc)


            grabbedBlock = False
            print(loc, "location")
            print(z_target, "target")
            if (ogLoc or z_target == 0):
                off = 40
                ogLoc = False
            else:
                off = 60
            if (z_target > 100):
                off = 50
            for block in self.camera.block_detections_big:
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]
                if(np.linalg.norm(np.array([x, y]) - np.array(loc)) < skyDistTh and (np.linalg.norm(np.array([x, y]) - np.array(loc)) > radiusAroundTarget)):
                    self.rxarm.go_to_home_pose(1.8, 0.5, True)

                    if(x > -250):
                        gridOffset = gridOffsetRight
                    else:
                        gridOffset = gridOffsetLeft
                        print("using grid offset left")
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + gridOffset]]), 1.2)
                    self.rxarm.close_gripper()

                    # self.goto(np.array([[x], [y], [z + 100 + z_target]]), 0.5)
                    # self.goto(np.array([[loc[0]], [loc[1]], [150 + z_target]]), 0.5)
                    # self.goto(np.array([[loc[0]], [loc[1]], [off + z_target]]), 0.5)
                    # print(off + z_target, "offset")
                    # self.rxarm.open_gripper()
                    # self.goto(np.array([[loc[0]], [loc[1]], [200 + z_target]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 100 + z_target]]), 0.5)
                    self.goto(np.array([[loc[0]], [loc[1]], [450]]), 0.5)
                    print("going to place now at", loc[0], loc[1], off+z_target)
                    self.gotoSlow(np.array([[loc[0]], [loc[1]], [off + z_target]]), 0.5)
                    self.rxarm.open_gripper()
                    self.gotoSlow(np.array([[loc[0] + 10], [loc[1] - 6], [off + z_target]]), 0.5)
                    print(off + z_target, "offset")
                    self.goto(np.array([[loc[0] + 5], [loc[1] - 3], [450]]), 0.5)

                    self.rxarm.go_to_home_pose(1.8, 0.5, True)
                    i = i + 1
                    radiusAroundTarget = radiusAroundTarget + 10
                    grabbedBlock = True
                    break
            if (not grabbedBlock):
                skyDistTh = skyDistTh + 30
                print(skyDistTh, "skyDistTh")


        self.next_state = "idle"


    def getTargetZ(self, loc):
        skyDistTh = 20
        x_default = loc[0]
        y_default = loc[1]
        z_default = 0
 

        for block in self.camera.block_detections_big:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            z_h = block[-1]
            if(np.linalg.norm(np.array([x, y]) - np.array(loc)) < skyDistTh):
                return x, y, z_h
        return x_default, y_default, z_default

    def getTargetZSmall(self, loc):
        skyDistTh = 20
        x_default = loc[0]
        y_default = loc[1]
        z_default = 0
 

        for block in self.camera.block_detections_small:
            x = block[0][0][0]
            y = block[0][1][0]
            z = block[0][2][0]
            z_h = block[-1]
            if(np.linalg.norm(np.array([x, y]) - np.array(loc)) < skyDistTh):
                return x, y, z_h
        return x_default, y_default, z_default

    def unstack2(self):
        self.current_state = "unstack"
        self.calibrate()
        height_dic = {0: 910, 1: 935, 2: 960, 3: 975}
        ######################################################################
        #### for big blocks, position 1 and 6 are too close to each other ####
        ######################################################################
        bigDic = {
            10: [-115, 60], 11: [-185, 60], 12: [-275, 60], 9: [-350, 60], 8: [-115, -5], 7: [-185, -5],
            6: [-275, -5], 5: [-350, -5], 4: [-115, -105],
            3: [-185, -105], 2: [-275, -105], 1: [-350, -105]
        }
        # bigDic = {
        #     9: [150, 195], 8: [220, 195], 7: [290, 195],
        #     6: [360, 195], 5: [110, -75 + 200], 4: [180, -75 + 200],
        #     3: [250, -75 + 200], 2: [320, -75 + 200], 1: [380, -75 + 200]
        # }
        # # bigDic = {
        # #     9: [150, -15], 8: [220, -15], 7: [290, -15],
        # #     6: [360, -15], 5: [110, -65], 4: [180, -65],
        # #     3: [250, -65], 2: [320, -65], 1: [380, -65]
        # # }
        # smallDic = {
        #     9: [-130, -15 +200], 8: [-210, -15 +200], 7: [-290, -15 +200],
        #     6: [-130, -65 + 200], 1: [-380, -65 + 200], 2: [-330, -65 + 200],
        #     3: [-280, -65 + 200], 4: [-230, -65 + 200], 5: [-180, -65 + 200]
        # }

        gridOffsetRight = 45
        gridOffsetLeft = 28

        complete = False

        zOffsetBig = 0
        offf = 1

        zOffsetSmall = 0

        depth_dix = 0
        while(not complete):
            for block in self.camera.block_detections_big:
                print("big blocks")
                print(block, "block")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]

                if (y > 95 or x > -40):
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 20]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [150]]), 0.5)
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [50 + zOffsetBig]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [200]]), 0.5)
                    offf = offf + 1
                    moved = True
                    if (offf > 12):
                        offf = 1
                        zOffsetBig = zOffsetBig + 50

            for block in self.camera.block_detections_small:
                print("small blocks")
                x = block[0][0][0]
                y = block[0][1][0]
                z = block[0][2][0]


                print(block, "block")
                if (y > 95 or x > -40):
                    self.rxarm.go_to_home_pose(1.2, 0.5, True)
                    if(x > 50):
                        offset = gridOffsetRight
                    else:
                        offset = gridOffsetLeft
                    self.goto(np.array([[x], [y], [z + 150]]), 0.5)
                    self.goto(np.array([[x], [y], [z + 30]]), 1.2)
                    self.rxarm.close_gripper()
                    self.goto(np.array([[x], [y], [z + 200]]), 0.5)
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [150]]), 0.5)
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [50 + zOffsetSmall]]), 0.5)
                    self.rxarm.open_gripper()
                    self.goto(np.array([[bigDic[offf][0]], [bigDic[offf][1]], [200]]), 0.5)
                    offf = offf + 1
                    moved = True
                    if (offf > 12):
                        offf = 1
                        zOffsetSmall = zOffsetSmall + 25
                #depthVal = depthVal + 20
                        
            complete = True
            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            self.rxarm.go_to_sleep_pose(2.0, 0.5, False)
            rospy.sleep(2)
            if(depth_dix > 3):
                depth_dix = 3
            
            self.camera.blockDetector(height_dic[depth_dix])
            print(height_dic[depth_dix])

            self.rxarm.go_to_home_pose(2.0, 0.5, True)
            
            if(depth_dix != 3):
                complete = False
            for block in self.camera.block_detections_big:
                if block[0][1][0] > 95 or block[0][0][0] > -40:
                    print(block, "bad block")
                    complete = False
                    break
            for block in self.camera.block_detections_small:
                if block[0][1][0] > 95 or block[0][0][0] > -40:
                    print(block, "bad block")
                    complete = False
                    break
                
            depth_dix += 1
                    
        print("done unstack")
        self.next_state = "idle"
        



    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.next_state = "idle"

    def teach(self):
        self.current_state = "teach"
        cur_position = self.rxarm.get_positions()

        #print("cur_position", cur_position)
        # self.teach_trajectory = self.teach_trajectory.append(cur_position)
        self.teach_trajectory.append(cur_position)
        #print("teach traj: ", self.teach_trajectory)
        self.next_state = "idle"
        # pass

    def repeat(self):
        self.theta1 = []
        self.theta2 = []
        self.theta3 = []
        self.theta4 = []
        self.theta5 = []
        self.timeList = []
        self.current_state = "repeat"
        if self.teach_trajectory != []:
            print("teach traj: ", self.teach_trajectory)
            self.status_message = "State: Repeat - Repeating motion plan"

            close_gripper = True

            for i in range(len(self.teach_trajectory)):
                self.rxarm.set_joint_positions(self.teach_trajectory[i], 2.0, 0.5, True)

                if (i-1) % 3 == 0:
                    if close_gripper == True:
                        self.rxarm.close_gripper()
                        close_gripper = not close_gripper
                    else:
                        self.rxarm.open_gripper()
                        close_gripper = not close_gripper

                rospy.sleep(1)

        self.next_state = "idle"  


    def clear(self):
        self.current_state = "clear"
        self.teach_trajectory = []
        self.next_state = "idle"

    def pickAndPlace(self):
        self.current_state = "pick&place"
        p_world = self.findClickPos()
        


        r = np.sqrt(p_world[0][0]*p_world[0][0] + p_world[1][0]*p_world[1][0])

        if (r <= 410):
            wrist_angle = 1.4
        else:
            wrist_angle = 0.1


        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 150.0, 0, wrist_angle, 0]
        print("going to ", position)
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)
        rospy.sleep(0.5)

        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 25.0, 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)
        self.rxarm.close_gripper()
        rospy.sleep(0.5)

        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 200.0, 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)
        rospy.sleep(0.5)

        p_world = self.findClickPos()
        r = np.sqrt(p_world[0][0]*p_world[0][0] + p_world[1][0]*p_world[1][0])

        if (r <= 410):
            wrist_angle = 1.4
        else:
            wrist_angle = 0.1


        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 150.0, 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)
        rospy.sleep(0.5)

        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 65.0, 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)
        self.rxarm.open_gripper()
        rospy.sleep(0.5)

        position = [p_world[0][0], p_world[1][0], p_world[2][0] + 200.0, 0, 0.9, 0]
        ik_angles = IK_geometric(position)[1]
        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)

        angles = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.rxarm.set_joint_positions(angles, 2.0, 0.5, True)


        self.next_state = "idle"

    def findClickPos(self):
        if self.camera.new_click:
            self.camera.new_click = False
        while(not self.camera.new_click):
            pass

        self.camera.new_click = False
        pixX = self.camera.last_click[0]
        pixY = self.camera.last_click[1]
        z = self.camera.DepthFrameRaw[pixY][pixX]
 

        self.camera.extrinsic_matrix = self.extrinsic_matrix


        intrinsic = np.array(self.camera.intrinsic_matrix)
        extrinsic = np.array(self.camera.extrinsic_matrix)


        p_pixel = np.array([[pixX, pixY, 1]])
        p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
        p_camera = np.vstack((p_camera, [[1]]))
        p_world = np.linalg.inv(extrinsic).dot(p_camera)


        return p_world

    def goto(self, p_world, sleep):

        # if ((p_world[0][0] > 255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        # if ((p_world[0][0] < -255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        r = np.sqrt(p_world[0][0]*p_world[0][0] + p_world[1][0]*p_world[1][0])

        if (r <= 410):
            wrist_angle = 1.4
        else:
            wrist_angle = 0.1

        if (p_world[2][0] > 250):
            wrist_angle = 0.05
            

        position = [p_world[0][0], p_world[1][0], p_world[2][0], 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]

        # if(abs(ik_angles[0]) > 1.57):
        #     print("3")
        #     ik_angles = IK_geometric(position)[3]

        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 1.5, 0.5, True)
        # self.rxarm.set_joint_positions(angles, 1.2, 0.5, True)
        rospy.sleep(sleep)

        
    def gotoSlow(self, p_world, sleep):

        # if ((p_world[0][0] > 255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        # if ((p_world[0][0] < -255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        r = np.sqrt(p_world[0][0]*p_world[0][0] + p_world[1][0]*p_world[1][0])

        if (r <= 410):
            wrist_angle = 1.4
        else:
            wrist_angle = 0.1

        if (p_world[2][0] > 150):
            wrist_angle = 0.05

        position = [p_world[0][0], p_world[1][0], p_world[2][0], 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]

        # if(abs(ik_angles[0]) > 1.57):
        #     print("3")
        #     ik_angles = IK_geometric(position)[3]

        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 2.5, 0.5, True)
        rospy.sleep(sleep)

    def gotoRot(self, p_world, ang, sleep):

        # if ((p_world[0][0] > 255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        # if ((p_world[0][0] < -255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4
            

        r = np.sqrt(p_world[0][0]*p_world[0][0] + p_world[1][0]*p_world[1][0])

        if (r <= 410):
            wrist_angle = 1.4
        else:
            wrist_angle = 0.1


        position = [p_world[0][0], p_world[1][0], p_world[2][0], 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]

        # if(abs(ik_angles[0]) > 1.57):
        #     print("3")
        #     ik_angles = IK_geometric(position)[3]

        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ang]
        self.rxarm.set_joint_positions(angles, 1.2, 0.5, True)
        rospy.sleep(sleep)

    def gotoDown(self, p_world, sleep):

        # if ((p_world[0][0] > 255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        # if ((p_world[0][0] < -255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4


        wrist_angle = 1.4


        position = [p_world[0][0], p_world[1][0], p_world[2][0], 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]

        # if(abs(ik_angles[0]) > 1.57):
        #     print("3")
        #     ik_angles = IK_geometric(position)[3]

        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 1.2, 0.5, True)
        rospy.sleep(sleep)

    def gotoSide(self, p_world, sleep):
        # if ((p_world[0][0] > 255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4

        # if ((p_world[0][0] < -255) and (p_world[1][0] > 275)):
        #     p_world[0][0] = p_world[0][0] + 8
        #     p_world[1][0] = p_world[1][0] - 4


        wrist_angle = 0.05


        position = [p_world[0][0], p_world[1][0], p_world[2][0], 0, wrist_angle, 0]
        ik_angles = IK_geometric(position)[1]

        # if(abs(ik_angles[0]) > 1.57):
        #     print("3")
        #     ik_angles = IK_geometric(position)[3]

        angles = [ik_angles[0], ik_angles[1], ik_angles[2], ik_angles[3], ik_angles[4]]
        self.rxarm.set_joint_positions(angles, 1.2, 0.5, True)
        rospy.sleep(sleep)


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)