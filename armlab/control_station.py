#!/usr/bin/python
"""!
Main GUI for Arm lab
"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))

import argparse
import sys
import cv2
import numpy as np
import rospy
import time
from functools import partial
import time

from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
from PyQt4.QtGui import (QPixmap, QImage, QApplication, QWidget, QLabel,
                         QMainWindow, QCursor, QFileDialog)

from ui import Ui_MainWindow
from rxarm import RXArm, RXArmThread
from camera import Camera, VideoThread
from state_machine import StateMachine, StateMachineThread
""" Radians to/from  Degrees conversions """
D2R = np.pi / 180.0
R2D = 180.0 / np.pi



class Gui(QMainWindow):
    """!
    Main GUI Class

    Contains the main function and interfaces between the GUI and functions.
    """

    def __init__(self, parent=None, dh_config_file=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        """ Groups of ui commonents """
        self.joint_readouts = [
            self.ui.rdoutBaseJC,
            self.ui.rdoutShoulderJC,
            self.ui.rdoutElbowJC,
            self.ui.rdoutWristAJC,
            self.ui.rdoutWristRJC,
        ]
        self.joint_slider_rdouts = [
            self.ui.rdoutBase,
            self.ui.rdoutShoulder,
            self.ui.rdoutElbow,
            self.ui.rdoutWristA,
            self.ui.rdoutWristR,
        ]
        self.joint_sliders = [
            self.ui.sldrBase,
            self.ui.sldrShoulder,
            self.ui.sldrElbow,
            self.ui.sldrWristA,
            self.ui.sldrWristR,
        ]
        """Objects Using Other Classes"""
        self.camera = Camera()

        print("Creating rx arm...")
        if (dh_config_file is not None):
            self.rxarm = RXArm(dh_config_file=dh_config_file)
        else:
            self.rxarm = RXArm()
        print("Done creating rx arm instance.")
        self.sm = StateMachine(self.rxarm, self.camera)
        """
        Attach Functions to Buttons & Sliders
        TODO: NAME AND CONNECT BUTTONS AS NEEDED
        """
        # Video
        self.ui.videoDisplay.setMouseTracking(True)
        self.ui.videoDisplay.mouseMoveEvent = self.trackMouse
        self.ui.videoDisplay.mousePressEvent = self.calibrateMousePress

        # Buttons
        # Handy lambda function falsethat can be used with Partial to only set the new state if the rxarm is initialized
        #nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state if self.rxarm.initialized else None)
        nxt_if_arm_init = lambda next_state: self.sm.set_next_state(next_state)
        self.ui.btn_estop.clicked.connect(self.estop)
        self.ui.btn_init_arm.clicked.connect(self.initRxarm)
        self.ui.btn_torq_off.clicked.connect(
            lambda: self.rxarm.disable_torque())
        self.ui.btn_torq_on.clicked.connect(lambda: self.rxarm.enable_torque())
        self.ui.btn_sleep_arm.clicked.connect(lambda: self.rxarm.sleep())
        self.ui.btn_task1.clicked.connect(partial(nxt_if_arm_init, 'pickNsort'))
        self.ui.btn_task2.clicked.connect(partial(nxt_if_arm_init, 'pickNstack'))
        self.ui.btn_task3.clicked.connect(partial(nxt_if_arm_init, 'lineMup'))
        self.ui.btn_task4.clicked.connect(partial(nxt_if_arm_init, 'stackMhigh'))
        self.ui.btn_task5.clicked.connect(partial(nxt_if_arm_init, 'sky'))

        #User Buttons
        self.ui.btnUser1.setText("Calibrate")
        self.ui.btnUser1.clicked.connect(partial(nxt_if_arm_init, 'calibrate'))
        self.ui.btnUser2.setText('Open Gripper')
        self.ui.btnUser2.clicked.connect(lambda: self.rxarm.open_gripper())
        self.ui.btnUser3.setText('Close Gripper')
        self.ui.btnUser3.clicked.connect(lambda: self.rxarm.close_gripper())
        self.ui.btnUser4.setText('Execute')
        self.ui.btnUser4.clicked.connect(partial(nxt_if_arm_init, 'execute'))
        self.ui.btnUser5.setText('Teach')
        self.ui.btnUser5.clicked.connect(partial(nxt_if_arm_init, 'teach'))
        self.ui.btnUser6.setText('Repeat')
        self.ui.btnUser6.clicked.connect(partial(nxt_if_arm_init, 'repeat'))
        self.ui.btnUser7.setText('Clear')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'clear'))
        self.ui.btnUser7.setText('Pick and Place')
        self.ui.btnUser7.clicked.connect(partial(nxt_if_arm_init, 'pick&place'))
        self.ui.btnUser8.setText('Detect')
        self.ui.btnUser8.clicked.connect(partial(nxt_if_arm_init, 'detect'))
        self.ui.btnUser9.setText('Unstack')
        self.ui.btnUser9.clicked.connect(partial(nxt_if_arm_init, 'unstack'))
        self.ui.btnUser10.setText('Stack')
        self.ui.btnUser10.clicked.connect(partial(nxt_if_arm_init, 'stack'))
        self.ui.btnUser11.setText('Get Data')
        self.ui.btnUser11.clicked.connect(partial(nxt_if_arm_init, 'get_data'))


        # Sliders
        for sldr in self.joint_sliders:
            sldr.valueChanged.connect(self.sliderChange)
        self.ui.sldrMoveTime.valueChanged.connect(self.sliderChange)
        self.ui.sldrAccelTime.valueChanged.connect(self.sliderChange)
        # Direct Control
        self.ui.chk_directcontrol.stateChanged.connect(self.directControlChk)
        # Status
        self.ui.rdoutStatus.setText("Waiting for input")
        """initalize manual control off"""
        self.ui.SliderFrame.setEnabled(False)
        """Setup Threads"""

        # State machine
        self.StateMachineThread = StateMachineThread(self.sm)
        self.StateMachineThread.updateStatusMessage.connect(
            self.updateStatusMessage)
        self.StateMachineThread.start()
        self.VideoThread = VideoThread(self.camera)
        self.VideoThread.updateFrame.connect(self.setImage)
        self.VideoThread.start()
        self.ArmThread = RXArmThread(self.rxarm)
        self.ArmThread.updateJointReadout.connect(self.updateJointReadout)
        self.ArmThread.updateEndEffectorReadout.connect(
            self.updateEndEffectorReadout)
        self.ArmThread.start()

    """ Slots attach callback functions to signals emitted from threads"""

    @pyqtSlot(str)
    def updateStatusMessage(self, msg):
        self.ui.rdoutStatus.setText(msg)

    @pyqtSlot(list)
    def updateJointReadout(self, joints):
        if (self.sm.First):
            self.sm.First = False
            self.sm.starttime = time.time()
            self.sm.currenttime = self.sm.starttime - self.sm.starttime


        else:
            self.sm.currenttime = time.time() - self.sm.starttime


        self.sm.timeList.append(self.sm.currenttime)
        #print(self.sm.j)
        self.sm.j = self.sm.j + 1
        i = 1
        for rdout, joint in zip(self.joint_readouts, joints):
            rdout.setText(str('%+.2f' % (joint * R2D)))
            # if i == 1 and self.sm.j%10 == 0:
            #     self.sm.theta1.append(joint * R2D)
            # if i == 2 and self.sm.j%10 == 0:
            #     self.sm.theta2.append(joint * R2D)

            # if i == 3 and self.sm.j%10 == 0:
            #     self.sm.theta3.append(joint * R2D)
            # if i == 4 and self.sm.j%10 == 0:
            #     self.sm.theta4.append(joint * R2D)

            # if i == 5 and self.sm.j%10 == 0:
            #     self.sm.theta5.append(joint * R2D)

            # i = i + 1





    ### TODO: output the rest of the orientation according to the convention chosen
    @pyqtSlot(list)
    def updateEndEffectorReadout(self, pos):
        self.ui.rdoutX.setText(str("%+.2f mm" % (pos[0])))
        self.ui.rdoutY.setText(str("%+.2f mm" % (pos[1])))
        self.ui.rdoutZ.setText(str("%+.2f mm" % (pos[2])))
        self.ui.rdoutPhi.setText(str("%+.2f rad" % (pos[3])))
        self.ui.rdoutTheta.setText(str("%+.2f rad" % (pos[4])))
        self.ui.rdoutPsi.setText(str("%+.2f rad" % (pos[5])))

    @pyqtSlot(QImage, QImage, QImage, QImage)
    def setImage(self, rgb_image, depth_image, tag_image, cnt_image):
        """!
        @brief      Display the images from the camera.

        @param      rgb_image    The rgb image
        @param      depth_image  The depth image
        """
        if (self.ui.radioVideo.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(rgb_image))
        if (self.ui.radioDepth.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(depth_image))
        if (self.ui.radioUsr1.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(tag_image))
        if (self.ui.radioUsr2.isChecked()):
            self.ui.videoDisplay.setPixmap(QPixmap.fromImage(cnt_image))

    """ Other callback functions attached to GUI elements"""

    def estop(self):
        self.rxarm.disable_torque()
        self.sm.set_next_state('estop')

    def sliderChange(self):
        """!
        @brief Slider changed

        Function to change the slider labels when sliders are moved and to command the arm to the given position
        """
        for rdout, sldr in zip(self.joint_slider_rdouts, self.joint_sliders):
            rdout.setText(str(sldr.value()))

        self.ui.rdoutMoveTime.setText(
            str(self.ui.sldrMoveTime.value() / 10.0) + "s")
        self.ui.rdoutAccelTime.setText(
            str(self.ui.sldrAccelTime.value() / 20.0) + "s")
        self.rxarm.set_moving_time(self.ui.sldrMoveTime.value() / 10.0)
        self.rxarm.set_accel_time(self.ui.sldrAccelTime.value() / 20.0)

        # Do nothing if the rxarm is not initialized
        if self.rxarm.initialized:
            joint_positions = np.array(
                [sldr.value() * D2R for sldr in self.joint_sliders])
            # Only send the joints that the rxarm has
            self.rxarm.set_positions(joint_positions[0:self.rxarm.num_joints])

    def directControlChk(self, state):
        """!
        @brief      Changes to direct control mode

                    Will only work if the rxarm is initialized.

        @param      state  State of the checkbox
        """
        if state == Qt.Checked and self.rxarm.initialized:
            # Go to manual and enable sliders
            self.sm.set_next_state("manual")
            self.ui.SliderFrame.setEnabled(True)
        else:
            # Lock sliders and go to idle
            self.sm.set_next_state("idle")
            self.ui.SliderFrame.setEnabled(False)
            self.ui.chk_directcontrol.setChecked(False)

    def trackMouse(self, mouse_event):
        """!
        @brief      Show the mouse position in GUI

                    TODO: after implementing workspace calibration display the world coordinates the mouse points to in the RGB
                    video image.

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """

        pt = mouse_event.pos()
        if self.camera.DepthFrameRaw.any() != 0:
            z = self.camera.DepthFrameRaw[pt.y()][pt.x()] + self.camera.virtual_plane_depth - self.camera.DepthEmptyFrame[pt.y()][pt.x()]
            self.ui.rdoutMousePixels.setText("(%.0f,%.0f,%.0f)" %
                                             (pt.x(), pt.y(), z))

            # self.camera.intrinsic_matrix = [[922.357834, 0, 649.483591], [0, 921.135476, 338.516031], [0, 0, 1]] # 3x3

            # self.camera.extrinsic_matrix = [[1, 0, 0, 0], [0, -1, 0, 190.5], [0, 0, -1, 980], [0, 0, 0, 1]]
            self.camera.extrinsic_matrix = self.sm.extrinsic_matrix
            # print("intrinsic: ", self.camera.intrinsic_matrix)
            p_camera = [[pt.x(), pt.y(), 1]]
            p_camera = np.transpose(p_camera)
            intrinsic = np.array(self.camera.intrinsic_matrix)
            extrinsic = np.array(self.camera.extrinsic_matrix)


            p_pixel = np.array([[pt.x(), pt.y(), 1]])
            p_camera = np.linalg.inv(intrinsic).dot(z*np.transpose(p_pixel))
            p_camera = np.vstack((p_camera, [[1]]))

            # depth correction
            # print("real: ", p_camera)
            # print("empty: ", self.camera.DepthEmptyFrame[int(p_camera[1])][int(p_camera[0])])
            # p_camera[2] = 980 + (p_camera[2] - self.camera.DepthEmptyFrame[int(p_camera[0])][int(p_camera[1])])
            p_world = np.linalg.inv(extrinsic).dot(p_camera)
            # print("origin: ", p_world[2])
            p_world[2] = p_world[2] + self.camera.virtual_plane_depth - self.camera.DepthEmptyFrame[pt.y()][pt.x()]
            # print("plane depth", self.camera.virtual_plane_depth)
            # print("offset", self.camera.DepthEmptyFrame[pt.y()][pt.x()])

            self.ui.rdoutMouseWorld.setText("(%.0f,%.0f,%.0f)" %
                                             (p_world[0], p_world[1], p_world[2]))
  

    def calibrateMousePress(self, mouse_event):
        """!
        @brief Record mouse click positions for calibration

        @param      mouse_event  QtMouseEvent containing the pose of the mouse at the time of the event not current time
        """
        """ Get mouse posiiton """
        pt = mouse_event.pos()
        
        self.camera.last_click[0] = pt.x()
        self.camera.last_click[1] = pt.y()
        self.camera.new_click = True
        # print(self.camera.last_click)

    def initRxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.ui.SliderFrame.setEnabled(False)
        self.ui.chk_directcontrol.setChecked(False)
        self.rxarm.enable_torque()
        self.sm.set_next_state('initialize_rxarm')


### TODO: Add ability to parse POX config file as well
def main(args=None):
    """!
    @brief      Starts the GUI
    """
    app = QApplication(sys.argv)
    app_window = Gui(dh_config_file=args['dhconfig'])
    app_window.show()
    sys.exit(app.exec_())


# Run main if this file is being run directly
### TODO: Add ability to parse POX config file as well
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--dhconfig",
                    required=False,
                    help="path to DH parameters csv file")
    main(args=vars(ap.parse_args()))
