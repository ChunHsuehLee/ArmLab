"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""

import numpy as np
import cv2
# expm is a matrix exponential function
from scipy.linalg import expm


def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle


def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    pass


def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """
    pass


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    
    """
    theta = np.arctan2(np.sqrt(1 - (T[2][2]*T[2][2])), T[2][2])
    if np.sin(theta) > 0:
        phi= np.arctan2(T[1][2], T[0][2])
        psi = np.arctan2(T[2][1], -T[2][0])
    else:
        phi = np.arctan2(-T[1][2], -T[0][2])
        psi = np.arctan2(-T[2][1], T[2][0])       


    return [phi, theta, psi]


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """


    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """


    matExps = []

    for i in range(len(s_lst)):

        angle = np.array([joint_angles[i]])

        w = np.array([(s_lst[i])[3], (s_lst[i])[4], (s_lst[i])[5]])
        rot, _ = cv2.Rodrigues(w*angle, np.zeros((3, 3)))
        

        v = np.array([(s_lst[i])[0], (s_lst[i])[1], (s_lst[i])[2]])
        P1 = np.matmul((np.identity(3) - rot), np.transpose((np.cross(np.transpose(w), np.transpose(v)))))
        P2 = angle*np.matmul(w, np.matmul(np.transpose(w), v))
        P = P1 + P2


        matExps.append(np.array([[rot[0][0], rot[0][1], rot[0][2], P[0]], [rot[1][0], rot[1][1], rot[1][2], P[1]], 
                                [rot[2][0], rot[2][1], rot[2][2], P[2]], [0, 0, 0, 1]]))



    Gst = np.matmul(matExps[len(matExps)-1], m_mat)
    for i in range(len(matExps)-1):
        Gst = np.matmul(matExps[len(matExps)-2-i], Gst)


    euler_angles = get_euler_angles_from_T(Gst)
    pose = (Gst[0][3], Gst[1][3], Gst[2][3], euler_angles[0], euler_angles[1], euler_angles[2])
    #print(pose)


    #print((IK_geometric(pose)*180)/(np.pi))

    return pose


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass


def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """   

    phi = pose[3]
    theta = pose[4]
    psi = pose[5]

    r11 = np.cos(phi)*np.cos(theta)*np.cos(psi) - np.sin(phi)*np.sin(psi)
    r12 = -np.cos(phi)*np.cos(theta)*np.sin(psi) - np.sin(phi)*np.cos(psi)
    r13 = np.cos(phi)*np.sin(theta)
    r21 = np.sin(phi)*np.cos(theta)*np.cos(psi) + np.cos(phi)*np.sin(psi)
    r22 = -np.sin(phi)*np.cos(theta)*np.sin(psi) + np.cos(phi)*np.cos(psi)
    r23 = np.sin(phi)*np.sin(theta)
    r31 = -np.sin(theta)*np.cos(psi)
    r32 = np.sin(theta)*np.sin(psi)
    r33 = np.cos(theta)

    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    # if (pose[0] > 0):
    #     pose[0] = pose[0] + 5
    #     pose[1] = pose[1] + 5
    # else:
    #     pose[0] = pose[0] - 5
    #     pose[1] = pose[1] + 5





    L0 = 68.66
    L1 = 35.25
    L2 = 200
    L3 = 50
    L4 = 200
    L5 = 65
    L6 = (109.15 - 15)

    awk_ang = np.arctan2(L3, L2)


    #theta1_1 = -np.arctan2(xc, yc)
    theta1_1 = -np.arctan2(pose[0], pose[1])
    theta1_2 = theta1_1 + np.pi

    R13 = np.cos(theta)*np.sin(theta1_1)
    R23 = -np.sin(theta)*np.cos(theta1_1)
    R33 = np.cos(theta)

    # xc = pose[0] - (L5+L6)*R13
    # yc = pose[1] - (L5+L6)*R23
    # zc = pose[2] - (L5+L6)*R33
    xc = pose[0] + (L5+L6)*np.cos(theta)*np.sin(theta1_1)
    yc = pose[1] - (L5+L6)*np.cos(theta)*np.cos(theta1_1)
    zc = pose[2] + (L5+L6)*np.sin(theta)




    r = np.sqrt(xc*xc + yc*yc)
    d1 = L0 + L1
    s = zc - d1

    link1 = 205.73
    link2 = L4

    offset = ((np.pi)/2 - awk_ang)

    input = ((r*r + s*s) - link1*link1 - link2*link2)/(2*link1*link2)
    if (input <= -0.999):
        input = -0.999
    elif (input >= 0.999):
        input = 0.999

    theta3_elbowDown = np.arccos(input)
    theta3_elbowUp = -theta3_elbowDown

    alpha = np.arctan2(s, r)
    betaDown = np.arctan2(link2*np.sin(theta3_elbowDown), link1 + link2*np.cos(theta3_elbowDown))
    betaUp = np.arctan2(link2*np.sin(theta3_elbowUp), (link1 + link2*np.cos(theta3_elbowUp)))

    theta2_elbowDown = -(alpha - betaDown)
    theta2_elbowUp = (alpha - betaUp)





    # R03_base1_elbowDown = np.array([[np.cos(theta1_1)*np.cos(theta2_elbowDown + theta3_elbowDown), -np.cos(theta1_1)*np.sin(theta2_elbowDown + theta3_elbowDown), np.sin(theta1_1)], 
    #                                 [np.sin(theta1_1)*np.cos(theta2_elbowDown + theta3_elbowDown), -np.sin(theta1_1)*np.sin(theta2_elbowDown + theta3_elbowDown), -np.cos(theta1_1)], 
    #                                 [np.sin(theta2_elbowDown + theta3_elbowDown), np.cos(theta2_elbowDown + theta3_elbowDown), 0]])

    # R03_base1_elbowUp = np.array([[np.cos(theta1_1)*np.cos(theta2_elbowUp + theta3_elbowUp), -np.cos(theta1_1)*np.sin(theta2_elbowUp + theta3_elbowUp), np.sin(theta1_1)], 
    #                                 [np.sin(theta1_1)*np.cos(theta2_elbowUp + theta3_elbowUp), -np.sin(theta1_1)*np.sin(theta2_elbowUp + theta3_elbowUp), -np.cos(theta1_1)], 
    #                                 [np.sin(theta2_elbowUp + theta3_elbowUp), np.cos(theta2_elbowUp + theta3_elbowUp), 0]])

    # R03_base2_elbowDown = np.array([[np.cos(theta1_2)*np.cos(theta2_elbowDown + theta3_elbowDown), -np.cos(theta1_2)*np.sin(theta2_elbowDown + theta3_elbowDown), np.sin(theta1_2)], 
    #                                 [np.sin(theta1_2)*np.cos(theta2_elbowDown + theta3_elbowDown), -np.sin(theta1_2)*np.sin(theta2_elbowDown + theta3_elbowDown), -np.cos(theta1_2)], 
    #                                 [np.sin(theta2_elbowDown + theta3_elbowDown), np.cos(theta2_elbowDown + theta3_elbowDown), 0]])

    # R03_base2_elbowUp = np.array([[np.cos(theta1_2)*np.cos(theta2_elbowUp + theta3_elbowUp), -np.cos(theta1_2)*np.sin(theta2_elbowUp + theta3_elbowUp), np.sin(theta1_2)], 
    #                                 [np.sin(theta1_2)*np.cos(theta2_elbowUp + theta3_elbowUp), -np.sin(theta1_2)*np.sin(theta2_elbowUp + theta3_elbowUp), -np.cos(theta1_2)], 
    #                                 [np.sin(theta2_elbowUp + theta3_elbowUp), np.cos(theta2_elbowUp + theta3_elbowUp), 0]])

    
    # R35_base1_elbowDown = np.matmul(np.transpose(R03_base1_elbowDown), R)
    # R35_base1_elbowUp = np.matmul(np.transpose(R03_base1_elbowUp), R)
    # R35_base2_elbowDown = np.matmul(np.transpose(R03_base2_elbowDown), R)
    # R35_base2_elbowUp = np.matmul(np.transpose(R03_base2_elbowUp), R)
    

    # theta4_base1_elbowDown = np.arctan2(R35_base1_elbowDown[2][1], R35_base1_elbowDown[1][1])
    # theta4_base1_elbowUp = np.arctan2(R35_base1_elbowUp[2][1], R35_base1_elbowUp[1][1])
    # theta4_base2_elbowDown = np.arctan2(R35_base2_elbowDown[2][1], R35_base2_elbowDown[1][1])
    # theta4_base2_elbowUp = np.arctan2(R35_base2_elbowUp[2][1], R35_base2_elbowUp[1][1])

    # theta5_base1_elbowDown = np.arctan2(R35_base1_elbowDown[0][2], R35_base1_elbowDown[0][0])
    # theta5_base1_elbowUp = np.arctan2(R35_base1_elbowUp[0][2], R35_base1_elbowUp[0][0])
    # theta5_base2_elbowDown = np.arctan2(R35_base2_elbowDown[0][2], R35_base2_elbowDown[0][0])
    # theta5_base2_elbowUp = np.arctan2(R35_base2_elbowUp[0][2], R35_base2_elbowUp[0][0])

    theta4_elbowDown = -theta - theta3_elbowDown - theta2_elbowDown
    theta4_elbowUp = -theta - theta3_elbowUp - theta2_elbowUp

    


    config1_elbowDown = np.array([theta1_1, theta2_elbowDown + offset, theta3_elbowDown + offset, theta4_elbowDown, 0])
    config1_elbowUp = np.array([theta1_1, -theta2_elbowUp + offset, theta3_elbowUp + offset, theta4_elbowUp, 0])
    config2_elbowDown = np.array([theta1_2, theta2_elbowDown + offset, theta3_elbowDown + offset, theta4_elbowDown, 0])
    config2_elbowUp = np.array([theta1_2, theta2_elbowUp + offset, theta3_elbowUp + offset, theta4_elbowUp, 0])

    configs = np.array([config1_elbowDown, config1_elbowUp, config2_elbowDown, config2_elbowUp])

    return configs