"""
    @brief: This file contains the UR5 class, which is used to control the UR5 robot in CoppeliaSim
    @version: v1.0
    @date: 2023/05/25
    @author: Allan Souza Almeida

    @how_to_use:
        1. Import the class: from ur5 import UR5
        2. Create an instance of the class: ur5 = UR5()
        3. Use the functions to control the robot: ur5.move_to_config([0, 0, 0, 0, 0, 0])
        4. Use the functions to control the gripper: ur5.open_gripper()

"""
import time

import cv2
from coppeliasim_zmqremoteapi_client import *
import math
import numpy as np
import json
from math import sin, cos
from functools import reduce
from matplotlib import pyplot as plt
from keras.models import load_model

PI = math.pi


def limit_angle(angle):
    """
        Limits the angle between -pi and pi

        Parameters:
            angle (float): angle in radians

        Returns:
            angle (float): angle in radians between -pi and pi
    """
    angle_mod = angle % (2*np.pi)
    if angle_mod > np.pi:
        return angle_mod - 2*np.pi
    else:
        return angle_mod


def rot_z(theta):
    """
        Returns the rotation matrix around the z axis

        Parameters:
            theta (float): angle in radians

        Returns:
            R (np.array): rotation matrix
    """
    return np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                     [np.sin(theta), np.cos(theta), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def rot_y(theta):
    """
        Returns the rotation matrix around the y axis

        Parameters:
            theta (float): angle in radians

        Returns:
            R (np.array): rotation matrix
    """
    return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                     [0, 1, 0, 0],
                     [-np.sin(theta), 0, np.cos(theta), 0],
                     [0, 0, 0, 1]])


def rot_x(theta):
    """
        Returns the rotation matrix around the x axis

        Parameters:
            theta (float): angle in radians

        Returns:
            R (np.array): rotation matrix
    """
    return np.array([[1, 0, 0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta), np.cos(theta), 0],
                     [0, 0, 0, 1]])


def build_matrix(pos: 'np.ndarray', rot: 'np.ndarray'):
    """
        Builds the transformation matrix from position and Euler angles

        Parameters:
            pos (list[float | int]): position
            rot (list[float | int]): XYZ rotation (Euler angles)

        Returns:
            R (np.array): transformation matrix
    """
    Rx = rot_x(rot[0])
    Ry = rot_y(rot[1])
    Rz = rot_z(rot[2])
    R = np.dot(Rx, Ry)
    R = np.dot(R, Rz)
    R[0][3] = pos[0]
    R[1][3] = pos[1]
    R[2][3] = pos[2]
    return R


class UR5:
    """
        This class initializes the UR5 parameters and defines the control and utility functions
    """

    def __init__(self):
        """
            This function initializes the UR5 parameters
        """
        print('Inicializando parâmetros do robô...')

        # Simulation Parameters
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')

        # Handles
        self.joint_handles = []
        self.finger_handles = []
        self.link_handles = []
        self.connector = self.sim.getObject('/ROBOTIQ_85_attachPoint')
        self.objectSensor = self.sim.getObject('/ROBOTIQ_85_attachProxSensor')
        self.attached_shape = -1

        # DH Forward Kinematics
        self.frame0 = self.sim.getObject('/frame0')
        self.frame6 = self.sim.getObject('/frame6')
        self._init_handles()
        self.start_simulation()
        # clear_output()
        print('Pronto!')

    def start_simulation(self):
        """
            This function starts the simulation
        """
        if self.sim.getSimulationState() != 17:
            print('Iniciando simulação...')
            self.sim.startSimulation()

    def _init_handles(self):
        """
            This function obtains the joints, links and gripper handles
        """
        print('Obtendo handles das juntas...')
        for i in range(1, 7):
            joint_handle = self.sim.getObject('/UR5_joint'+str(i))
            self.joint_handles.append(joint_handle)
            self.sim.setJointMode(joint_handle, self.sim.jointmode_dynamic, 0)
            self.sim.setObjectInt32Param(joint_handle, self.sim.jointintparam_dynctrlmode,
                                         self.sim.jointdynctrl_velocity)
            self.sim.setObjectInt32Param(
                joint_handle, self.sim.jointintparam_motor_enabled, 1)
            self.sim.setObjectInt32Param(
                joint_handle, self.sim.jointintparam_velocity_lock, 1)
            self.sim.setJointTargetVelocity(joint_handle, 0)
            link_handle = self.sim.getObject('/UR5_link'+str(i+1)+'_visible')
            self.link_handles.append(link_handle)

            if i < 3:
                finger_handle = self.sim.getObject('/ROBOTIQ_85_active'+str(i))
                self.finger_handles.append(finger_handle)
                self.sim.setJointMode(
                    finger_handle, self.sim.jointmode_dynamic, 0)
                self.sim.setObjectInt32Param(finger_handle, self.sim.jointintparam_dynctrlmode,
                                             self.sim.jointdynctrl_velocity)
                self.sim.setObjectInt32Param(
                    finger_handle, self.sim.jointintparam_motor_enabled, 1)
                self.sim.setObjectInt32Param(
                    finger_handle, self.sim.jointintparam_velocity_lock, 1)
                self.sim.setJointTargetVelocity(finger_handle, 0.01)

    def forward_kinematics(self, theta: 'list[float | int]'):
        """
            Defines Denavit-Hartenberger parameters for UR5 and calculates
            forward kinematics

            Parameters:
                theta (list[float | int]): joint angles in radians

            Returns:
                T (np.array): transformation matrix
        """
        d1 = 0.089159
        a2 = 0.425
        a3 = 0.39225
        d4 = 0.10915
        d5 = 0.09465
        d6 = 0.17591
        dh_table = np.array([[0, PI/2, d1, 0],
                             [a2, 0, 0, PI/2],
                             [a3, 0, 0, 0],
                             [0, -PI/2, d4, -PI/2],
                             [0, PI/2, d5, 0],
                             [0, 0, d6, 0]])

        A = np.array([np.array([[cos(theta[i]+dh_table[i][3]),
                                 -sin(theta[i]+dh_table[i][3]) *
                                 cos(dh_table[i][1]),
                                 sin(theta[i]+dh_table[i][3]) *
                                 sin(dh_table[i][1]),
                                 dh_table[i][0]*cos(theta[i]+dh_table[i][3])],
                                [sin(theta[i]+dh_table[i][3]),
                                 cos(theta[i]+dh_table[i][3]) *
                                 cos(dh_table[i][1]),
                                 -cos(theta[i]+dh_table[i][3]) *
                                 sin(dh_table[i][1]),
                                 dh_table[i][0]*sin(theta[i]+dh_table[i][3])],
                                [0, sin(dh_table[i][1]),
                                 cos(dh_table[i][1]),
                                 dh_table[i][2]],
                                [0, 0, 0, 1]]) for i in range(6)])

        T = reduce(np.dot, A)
        return T, A

    def transform(self, theta: 'int | float', idx):
        """
            Calculate the transformation matrix between two consecutive frames

            Ex: T_0_1, T_1_2, T_2_3, T_3_4, T_4_5, T_5_6

            Parameters:
                theta (float | int): joint angle in radians
                idx (int): index of the transformation matrix

            Returns:
                T (np.array): transformation matrix
        """
        d1 = 0.089159
        a2 = 0.425
        a3 = 0.39225
        d4 = 0.10915
        d5 = 0.09465
        d6 = 0.17591
        dh_table = np.array([[0, PI/2, d1, 0],
                             [a2, 0, 0, PI/2],
                             [a3, 0, 0, 0],
                             [0, -PI/2, d4, -PI/2],
                             [0, PI/2, d5, 0],
                             [0, 0, d6, 0]])

        th = np.array([[cos(theta+dh_table[idx][3]),
                        -sin(theta+dh_table[idx][3]) *
                        cos(dh_table[idx][1]),
                        sin(theta+dh_table[idx][3]) *
                        sin(dh_table[idx][1]),
                        dh_table[idx][0]*cos(theta+dh_table[idx][3])],
                       [sin(theta+dh_table[idx][3]),
                        cos(theta+dh_table[idx][3]) *
                        cos(dh_table[idx][1]),
                        -cos(theta+dh_table[idx][3]) *
                        sin(dh_table[idx][1]),
                        dh_table[idx][0]*sin(theta+dh_table[idx][3])],
                       [0, sin(dh_table[idx][1]),
                        cos(dh_table[idx][1]),
                        dh_table[idx][2]],
                       [0, 0, 0, 1]])
        return th

    def inverse_kinematics(self, th: 'np.ndarray', shoulder='left', wrist='down', elbow='up'):
        """
            Calculates inverse kinematics for UR5

            Parameters:
                th (np.ndarray): transformation matrix
                shoulder (str): 'left' or 'right'
                wrist (str): 'up' or 'down'
                elbow (str): 'up' or 'down'

            Returns:
                theta (list[float]): joint angles in radians
        """
        a2 = 0.425
        a3 = 0.39225
        d4 = 0.10915
        d6 = 0.17591
        o5 = th.dot(np.array([[0, 0, -d6, 1]]).T)
        xc, yc, zc = o5[0][0], o5[1][0], o5[2][0]

        # Theta 1
        psi = math.atan2(yc, xc)
        phi = math.acos(d4/np.sqrt(xc**2 + yc**2))
        theta1 = np.array([psi - phi + PI/2, psi + phi + PI/2])
        T1 = np.array([limit_angle(theta1[0]), limit_angle(theta1[1])])
        if shoulder == 'left':
            theta1 = T1[0]
        else:
            theta1 = T1[1]

        # Theta 5
        P60 = np.dot(th, np.array([[0, 0, 0, 1]]).T)
        x60 = P60[0][0]
        y60 = P60[1][0]
        z61 = x60*np.sin(T1) - y60*np.cos(T1)
        T5 = np.array([np.arccos((z61 - d4)/d6), -np.arccos((z61 - d4)/d6)]).T
        if shoulder == 'left':
            T5 = T5[0]
            if wrist == 'up':
                theta5 = T5[0]
            else:
                theta5 = T5[1]
        else:
            T5 = T5[1]
            if wrist == 'down':
                theta5 = T5[0]
            else:
                theta5 = T5[1]

        # Theta 6
        th10 = self.transform(theta1, 0)
        th01 = np.linalg.inv(th10)
        th16 = np.linalg.inv(np.dot(th01, th))
        z16_y = th16[1][2]
        z16_x = th16[0][2]
        theta6 = math.atan2(-z16_y/np.sin(theta5), z16_x/np.sin(theta5))+PI
        theta6 = limit_angle(theta6)


        # Theta 3
        th61 = np.dot(th01, th)
        th54 = self.transform(theta5, 4)
        th65 = self.transform(theta6, 5)
        inv = np.linalg.inv(np.dot(th54, th65))
        th41 = np.dot(th61, inv)
        p31 = np.dot(th41, np.array([[0, d4, 0, 1]]).T) - np.array([[0, 0, 0, 1]]).T

        p31_x = p31[0][0]
        p31_y = p31[1][0]
        D = (p31_x**2 + p31_y**2 - a2**2 - a3**2)/(2*a2*a3)
        T3 = np.array([math.atan2(-np.sqrt(1-D**2), D), math.atan2(np.sqrt(1-D**2), D)])
        if shoulder == 'left':
            if elbow == 'up':
                theta3 = T3[0]
            else:
                theta3 = T3[1]
        else:
            if elbow == 'up':
                theta3 = T3[1]
            else:
                theta3 = T3[0]


        # Theta 2
        delta = math.atan2(p31_x, p31_y)
        epsilon = math.acos((a2**2 + p31_x**2 + p31_y**2 - a3**2)/(2*a2*np.sqrt(p31_x**2 + p31_y**2)))
        T2 = np.array([ - delta + epsilon, - delta - epsilon])
        if shoulder == 'left':
            theta2 = T2[0]
        else:
            theta2 = T2[1]

        # Theta 4
        th21 = self.transform(theta2, 1)
        th32 = self.transform(theta3, 2)
        inv = np.linalg.inv(np.dot(th21, th32))
        th43 = np.dot(inv, th41)
        x43_x = th43[0][0]
        x43_y = th43[1][0]
        theta4 = math.atan2(x43_x, -x43_y)

        return [theta1, theta2, theta3, theta4, theta5, theta6]

    def get_ground_truth(self):
        """
            Get ground truth of frame 6 (with respect to frame0)

            Returns:
                np.ndarray: 4x4 homogeneous transformation matrix
        """
        # Get frame6 pose quaternion and convert to homogeneous transformation matrix
        return np.vstack([np.array(self.sim.poseToMatrix(self.sim.getObjectPose(self.frame6, self.frame0)),
                                   dtype=np.float64).reshape(3, 4), [0, 0, 0, 1]])

    def get_joint_angles(self, handles):
        """
            Used internally to get joint angles

            Args:
                handles: list of joint handles

            Returns:
                angles: list of joint angles
        """
        angles = np.zeros(len(handles))
        for handle in handles:
            angles[handles.index(handle)] = limit_angle(
                self.sim.getJointPosition(handle))
        return angles

    def get_jacobian(self, velocities: np.ndarray = None):
        """
            Calculate Jacobian and get the end-effector velocities (linear and angular)
            from the joint velocities

            Parameters:
                velocities (np.ndarray): do not assign any values when using as standalone
                function

            Returns:
                qsi (np.ndarray): 6x1 vector containing 3 linear velocities (x, y, z) and
                3 angular velocities (x, y, z)
        """
        angles = self.get_joint_angles(self.joint_handles)
        if velocities is None:
            velocities = np.array([self.sim.getJointVelocity(j) for j in self.joint_handles]).reshape((6,1))
        _, A = self.forward_kinematics(angles)
        A10 = A[0]
        A20 = np.dot(A[0], A[1])
        A30 = np.dot(A20, A[2])
        A40 = np.dot(A30, A[3])
        A50 = np.dot(A40, A[4])
        A60 = np.dot(A50, A[5])
        Z0 = np.array([[0, 0, 1]]).T
        Z1 = A10[:3, 2].reshape(3, 1)
        Z2 = A20[:3, 2].reshape(3, 1)
        Z3 = A30[:3, 2].reshape(3, 1)
        Z4 = A40[:3, 2].reshape(3, 1)
        Z5 = A50[:3, 2].reshape(3, 1)
        O0 = np.zeros((3,1))
        O1 = A10[:3, 3].reshape(3, 1)
        O2 = A20[:3, 3].reshape(3, 1)
        O3 = A30[:3, 3].reshape(3, 1)
        O4 = A40[:3, 3].reshape(3, 1)
        O5 = A50[:3, 3].reshape(3, 1)
        O6 = A60[:3, 3].reshape(3, 1)
        Jw = np.hstack((Z0, Z1, Z2, Z3, Z4, Z5))
        Jv = np.hstack((np.cross(Z0.T, (O6-O0).T).T,
                        np.cross(Z1.T, (O6-O1).T).T,
                        np.cross(Z2.T, (O6-O2).T).T,
                        np.cross(Z3.T, (O6-O3).T).T,
                        np.cross(Z4.T, (O6-O4).T).T,
                        np.cross(Z5.T, (O6-O5).T).T,))
        J = np.vstack((Jv, Jw))
        qsi = np.dot(J, velocities)
        return qsi

    def move_to_config(self, target: 'list[float | int]', duration=None, graph=False, jacob=False):
        """
            Move to configuration using quintic trajectory

            Args:
                target: list of target angles

                duration: time to reach target in seconds

                graph: whether to plot the trajectory

                jacob: wheter to calculate and return jacobian

            Returns:
                duration: time to reach target in seconds

                max_error: maximum final joint error in degrees

                mean_error: mean final joint error in degrees

                graphs: list of graphs if graph=True

                jacob: linear and angular end-effector velocities
            """
        target = [limit_angle(t) for t in target]
        t0 = self.sim.getSimulationTime()
        v0 = np.zeros(len(self.joint_handles))
        vf = np.zeros(len(self.joint_handles))
        q0 = self.get_joint_angles(self.joint_handles)
        qf = np.array(target)
        a0 = np.zeros(len(self.joint_handles))
        af = np.zeros(len(self.joint_handles))
        if duration is None:
            duration = np.max(np.abs(qf-q0))*(4/(.5*PI))
            if duration < 1.5:
                duration = 1.5
        tf = t0 + duration
        A = np.array([[1, t0, t0**2, t0**3, t0**4, t0**5],
                      [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
                      [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
                      [1, tf, tf**2, tf**3, tf**4, tf**5],
                      [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
                      [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]])
        b = np.array([q0, v0, a0, qf, vf, af])
        x = [np.linalg.solve(A, b[:, i]) for i in range(len(self.joint_handles))]
        time0 = self.sim.getSimulationTime()
        iterations = 0
        end_effector_vel = []
        vel_jacob = [[], [], [], [], [], []]
        pos = [[], [], [], [], [], []]
        vel = [[], [], [], [], [], []]
        acc = [[], [], [], [], [], []]
        jerk = [[], [], [], [], [], []]
        time_arr = [[], [], [], [], [], []]
        while self.sim.getSimulationTime() < tf:
            t = self.sim.getSimulationTime()
            for joint_handle in self.joint_handles:
                self.sim.setJointTargetVelocity(joint_handle, x[self.joint_handles.index(joint_handle)][1] +
                                                2*x[self.joint_handles.index(joint_handle)][2]*t +
                                                3*x[self.joint_handles.index(joint_handle)][3]*t**2 +
                                                4*x[self.joint_handles.index(joint_handle)][4]*t**3 +
                                                5*x[self.joint_handles.index(joint_handle)][5]*t**4)
                if graph:
                    idx = self.joint_handles.index(joint_handle)
                    p = x[idx][0] + x[idx][1]*t + x[idx][2]*t**2 + x[idx][3]*t**3 + x[idx][4]*t**4 + x[idx][5]*t**5
                    v = x[idx][1] + 2*x[idx][2]*t + 3*x[idx][3]*t**2 + 4*x[idx][4]*t**3 + 5*x[idx][5]*t**4
                    a = 2*x[idx][2] + 6*x[idx][3]*t + 12*x[idx][4]*t**2 + 20*x[idx][5]*t**3
                    j = 6*x[idx][3] + 24*x[idx][4]*t + 60*x[idx][5]*t**2
                    time_arr[idx].append(t-time0)
                    pos[idx].append(p)
                    vel[idx].append(v)
                    acc[idx].append(a)
                    jerk[idx].append(j)
                if jacob:
                    idx = self.joint_handles.index(joint_handle)
                    v = x[idx][1] + 2*x[idx][2]*t + 3*x[idx][3]*t**2 + 4*x[idx][4]*t**3 + 5*x[idx][5]*t**4
                    vel_jacob[idx].append(v)
            if jacob:
                end_effector_vel.append(self.get_jacobian(velocities=np.array([vj[-1] for vj in vel_jacob]).reshape((6, 1))))
            time.sleep(0.001)
            iterations += 1
        for joint_handle in self.joint_handles:
            self.sim.setJointTargetVelocity(joint_handle, 0)
        timef = self.sim.getSimulationTime()
        error = np.abs(np.array(target) - self.get_joint_angles(self.joint_handles))*180/np.pi
        print('Iterações totais: ', iterations)
        return timef-time0, np.max(error), np.mean(error), (pos, vel, acc, jerk, time_arr), end_effector_vel

    def move_to_pose(self, pos: 'np.ndarray', rot: 'np.ndarray', wrist='down', duration=None):
        """
            Move to specified position and orientation

            Parameters:
                pos: [x, y, z] coordinates
                rot: [rot_x, rot_y, rot_z] Euler angles
                wrist: 'up' or 'down'
                duration: time to reach position
        """
        T = build_matrix(pos, rot)
        joint_angles = self.inverse_kinematics(T, wrist=wrist, shoulder='left', elbow='up')
        if duration is not None:
            self.move_to_config(target=joint_angles, duration=duration)
        else:
            self.move_to_config(joint_angles)
        gt = self.get_ground_truth()
        print('Erro pose final: ', np.linalg.norm(T-gt)/np.linalg.norm(gt)*100, '%')

    def close_gripper(self):
        """
            Close the Robotiq_85 gripper
        """
        p1 = self.sim.getJointPosition(self.finger_handles[0])
        p2 = self.sim.getJointPosition(self.finger_handles[1])
        while p1 < p2 - 0.008:
            self.sim.setJointTargetVelocity(self.finger_handles[0], -0.01)
            self.sim.setJointTargetVelocity(self.finger_handles[1], -0.04)
            time.sleep(0.01)
        self.sim.setJointTargetVelocity(self.finger_handles[0], -0.04)
        self.sim.setJointTargetVelocity(self.finger_handles[1], -0.04)
        time.sleep(1.3)

    def open_gripper(self):
        """
            Open the Robotiq_85 gripper
        """
        p1 = self.sim.getJointPosition(self.finger_handles[0])
        p2 = self.sim.getJointPosition(self.finger_handles[1])
        if p1 < p2:
            self.sim.setJointTargetVelocity(self.finger_handles[0], 0.04)
            self.sim.setJointTargetVelocity(self.finger_handles[1], 0.02)
        else:
            self.sim.setJointTargetVelocity(self.finger_handles[0], 0.02)
            self.sim.setJointTargetVelocity(self.finger_handles[1], 0.04)
        time.sleep(1.5)


def get_dataset(folder='machine_learning/img/'):
    """
        Get the dataset json with all labels and image paths

        Parameters:
            folder (string): dataset containing folder

        Return:
            data (list): list of json objects with keys ('x', 'y', 'x_img', 'y_img', 'raw_file')
    """
    data = []
    with open(folder + 'ground_truth.json', 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


class ComputerVision:

    def __init__(self):
        """
            Initialize variables
        """
        self.model = None
        client = RemoteAPIClient()
        self.sim = client.getObject('sim')
        self._get_handles()
        self.folder = 'machine_learning/img/'
        self.x_range = [0.1, -0.24]
        self.y_range = [-0.83, -0.55]
        self.x_img_range = [52, 184]
        self.y_img_range = [52, 161]
        self.z_cup = 0.05797996154999335

    def _get_handles(self):
        """
            Get object handles
        """
        self.cup = self.sim.getObject('/Cup')
        self.camera = self.sim.getObject('/Vision_sensor')
        self.frame0 = self.sim.getObject('/frame0')

    def create_dataset(self, size=5000, folder=None):
        """
            Create a dataset with different cup positions and corresponding labels
            for machine learning training

            Parameters:
                size (int): number of images of the dataset
                folder (string): path to the folder where the dataset will be created
        """
        if folder is not None:
            self.folder = folder
        with open(self.folder + 'grount_truth.json', 'w') as f:
            for i in range(size):
                X = np.linspace(self.x_range[0], self.x_range[1], 100)
                Y = np.linspace(self.y_range[0], self.y_range[1], 100)
                X_img = np.linspace(self.x_img_range[0], self.x_img_range[1], 100)
                Y_img = np.linspace(self.y_img_range[0], self.y_img_range[1], 100)
                xc, yc = np.random.randint(0, 99, 2)
                self.sim.getObjectPosition(self.cup, self.frame0)
                self.sim.setObjectPosition(self.cup, self.frame0, [X[xc], Y[yc], self.z_cup])
                img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
                img = np.frombuffer(img, dtype=np.uint8).reshape((resY, resX, 3))
                img = np.flip(img, axis=0)
                plt.imsave(folder + str(i)+ '.png', img)
                json.dump({'x': X[xc], 'y': Y[yc], 'x_img': int(X_img[xc]), 'y_img': int(Y_img[yc]), 'raw_file': str(i) + '.png' }, f)
                f.write('\n')

    def move_cup(self):
        """
            Move the cup to a different position within the limits defined

            Returns:
                p (list[float]): cup position
        """
        X = np.linspace(self.x_range[0], self.x_range[1], 100)
        Y = np.linspace(self.y_range[0], self.y_range[1], 100)
        xc, yc = np.random.randint(0, 99, 2)
        self.sim.getObjectPosition(self.cup, self.frame0)
        self.sim.setObjectPosition(self.cup, self.frame0, [X[xc], Y[yc], self.z_cup])
        img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape((resY, resX, 3))
        img = np.flip(img, axis=0)
        img1 = img.copy()
        limits = [(self.x_img_range[0], self.y_img_range[0]),
                  (self.x_img_range[1], self.y_img_range[0]),
                  (self.x_img_range[1], self.y_img_range[1]),
                  (self.x_img_range[0], self.y_img_range[1])]
        for i in range(2):
            for j in range(2):
                cv2.polylines(img1, np.int32([limits]), isClosed=True, color=(0, 255, 0), thickness=1)
        plt.imshow(img1)
        plt.show()
        return np.array(self.sim.getObjectPosition(self.cup, self.frame0))

    def load_model(self, model_path='machine_learning/cup_position_model.h5'):
        """
            Load the trained model

            Parameters:
                model_path (string): path to the model h5 file
        """
        self.model = load_model(model_path)

    def predict(self):
        """
            Predict the cup position

            Returns:
                p (np.ndarray): cup position
        """
        img, resX, resY = self.sim.getVisionSensorCharImage(self.camera)
        img = np.frombuffer(img, dtype=np.uint8).reshape((resY, resX, 3))
        img = np.flip(img, axis=0)
        img = img / 255.0
        input_img = np.expand_dims(img, axis=0)
        prediction = self.model.predict(input_img)
        prediction = prediction.astype(int)
        x = prediction[0][0]
        y = prediction[0][1]
        img1 = img.copy()
        img1 = cv2.circle(img1, (x, y), 3, (255, 0, 0), -1)
        plt.imshow(img1)
        x_normalized = (x - self.x_img_range[0]) / (self.x_img_range[1] - self.x_img_range[0])
        y_normalized = (y - self.y_img_range[0]) / (self.y_img_range[1] - self.y_img_range[0])
        x_real = self.x_range[0] + x_normalized * (self.x_range[1] - self.x_range[0])
        y_real = self.y_range[0] + y_normalized * (self.y_range[1] - self.y_range[0])
        z_real = self.z_cup
        cup_position = np.array([x_real, y_real, z_real])
        return cup_position

