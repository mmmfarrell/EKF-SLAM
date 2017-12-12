#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from pyquat import Quaternion
import rosbag
from geometry_msgs.msg import Vector3Stamped, PoseStamped
from sensor_msgs.msg import Imu, Range
from aruco_localization.msg import MarkerMeasurementArray
from rosflight_msgs.msg import Attitude
import scipy.signal


def calculate_velocity_from_position(t, position, orientation):
    # Calculate body-fixed velocity by differentiating position and rotating
    # into the body frame
    b, a = scipy.signal.butter(8, 0.03)  # Create a Butterworth Filter
    # differentiate Position
    delta_x = np.diff(position, axis=0)
    delta_t = np.diff(t)
    unfiltered_inertial_velocity = np.vstack((np.zeros((1, 3)), delta_x / delta_t[:, None]))
    # Filter
    v_inertial = scipy.signal.filtfilt(b, a, unfiltered_inertial_velocity, axis=0)
    # Rotate into Body Frame
    vel_data = []
    for i in range(len(t)):
        q_I_b = Quaternion(orientation[i, :, None])
        vel_data.append(q_I_b.rot(v_inertial[i, None].T).T)

    vel_data = np.array(vel_data).squeeze()
    return vel_data


def load_data(filename, outbag):
    print "loading rosbag", filename
    # First, load IMU data
    bag = rosbag.Bag(filename)
    # outbag = rosbag.Bag('thor_mocap_new.bag', 'w')
    imu_data = []
    truth_pose_data = []
    timestamp_data = []

    for topic, msg, t in tqdm(bag.read_messages(topics=['/aruco/measurements',
                                                   '/mocapNED',
                                                   '/mocap/thor/pose',
                                                   '/imu/data',
                                                   '/sonar',
                                                   '/attitude'])):


        if topic == '/mocap/thor/pose':
            # outbag.write(topic, msg, msg.header.stamp)

            truth_meas = [msg.header.stamp.to_sec(),
                          msg.pose.position.z, -msg.pose.position.x, -msg.pose.position.y,
                          -msg.pose.orientation.w, -msg.pose.orientation.z, msg.pose.orientation.x, msg.pose.orientation.y]
            truth_pose_data.append(truth_meas)
            timestamp_datapoint = [msg.header.stamp.secs, msg.header.stamp.nsecs, msg.header.seq]#, msg.header.frame_id]
            timestamp_data.append(timestamp_datapoint)
            # z = int('%d%d' % (timestamp_datapoint[0], timestamp_datapoint[1]))
            # print z
            # print t



        # if topic == '/imu/data':
        #     outbag.write(topic, msg, msg.header.stamp)
        #
        # if topic == '/aruco/measurements':
        #     outbag.write(topic, msg, msg.header.stamp)
        #
        # if topic == '/sonar':
        #     outbag.write(topic, msg, msg.header.stamp)
        #
        # if topic == '/attitude':
        #     outbag.write(topic, msg, msg.header.stamp)
        #
        # if topic == '/mocapNED':
        #     outbag.write(topic, msg, msg.header.stamp)

    imu_data = np.array(imu_data)
    truth_pose_data = np.array(truth_pose_data)
    timestamp_data = np.array(timestamp_data)

    # Remove Bad Truth Measurements
    good_indexes = np.hstack((True, np.diff(truth_pose_data[:,0]) > 1e-3))
    truth_pose_data = truth_pose_data[good_indexes]
    timestamp_data = timestamp_data[good_indexes]
    vel_data = calculate_velocity_from_position(truth_pose_data[:,0], truth_pose_data[:,1:4], truth_pose_data[:,4:8])
    vel_data = np.hstack((timestamp_data,vel_data))

    return vel_data

def bagwriter(filename, outbag, data):

    bag = rosbag.Bag(filename)
    i=0
    for topic, msg, t in tqdm(bag.read_messages(topics=['/aruco/measurements',
                                               '/mocapNED',
                                               '/mocap/thor/pose',
                                               '/imu/data',
                                               '/sonar',
                                               '/attitude'])):
        if topic == '/imu/data':
            outbag.write(topic, msg, msg.header.stamp)

        if topic == '/aruco/measurements':
            outbag.write(topic, msg, msg.header.stamp)

        if topic == '/sonar':
            outbag.write(topic, msg, msg.header.stamp)

        if topic == '/attitude':
            outbag.write(topic, msg, msg.header.stamp)

        if topic == '/mocapNED':
            outbag.write(topic, msg, msg.header.stamp)

        if topic == '/mocap/thor/pose':
            outbag.write(topic, msg, msg.header.stamp)

        velocities = Vector3Stamped()
        velocities.header.stamp.secs = data[i,0]
        velocities.header.stamp.nsecs = data[i,1]
        # print velocities.header.stamp
        # print t
        if velocities.header.stamp < t and i<len(data)-1:
            # print 'should be working'
            velocities.header.seq = data[i,2]
            velocities.vector.x = data[i,3]
            velocities.vector.y = data[i,4]
            velocities.vector.z = data[i,5]
            outbag.write('/velocities', velocities, velocities.header.stamp)
            i = i+1


            # for i in range (0,len(data)):
            #     velocities.header.seq = i
            #     velocities.header.stamp.secs = data[i,0]
            #     velocities.header.seq = data[i,1]
            #     # velocities.header.frame_id = data[i,2]
            #     velocities.vector.x = data[i,3]
            #     velocities.vector.y = data[i,4]
            #     velocities.vector.z = data[i,5]
            #     outbag.write('/velocities', velocities, velocities.header.stamp)

            # finally:
            #     outbag.close()

if __name__ == '__main__':
    outbag = rosbag.Bag('thor_mocap_new.bag', 'w')
    data = load_data('thor_mocap_first_good.bag', outbag)

    bagwriter('thor_mocap_first_good.bag',outbag, data)

    print "done"
    outbag.close()
