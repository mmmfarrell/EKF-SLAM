#!/usr/bin/env python
# Python implementation of "ekf_slam"

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from math import *
import numpy as np
import tf

class ekf_slam:
    #init functions
    def __init__(self):
        #init stuff
        #get stuff

        # Estimator stuff
        # x = pn, pe, pd, phi, theta, psi
        self.xhat = np.zeros((6,1))
        self.xhat_odom = Odometry()

        # Covariance matrix
        # self.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

        # Measurements stuff
        # Truth
        self.truth_pn = 0.0
        self.truth_pe = 0.0
        self.truth_pd = 0.0
        self.truth_phi = 0.0
        self.truth_theta = 0.0
        self.truth_psi = 0.0
        self.truth_p = 0.0
        self.truth_q = 0.0
        self.truth_r = 0.0
        self.truth_pndot = 0.0
        self.truth_pedot = 0.0
        self.truth_pddot = 0.0
        self.truth_u = 0.0
        self.truth_v = 0.0
        self.truth_w = 0.0
        self.prev_time = 0.0

        # Number of propagate steps
        self.N = 5

        # ROS Stuff
        # Init subscribers
        self.truth_sub_ = rospy.Subscriber('/slammer/ground_truth/odometry/NED', Odometry, self.truth_callback)
        self.imu_sub_ = rospy.Subscriber('/slammer/imu/data', Imu, self.imu_callback)

        # Init publishers
        self.estimate_pub_ = rospy.Publisher('/ekf_estimate', Odometry, queue_size=10)

        # # Init Timer
        self.pub_rate_ = 100. #
        self.update_timer_ = rospy.Timer(rospy.Duration(1.0/self.pub_rate_), self.pub_est)


    def propagate(self, dt):
        # Using truth for:
        #           - p (near perfect IMU)
        #           - q (near perfect IMU)
        #           - r (near perfect IMU)
        #           - u
        #           - v
        #           - w

        for _ in range(self.N):
            # Calc trig functions
            sp = sin(self.xhat[3])
            cp = cos(self.xhat[3])
            st = sin(self.xhat[4])
            ct = cos(self.xhat[4])
            tt = tan(self.xhat[4])
            spsi = sin(self.xhat[5])
            cpsi = cos(self.xhat[5])
            # sp = sin(self.truth_phi)
            # cp = cos(self.truth_phi)
            # st = sin(self.truth_theta)
            # ct = cos(self.truth_theta)
            # tt = tan(self.truth_theta)
            # spsi = sin(self.truth_psi)
            # cpsi = cos(self.truth_psi)

            # Calc pos_dot
            lin_vel = np.zeros((3,1))
            lin_vel[0] = self.truth_u
            lin_vel[1] = self.truth_v
            lin_vel[2] = self.truth_w

            pos_rot_mat = np.array([[ct*cpsi, sp*st*cpsi-cp*spsi, cp*st*cpsi+sp*spsi],
                                    [ct*spsi, sp*st*spsi+cp*cpsi, cp*st*spsi-sp*cpsi],
                                    [-st, sp*ct, cp*ct]])

            pos_dot = np.matmul(pos_rot_mat, lin_vel)

            # calc eul dot
            ang_rates = np.zeros((3,1))
            ang_rates[0] = self.truth_p
            ang_rates[1] = self.truth_q
            ang_rates[2] = self.truth_r


            rot_matrix = np.array([[1., sp*tt, cp*tt], [0., cp, -sp], [0., sp/ct, cp/ct]])

            eul_dot = np.matmul(rot_matrix, ang_rates)

            # Calc xdot
            xdot = np.zeros((6,1))
            xdot[0] = pos_dot[0]
            xdot[1] = pos_dot[1]
            xdot[2] = pos_dot[2]
            xdot[3] = eul_dot[0]
            xdot[4] = eul_dot[1]
            xdot[5] = eul_dot[2]

            self.xhat += xdot*dt/float(self.N)

    def update(self):
        pass

    def pub_est(self, event):

        # pack up estimate to ROS msg and publish
        self.xhat_odom.header.stamp = rospy.Time.now()
        self.xhat_odom.pose.pose.position.x = self.xhat[0] # pn
        self.xhat_odom.pose.pose.position.y = self.xhat[1] # pe
        self.xhat_odom.pose.pose.position.z = self.xhat[2] # pd

        quat = tf.transformations.quaternion_from_euler(self.xhat[3].copy(), self.xhat[4].copy(), self.xhat[5].copy())

        # These are euler angles
        self.xhat_odom.pose.pose.orientation.x = quat[0]
        self.xhat_odom.pose.pose.orientation.y = quat[1]
        self.xhat_odom.pose.pose.orientation.z = quat[2]
        self.xhat_odom.pose.pose.orientation.w = quat[3]
        # self.xhat_odom.twist.twist.angular.x = self.xhat[3] # phi
        # self.xhat_odom.twist.twist.angular.y = self.xhat[4] # theta
        # self.xhat_odom.twist.twist.angular.z = self.xhat[5] # psi

        self.estimate_pub_.publish(self.xhat_odom)

    # Callback Functions
    def truth_callback(self, msg):

        time = msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9


        # Map msg to class variables
        self.truth_pn = msg.pose.pose.position.x
        self.truth_pe = msg.pose.pose.position.y
        self.truth_pd = msg.pose.pose.position.z

        quat = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quat)

        self.truth_phi = euler[0]
        self.truth_theta = euler[1]
        self.truth_psi = euler[2]

        # self.truth_p = msg.twist.twist.angular.x
        # self.truth_q = msg.twist.twist.angular.y
        # self.truth_r = msg.twist.twist.angular.z

        self.truth_u = msg.twist.twist.linear.x
        self.truth_v = msg.twist.twist.linear.y
        self.truth_w = msg.twist.twist.linear.z

        if (self.prev_time != 0.0):
            dt = time - self.prev_time
            self.propagate(dt)

        self.prev_time = time

    def imu_callback(self, msg):

        # Map msg to class variables

        # Angular rates
        self.truth_p = msg.angular_velocity.x
        self.truth_q = msg.angular_velocity.y
        self.truth_r = msg.angular_velocity.z
        # self.imu_p = msg.angular_velocity.x
        # self.imu_q = msg.angular_velocity.y
        # self.imu_r = msg.angular_velocity.z

        # Linear accel
        self.imu_ax = msg.linear_acceleration.x
        self.imu_ay = msg.linear_acceleration.y
        self.imu_az = msg.linear_acceleration.z


##############################
#### Main Function to Run ####
##############################
if __name__ == '__main__':
    # Initialize Node
    rospy.init_node('slam_estimator')

    # init path_manager_base object
    estimator = ekf_slam()

    while not rospy.is_shutdown():
        rospy.spin()
