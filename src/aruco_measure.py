#!/usr/bin/env python
# Python implementation of "ekf_slam"

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32
from aruco_localization.msg import MarkerMeasurementArray
from math import *
import numpy as np
import tf

class ekf_slam:
    #init functions
    def __init__(self):
        #init stuff
        #get stuff

        # Estimator stuff
        # x = pn, pe, pd, u, v, w, phi, theta, psi
        self.xhat = np.zeros((9,1))
        self.xhat_odom = Odometry()

        # Covariance matrix
        self.P = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

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

        #aruco Stuff
        self.aruco_location = {
        100:[0.0, 0.0, 0.0],
        101:[0.0, 14.5, 5.0],
        102:[5.0, 14.5, 5.0],
        103:[-5.0, 14.5, 5.0],
        104:[0.0, -14.5, 5.0],
        105:[5.0, -14.5, 5.0],
        106:[-5.0, -14.5, 5.0],
        107:[7.0, 0.0, 5.0],
        108:[7.0, 7.5, 5.0],
        109:[7.0, -7.5, 5.0],
        110:[-7.0, 0.0, 5.0],
        111:[-7.0, 7.5, 5.0],
        112:[-7.0, -7.5, 5.0],
        }

        # IMU Stuff
        self.imu_p = 0.0
        self.imu_q = 0.0
        self.imu_r = 0.0
        self.imu_ax = 0.0
        self.imu_ay = 0.0
        self.imu_az = 0.0

        # Constants
        self.g = 9.8

        # ROS Stuff
        # Init subscribers
        self.truth_sub_ = rospy.Subscriber('/slammer/ground_truth/odometry/NED', Odometry, self.truth_callback)
        self.imu_sub_ = rospy.Subscriber('/slammer/imu/data', Imu, self.imu_callback)
        self.aruco_sub = rospy.Subscriber('/aruco/measurements', MarkerMeasurementArray, self.aruco_meas_callback )

        # Init publishers
        self.estimate_pub_ = rospy.Publisher('ekf_estimate', Odometry, queue_size=10)


        # Init Timer
        self.propagate_rate_ = 100. #
        self.update_timer_ = rospy.Timer(rospy.Duration(1.0/self.propagate_rate_), self.propagate)




    def propagate(self, event):
        # for i in range(0,self.N_):
        cp = cos(self.truth_phi) # cos(phi)
        sp = sin(self.truth_phi) # sin(phi)
        tt = tan(self.truth_theta) # tan(theta)
        ct = cos(self.truth_theta) # cos(theta)
        st = sin(self.truth_theta) # cos(theta)

        # cp = cos(self.xhat[6]) # cos(phi)
        # sp = sin(self.xhat[6]) # sin(phi)
        # tt = tan(self.xhat[7]) # tan(theta)
        # ct = cos(self.xhat[7]) # cos(theta)
        # st = sin(self.xhat[7]) # cos(theta)

        p = self.imu_p
        q = self.imu_q
        r = self.imu_r

        # self.f = np.array([[p_x_dot],   # pndot
        # [p_y_dot],                      # pedot
        # [p_z_dot],                      # pddot
        # [cp*st*a_z],                    # udot
        # [-sp*az],                       # vdot
        # [g+cp*ct*a_z],                  # wdot
        # [p+q*sp*tt + r*cp*tt],          # phidot
        # [q*sp- r*sp],                   # thetadot
        # [q*sp/ct+ r*cp/ct]])            # psidot

        self.f = np.array([[self.truth_pndot],   # pndot
        [self.truth_pedot],                      # pedot
        [self.truth_pddot],                      # pddot
        [cp*st*self.imu_az],                    # udot
        [-sp*self.imu_az],                       # vdot
        [self.g+cp*ct*self.imu_az],                  # wdot
        [p+q*sp*tt + r*cp*tt],          # phidot
        [q*sp- r*sp],                   # thetadot
        [q*sp/ct+ r*cp/ct]])            # psidot

        # x_hat += (params.Ts/self.N_)*self.f
        self.xhat += (1./self.propagate_rate_)*self.f

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -sp*st*self.imu_az, cp*ct*self.imu_az, 0],
        [0, 0, 0, 0, 0, 0, -cp*self.imu_az, 0, 0],
        [0, 0, 0, 0, 0, 0, -sp*ct*self.imu_az, -cp*st*self.imu_az, 0],
        [0, 0, 0, 0, 0, 0, q*cp*tt-r*sp*tt, (q*sp+r*cp)/(ct)**2, 0],
        [0, 0, 0, 0, 0, 0, -q*sp-r*cp, 0, 0],
        [0, 0, 0, 0, 0, 0, (q*cp-r*sp)/ct, -(q*sp+r*cp)*tt/ct, 0]])

        # P = P + (T_out/N)*(np.matmul(A,P)+np.matmul(P,A.T)+np.matmul(G,Q,G.T))
        # print self.P
        # print [0,0,0,0,0,0,-sp*st*self.imu_az,cp*ct*self.imu_az,0]
        self.P = self.P + (1./self.propagate_rate_)*(np.matmul(A,self.P)+np.matmul(self.P,A.T))#+np.matmul(G,Q,G.T))

        # TODO Move this to its own func
        # pack up estimate to ROS msg and publish
        self.xhat_odom.pose.pose.position.x = self.xhat[0] # pn
        self.xhat_odom.pose.pose.position.y = self.xhat[1] # pe
        self.xhat_odom.pose.pose.position.z = self.xhat[2] # pd
        self.xhat_odom.twist.twist.linear.x = self.xhat[3] # u
        self.xhat_odom.twist.twist.linear.y = self.xhat[4] # v
        self.xhat_odom.twist.twist.linear.z = self.xhat[5] # w

        # These are euler angles
        self.xhat_odom.pose.pose.orientation.x = self.xhat[6] # phi
        self.xhat_odom.pose.pose.orientation.y = self.xhat[7] # theta
        self.xhat_odom.pose.pose.orientation.z = self.xhat[8] # psi

        self.estimate_pub_.publish(self.xhat_odom)

    def update(self):
        m = self.aruco_location[self.aruco_id]
        rangehat = np.double(np.sqrt((m[0]-self.xhat[0])**2 + (-m[1]-self.xhat[1])**2))

        zhat = np.array([[rangehat],
                        [np.arctan2(-m[1]-self.xhat[1],m[0]-self.xhat[0])-self.xhat[8]]])

        C = np.array([[np.double(-(m[0]-self.xhat[0])/rangehat) , -((-m[1])-self.xhat[1])/rangehat,0,0,0,0,0,0,0 ],
                           [((-m[1])-self.xhat[1])/rangehat**2  , -(m[0]-self.xhat[0])/rangehat**2,0,0,0,0,0,0,-1]])

        # print self.aruco_id, self.xhat[0], self.xhat[1], self.range, self.H
        S = np.matmul(C,np.matmul(self.P,C.T))#+self.Q
        self.L =np.matmul(self.P,np.matmul(C.T,np.linalg.inv(S)))

        self.xhat = self.xhat + np.matmul(self.L,(self.z-zhat))
        self.P = np.matmul((np.identity(9)-np.matmul(self.L,C)),self.P)

        # Callback Functions
    def truth_callback(self, msg):

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

        self.truth_p = msg.twist.twist.angular.x
        self.truth_q = msg.twist.twist.angular.y
        self.truth_r = msg.twist.twist.angular.z
        self.truth_pndot = msg.twist.twist.linear.x
        self.truth_pedot = msg.twist.twist.linear.y
        self.truth_pddot = msg.twist.twist.linear.z

    def imu_callback(self, msg):

        # Map msg to class variables

        # Angular rates
        self.imu_p = msg.angular_velocity.x
        self.imu_q = msg.angular_velocity.y
        self.imu_r = msg.angular_velocity.z

        # Linear accel
        self.imu_ax = msg.linear_acceleration.x
        self.imu_ay = msg.linear_acceleration.y
        self.imu_az = msg.linear_acceleration.z

    def aruco_meas_callback(self, msg):
        if len(msg.poses)>0:
            for i in range (0,len(msg.poses)):
                self.aruco_id = msg.poses[i].aruco_id

                self.aruco_x = msg.poses[i].position.x
                self.aruco_y = msg.poses[i].position.y
                self.aruco_z = msg.poses[i].position.z

                self.aruco_phi = msg.poses[i].euler.x
                self.aruco_theta = msg.poses[i].euler.y
                self.aruco_psi = msg.poses[i].euler.z

                self.range = np.sqrt(self.aruco_x**2 + self.aruco_y**2 + self.aruco_z**2)
                self.bearing_2d = np.arctan2(self.aruco_x,self.aruco_z)#-self.truth_psi
                self.z = np.array([[self.range],[self.bearing_2d]])
                self.update()

##############################
#### Main Function to Run ####
##############################
if __name__ == '__main__':
    # Initialize Node
    rospy.init_node('slam_estimator')

    # init path_manager_base object
    estimator = ekf_slam()

    rospy.spin()
