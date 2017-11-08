#!/usr/bin/env python
# Python implementation of "ekf_slam"

import rospy
from fcu_common.msg import FW_State, GPS
from sensor_msgs.msg import Imu, FluidPressure
from std_msgs.msg import Float32
from math import *
import numpy as np

class ekf_slam:
    #init functions
    def __init__(self):
        #init stuff

        #get stuff


    #subclasses
    class input_s:
        #stuff
    class output_s:
        #stuff
    class params_s:
        #stuff
    #callback functions
    def propagate(self):
        for i in range(0,self.N_):
            cp = cos(self.xhat[0]) # cos(phi)
			sp = sin(self.xhat[0]) # sin(phi)
			tt = tan(self.xhat[1]) # tan(theta)
			ct = cos(self.xhat[1]) # cos(theta)
            st = sin(self.xhat[1]) # cos(theta)

            self.f = np.array([[p_x_dot], \
            [p_y_dot], \
            [p_z_dot], \
            [cp*st*a_z], \
            [-sp*az],[g+cp*ct*a_z], \
            [p+q*sp*tt + r*cp*tt], \
            [q*sp- r*sp], \
            [q*sp/ct+ r*cp/ct]])

            x_hat += (params.Ts/self.N_)*self.f

            A = np.array([[0,0,0,1,0,0,0,0,0], \
            [0,0,0,0,1,0,0,0,0], \
            [0,0,0,0,0,1,0,0,0], \
            [0,0,0,0,0,0,-sp*st*a_z,cp*ct*a_z,0], \
            [0,0,0,0,0,0,-cp*a_z,0,0], \
            [0,0,0,0,0,0,-sp*ct*a_z,-cp*st*a_z,0], \
            [0,0,0,0,0,0,q*cp*tt - r*sp*tt,(q*sp+r*cp/(ct)**2,0)], \
            [0,0,0,0,0,0,-q*sp-r*cp,0,0], \
            [0,0,0,0,0,0,(q*cp-r*sp)/ct,-(q*sp+r*cp)*tt/ct]])

            P = P + (T_out/N)*(np.matmul(A,P)+np.matmul(P,A.T)+np.matmul(G,Q,G.T))
    def update(self):
