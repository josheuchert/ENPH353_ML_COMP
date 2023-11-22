#! /usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import cv2
import numpy as np


#initialize publishers
rospy.init_node('controller_node')
move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=10)    
br = CvBridge()
rospy.sleep(1)

class Imitate:
    def __init__(self):
        self.cur_state = None
        self.prev_state = None

    def update_state(self, new_state):
        self.prev_state = self.cur_state
        self.cur_state = new_state

    def get_current_state(self):
        return self.cur_state

    def get_previous_state(self):
        return self.prev_state

    def camera_callback(self, data):
        try:
            cv_image = br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        im_cut = cv_image[360:720,0:1280]
        im_grey = img_gray = cv2.cvtColor(im_cut, cv2.COLOR_BGR2GRAY)
        threshold = 180
        _, binary = cv2.threshold(im_grey,threshold,255,cv2.THRESH_BINARY)
        
        out = cv2.putText(binary,self.cur_state, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            
        cv2.imshow("states", out)
        cv2.waitKey(1) 
        # save sate and image



    def move_callback(self, data):
        linear_x = data.linear.x
        angular_z = data.angular.z
        self.cur_state = f"{linear_x},{angular_z}"

trial = Imitate()

def controller():
    rospy.Subscriber('/R1/pi_camera/image_raw',Image,trial.camera_callback)
    rospy.Subscriber('/clock',Clock)
    rospy.Subscriber('/R1/cmd_vel',Twist,trial.move_callback)
    rospy.sleep(1)

while not rospy.is_shutdown():
    controller()
    rospy.sleep(1)
    rospy.spin()    



#!/usr/bin/env python3


