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
import keras
from keras.models import load_model
drive_model = load_model('test1.h5',compile=False)
drive_model.compile(
                   optimizer='adam',
                   loss=keras.losses.MeanSquaredError()
                  )
drive_model.summary()


#initialize publishers
rospy.init_node('controller_node')
move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=10)    
br = CvBridge()
score_pub = rospy.Publisher('/score_tracker', String, 
  queue_size=10)
rospy.sleep(1)
id = "winux"
password = "password"



# Clue message structure 
# team ID: max 8 characters (no spaces)
# team password: max 8 characters (no spaces)
# clue location: int (-1 to 8); -1 and 0 are special cases - see below
# clue prediction: n characters (no spaces, all capitals)

#publiser helper function
def pub_clue(id,password,location,prediciton):
    formatted_string = f"{id},{password},{location},{prediciton}"
    score_pub.publish(formatted_string)

#Time trials move forward and stop 

def camera_callback(data):
    try:
        cv_image = br.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    im_cut = cv_image[360:720,0:1280]
    im_grey = img_gray = cv2.cvtColor(im_cut, cv2.COLOR_BGR2GRAY)
    threshold = 180
    _, binary = cv2.threshold(im_grey,threshold,255,cv2.THRESH_BINARY)
    img = np.array(binary)  # Convert to NumPy array
      # resize image
    dim = (200, 66)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = resized / 255.0  # Normalize to range [0, 1] (assuming RGB images)
    cv2.imshow("states", binary)
    cv2.waitKey(1) 
    img_aug = np.expand_dims(img, axis=0)
    z_predict = drive_model.predict(img_aug)
    z_predict=z_predict*2-1
    print(z_predict)
    NN_move = Twist()
    NN_move.angular.z = z_predict
    NN_move.linear.y = 0
    NN_move.linear.x = .3
    move_pub.publish(NN_move)


def controller():
    rospy.Subscriber('/R1/pi_camera/image_raw',Image,camera_callback)
    rospy.Subscriber('/clock',Clock)
    rospy.sleep(1)

while not rospy.is_shutdown():
    pub_clue(id,password,0,"NA")
    controller()
    pub_clue(id,password,-1,"NA")
    rospy.spin()    
