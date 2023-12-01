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
import tensorflow as tf
# import keras
# from keras.models import load_model
# drive_model = load_model('test6.h5',compile=False)
# drive_model.compile(
#                    optimizer='adam',
#                    loss=keras.losses.MeanSquaredError()
#                   )
# drive_model.summary()
interpreter_road = tf.lite.Interpreter(model_path='quantized_model_road2.tflite')
interpreter_road.allocate_tensors()
input_details_road = interpreter_road.get_input_details()[0]
output_details_road = interpreter_road.get_output_details()[0]
print("Loaded Road")

interpreter_grass = tf.lite.Interpreter(model_path='quantized_model_grass2.tflite')
interpreter_grass.allocate_tensors()
input_details_grass = interpreter_grass.get_input_details()[0]
output_details_grass = interpreter_grass.get_output_details()[0]
print("Loaded Grass")

interpreter_mountain = tf.lite.Interpreter(model_path='quantized_model_mountain.tflite')
interpreter_mountain.allocate_tensors()
input_details_mountain = interpreter_mountain.get_input_details()[0]
output_details_mountain = interpreter_mountain.get_output_details()[0]
print("Loaded Mountain")

# interpreter_yoda = tf.lite.Interpreter(model_path='quantized_model_road.tflite')
# interpreter_yoda.allocate_tensors()
# input_details_yoda = interpreter_yoda.get_input_details()[0]
# output_details_yoda = interpreter_yoda.get_output_details()[0]
# print("Loaded Yoda")

def run_model(img, interpreter, input_details, output_details):
    img_aug = img.reshape((1, 90, 160, 1)).astype(input_details["dtype"])
    interpreter.set_tensor(input_details_road["index"], img_aug)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    print(output)
    output = denormalize_value(output, -2, 2)
    print(output)
    
    return output


def denormalize_value(normalized_value, min_val, max_val):
    return (normalized_value * (max_val - min_val)) + min_val



# Clue message structure 
# team ID: max 8 characters (no spaces)
# team password: max 8 characters (no spaces)
# clue location: int (-1 to 8); -1 and 0 are special cases - see below
# clue prediction: n characters (no spaces, all capitals)

#publiser helper function
def pub_clue(id,password,location,prediciton):
    formatted_string = f"{id},{password},{location},{prediciton}"
    score_pub.publish(formatted_string)

def pub_cmd_vel(x,z):
    move = Twist()
    move.angular.z = z
    move.linear.y = 0
    move.linear.x = x
    move_pub.publish(move)

class StateMachine:
    def __init__(self):
        self.current_state = "ROAD"
        self.drive_input = None
        self.pink_cooldown = False
        self.ped_xing = False
        self.state_data = None
        self.cv_image = None
        self.frame_counter = 0 
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=75, varThreshold=50, detectShadows=False)

    def road_state(self):
        # Behavior for driving on the road
        z = run_model(self.drive_input,interpreter_road,input_details_road,output_details_road)
        pub_cmd_vel(.5,z)

    def grass_state(self):
        # Behavior for driving on grass
        z = run_model(self.drive_input,interpreter_grass,input_details_grass,output_details_grass)
        pub_cmd_vel(.4,z)

    def yoda_state(self):
        # Behavior for driving on grass
        # z = run_model(self.drive_input,interpreter_grass,input_details_grass,output_details_grass)
        pub_cmd_vel(0,0)
    
    def mountain_state(self):
        z = run_model(self.drive_input,interpreter_mountain,input_details_mountain,output_details_mountain)
        pub_cmd_vel(.4,z)

    def pedestrian_state(self):
        fg_mask = self.bg_subtractor.apply(self.state_data)
        # Apply additional morphological operations to clean the mask (optional)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours and filter based on size (area)
        filtered_cnts = [contour for contour in contours if cv2.contourArea(contour) > 50]
        for cnts in filtered_cnts:
            print(cv2.contourArea(cnts))
            # Draw bounding rectangle or perform further processing on the detected object
            x, y, w, h = cv2.boundingRect(cnts)
            cv2.rectangle(self.cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("states", self.cv_image)
        cv2.waitKey(1)
        if filtered_cnts:
            self.frame_counter = 0
        else:
            self.frame_counter +=1 
            print(self.frame_counter)
        if self.frame_counter > 15:
            self.frame_counter = 0
            holder = self.current_state
            self.current_state = "ROAD"
            print(f'{holder} -------> {self.current_state}')
             

    def vehicle_state(self):
        z = run_model(self.drive_input,interpreter_mountain,input_details_mountain,output_details_mountain)
        pub_cmd_vel(.4,z)


    def event_occurred(self, event):
        # Callback function to handle the event
        if event == "PINK" and not self.pink_cooldown:
            print(event)
            self.pink_cooldown = True
            rospy.Timer(rospy.Duration(3), self.reset_pink_cooldown, oneshot=True)
            holder = self.current_state
            if holder == "ROAD": 
                self.current_state = "GRASS"
                self.road_state()
            elif holder == "GRASS": 
                self.current_state = "YODA"
                self.yoda_state()
            elif holder == "YODA": 
                self.current_state = "MOUNTAIN"
                self.mountain_state()
            print(f'{holder} -------> {self.current_state}')
        
        if event == "RED" and not self.ped_xing:
            pub_cmd_vel(0,0)
            print(event)
            self.ped_xing = True
            holder = self.current_state
            if holder == "ROAD": 
                self.current_state = "PEDESTRIAN"
            print(f'{holder} -------> {self.current_state}')
            
            
    def reset_pink_cooldown(self, event):
        self.pink_cooldown = False
        
            


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

state_machine = StateMachine()


def camera_callback(data):
    lower_pink= np.array([140,80,105])
    upper_pink= np.array([153,255,255])
    lower_red= np.array([0,42,102])
    upper_red= np.array([9,255,255])
    lower_blue = np.array([110,50,50])
    upper_blue= np.array([130,255,255])
    lower_ped = np.array([98,20,22])
    upper_ped = np.array([110,255,116])
    lower_blue = np.array([110,50,50])
    upper_blue= np.array([130,255,255])
    lower_vehicle = np.array([0,0,93])
    upper_vehicle = np.array([10,10,249])
    try:
        cv_image = br.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)

    # process image for NN
    im_grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    dim = (160, 90)
    resized = cv2.resize(im_grey, dim, interpolation = cv2.INTER_AREA)
    img = resized / 255.0 
    state_machine.drive_input = img

    # image masking for line and clue detection 
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_white = cv2.bitwise_not(mask_blue)
    mask_ped = cv2.inRange(hsv, lower_ped, upper_ped)
    mask_vehicle = cv2.inRange(hsv, lower_vehicle, upper_vehicle)

    
    if cv2.countNonZero(mask_pink) > 20000:
            state_machine.event_occurred("PINK")

    if cv2.countNonZero(mask_red) > 30000:
            state_machine.event_occurred("RED")
    state_machine.cv_image = cv_image
      # resize image
    

    if state_machine.current_state == "ROAD":
            state_machine.road_state()
    elif state_machine.current_state == "GRASS":
            state_machine.grass_state()
    elif state_machine.current_state == "PEDESTRIAN":
            state_machine.state_data = mask_ped
            state_machine.pedestrian_state()
            
    #pub_clue(id,password,-1,"NA")


def subscribe():
    rospy.Subscriber('/R1/pi_camera/image_raw',Image,camera_callback)
    rospy.Subscriber('/clock',Clock)
    rospy.sleep(1)

subscribe()

rate = rospy.Rate(10)

while not rospy.is_shutdown():
    pub_clue(id,password,0,"NA")
    rospy.spin()    
