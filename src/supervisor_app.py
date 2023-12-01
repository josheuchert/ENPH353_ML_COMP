#! /usr/bin/env python3
from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi
import cv2
import sys
import rospy
import os 
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rosgraph_msgs.msg import Clock
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np
import tensorflow as tf
# import keras
# from keras.models import load_model

# drive_model = load_model('full.h5',compile=False)
# drive_model.compile(
#                    optimizer='adam',
#                    loss=keras.losses.MeanSquaredError()
#                   )
# drive_model.summary()


interpreter = tf.lite.Interpreter(model_path='quantized_model_road.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

def denormalize_value(normalized_value, min_val, max_val):
    return (normalized_value * (max_val - min_val)) + min_val

def run_model(img):
    dim = (160, 90)
    img = np.array(img)  # Convert to NumPy array
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    img = resized / 255.0 
    img_aug = img.reshape((1, 90, 160, 1)).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], img_aug)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    output = denormalize_value(output, -2, 2)
    print(output)
    return output


parent_dir = '/home/fizzer/training'

class ROSHandler(QObject):
    image_signal1 = pyqtSignal(object)
    image_signal2 = pyqtSignal(object)
    

    def __init__(self):
        super().__init__()
        self.cur_state = None
        self.prev_state = None
        self.is_saving = False
        self.counter = 0
        self.path = ''
        self.controlled = False
        self.executed_once = False
        rospy.init_node('imitation_node')
        rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/R1/cmd_vel', Twist, self.move_callback)
        self.move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
        queue_size=10)  


    def get_clue(self, cv_image):
        bi = cv2.bilateralFilter(cv_image, 5, 75, 75)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lv = 50
        uh = 130
        us = 255
        uv = 255
        lh = 110
        ls = 50
        lv = 50
        lower_hsv = np.array([lh,ls,lv])
        upper_hsv = np.array([uh,us,uv])
        lower_white = np.array([0,0,90])
        upper_white = np.array([10,10,110])

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_white = cv2.bitwise_not(mask_blue)
        # threshold = 180
        # _, binary = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(mask_blue, cnts, (255,255,255))
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        mask_clue = cv2.bitwise_and(mask_blue,mask_white)
        num_white_pixels = cv2.countNonZero(mask_blue)
        cnts, _ = cv2.findContours(mask_clue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        print(num_white_pixels)
        if num_white_pixels > 3000:
            for c in filtered_cnts:
                epsilon = 0.08 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)
                #cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 2)
            print(approx)
            src = np.array([approx[0], approx[3], approx[1], approx[2]], dtype=np.float32)
            width = 600
            height= 400
            dest = np.float32([[0, 0],
                        [width, 0],
                        [0, height],
                        [width , height]])

            M = cv2.getPerspectiveTransform(src,dest)
            clue = cv2.warpPerspective(cv_image,M,(width, height),flags=cv2.INTER_LINEAR)

            gray_clue = cv2.cvtColor(clue, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_clue = clahe.apply(gray_clue)
            #gray_clue = cv2.equalizeHist(gray_clue)
            # perform threshold
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpen = cv2.filter2D(gray_clue, -1, sharpen_kernel)

            # # remove noise / close gaps
            # kernel =  np.ones((5,5),np.uint8)
            # noise = cv2.morphologyEx(sharpen, cv2.MORPH_CLOSE, kernel)

            # # dilate result to make characters more solid
            # kernel2 =  np.ones((3,3),np.uint8)
            # dilate = cv2.dilate(noise,kernel2,iterations = 1)

            retr, mask2 = cv2.threshold(gray_clue, 100, 255, cv2.THRESH_BINARY_INV)
            # #45 equihist

            

            # #invert to get black text on white background
            result = cv2.bitwise_not(mask2)
            return result
        return cv_image
            
    

    def camera_callback(self, data):
        br = CvBridge()
        try:
            cv_image = br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        im_cut = cv_image[360:720,0:1280]
        im_grey = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        threshold = 180
        _, binary = cv2.threshold(im_grey,threshold,255,cv2.THRESH_BINARY)

        img = np.array(im_grey)

        #clue = self.get_clue(cv_image)
        self.image_signal1.emit(cv_image)
        #self.image_signal2.emit(clue)
        
        if self.controlled is False:
            # img_aug = np.expand_dims(img, axis=0)
            z_predict = run_model(im_grey)
            NN_move = Twist()
            NN_move.angular.z = z_predict
            NN_move.linear.y = 0
            NN_move.linear.x = .5
            self.move_pub.publish(NN_move)
        else: 
            if not self.executed_once:
                reset_move = Twist()
                reset_move.angular.z = 0
                reset_move.linear.y = 0
                reset_move.linear.x = 0
                self.move_pub.publish(reset_move)
                self.executed_once = True
                print("stopped")

        if self.is_saving:
            if self.cur_state != 0:
                filename = f'#{self.counter}_'+self.cur_state+'.jpg'
                # Write the image using the specified filename
                cv2.imwrite(filename, binary)
                self.counter += 1

    
    def move_callback(self, data):
        linear_x = data.linear.x
        angular_z = data.angular.z
        self.prev_state = self.cur_state
        self.cur_state = f"{linear_x},{angular_z}"
    
    def connect_update_signal(self, slot_function):
        self.update_is_supervising.connect(slot_function)



class My_App(QtWidgets.QMainWindow):
	
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./supervisor.ui", self)

        self._is_saving = False
        self.controlled = False
        self.ros_handler = ROSHandler()  # Instantiate ROSHandler
        self.toggle_control.clicked.connect(self.SLOT_toggle_control)
        self.toggle_supervising.clicked.connect(self.SLOT_toggle_supervising)
        br = CvBridge()
        self.ros_handler.image_signal1.connect(self.update_image1)
        self.ros_handler.image_signal2.connect(self.update_image2)

    def SLOT_toggle_supervising(self):
        if self._is_saving:
            self.toggle_supervising.setText("&Start supervising")
            self.ros_handler.is_saving = False
            self._is_saving = False
            self.ros_handler.counter = 0
            self.ros_handler.path = ''
        else:
            self._is_saving = True
            self.toggle_supervising.setText("&Stop supervising")
            self.ros_handler.is_saving = True
            path = os.path.join(parent_dir, self.folder_name.text())
            self.ros_handler.path = path
            os.mkdir(path, mode = 0o777)
            os.chdir(path)


    def SLOT_toggle_control(self):
        if self.controlled:
            self.controlled = False
            self.ros_handler.executed_once = False
            self.toggle_control.setText("&Take Control")
            self.ros_handler.controlled = False
        else:
            self.controlled = True
            self.toggle_control.setText("&Relinquish control")
            print("give back control")
            self.ros_handler.controlled = True 
            
            

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)
    
    def update_image1(self, cv_image):
        
        pixmap = self.convert_cv_to_pixmap(cv_image)
        scaled_pixmap = pixmap.scaled(self.live_image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.live_image_label.setPixmap(pixmap)

    def update_image2(self, cv_image):
        pixmap = self.convert_cv_to_pixmap(cv_image)
        scaled_pixmap = pixmap.scaled(self.live_image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.live_image_label_2.setPixmap(pixmap)
    

app = QtWidgets.QApplication(sys.argv)
myApp = My_App()
myApp.show()
sys.exit(app.exec_())

