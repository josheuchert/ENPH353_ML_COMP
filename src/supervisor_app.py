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
import keras
from keras.models import load_model

drive_model = load_model('test5.h5',compile=False)
drive_model.compile(
                   optimizer='adam',
                   loss=keras.losses.MeanSquaredError()
                  )
drive_model.summary()

def denormalize_value(normalized_value, min_val, max_val):
    return (normalized_value * (max_val - min_val)) + min_val


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
        lower = np.array([90, 0, 0], dtype="uint8")
        upper = np.array([120, 10, 10], dtype="uint8")
        mask_blue = cv2.inRange(bi, lower, upper)
        mask_white = cv2.bitwise_not(mask_blue)
        # threshold = 180
        # _, binary = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 6000]
        if filtered_cnts:
            mask_blue[:] = 0
            cv2.fillPoly(mask_blue, filtered_cnts, (255,255,255))
            clue_mask = cv2.bitwise_and(mask_blue,mask_blue,mask=mask_white)
            cnts, _ = cv2.findContours(clue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.fillPoly(clue_mask, cnts, (255,255,255))
            dst = cv2.cornerHarris(clue_mask,2,3,0.04)
            ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
            dst = np.uint8(dst)
            ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(clue_mask,np.float32(centroids),(5,5),(-1,-1),criteria)
            #print(corners)
            #Now draw them
            src =corners[1:]
            
            if len(src) is 4:
                # Rearrange the corners based on the assigned indicesght
                centroid = corners[0]
                def sort_key(point):
                    angle = np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
                    return (angle + 2 * np.pi) % (2 * np.pi)
                # Sort the source points based on their relative positions to match the destination points format
                sorted_src = sorted(src, key=sort_key)
                sorted_src = np.array(sorted_src)

                # Reorder 'src' points to match the 'dest' format
                src = np.array([sorted_src[2], sorted_src[3], sorted_src[1], sorted_src[0]], dtype=np.float32)
                #print(src)

                width = 600
                height= 400
                dest = np.float32([[0, 0],
                            [width, 0],
                            [0, height],
                            [width , height]])

                M = cv2.getPerspectiveTransform(src,dest)
                clue = cv2.warpPerspective(cv_image,M,(width, height),flags=cv2.INTER_LINEAR)

                gray_clue = cv2.cvtColor(clue, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
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
                #45 equihist

                #invert to get black text on white background
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
        im_grey = img_gray = cv2.cvtColor(im_cut, cv2.COLOR_BGR2GRAY)
        threshold = 180
        _, binary = cv2.threshold(im_grey,threshold,255,cv2.THRESH_BINARY)

        img = np.array(binary)  # Convert to NumPy array
      # resize image
        dim = (200, 66)
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = resized / 255.0  # Normalize to range [0, 1] (assuming RGB images

        clue = self.get_clue(cv_image)
        self.image_signal1.emit(binary)
        self.image_signal2.emit(clue)
        
        if self.controlled is False:
            img_aug = np.expand_dims(img, axis=0)
            z_predict = drive_model.predict(img_aug)
            z_predict= denormalize_value(z_predict, -1 , 1)
            NN_move = Twist()
            NN_move.angular.z = z_predict*1.4
            NN_move.linear.y = 0
            NN_move.linear.x = .25
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
            if self.cur_state is not None:
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

