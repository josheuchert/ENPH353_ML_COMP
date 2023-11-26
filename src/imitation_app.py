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

parent_dir = '/home/fizzer/training'

class ROSHandler(QObject):
    image_signal = pyqtSignal(object)
    

    def __init__(self):
        super().__init__()
        self.cur_state = None
        self.prev_state = None
        self.is_saving = False
        self.counter = 0
        self.path = ''
        rospy.init_node('imitation_node')
        rospy.Subscriber('/R1/pi_camera/image_raw', Image, self.camera_callback)
        rospy.Subscriber('/R1/cmd_vel', Twist, self.move_callback)

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

        out = cv2.putText(binary,self.cur_state, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
        self.image_signal.emit(out)

        if self.is_saving:
            if self.cur_state is not None:
                filename = f'#{self.counter}_'+self.cur_state+'.jpg'
                # Write the image using the specified filename
                cv2.imwrite(filename, out)
                self.counter += 1
    
    def move_callback(self, data):
        linear_x = data.linear.x
        angular_z = data.angular.z
        self.prev_state = self.cur_state
        self.cur_state = f"{linear_x},{angular_z}"
    
    def connect_update_signal(self, slot_function):
        self.update_is_saving.connect(slot_function)



class My_App(QtWidgets.QMainWindow):
	
    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./imitate.ui", self)
        self.ros_handler = ROSHandler()  # Instantiate ROSHandler
        self._is_saving = False
        self.toggle_saving.clicked.connect(self.SLOT_toggle_saving)
        br = CvBridge()

    def SLOT_toggle_saving(self):
        if self._is_saving:
            self.toggle_saving.setText("&Start imitation")
            self.ros_handler.is_saving = False
            self._is_saving = False
            self.ros_handler.counter = 0
            self.ros_handler.path = ''
        else:
            self._is_saving = True
            self.toggle_saving.setText("&Stop imitation")
            
            self.ros_handler.is_saving = True
            path = os.path.join(parent_dir, self.folder_name.text())
            self.ros_handler.path = path
            os.mkdir(path, mode = 0o777)
            os.chdir(path)
            self.ros_handler.image_signal.connect(self.update_image)
            

    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)
    
    def update_image(self, cv_image):
        pixmap = self.convert_cv_to_pixmap(cv_image)
        scaled_pixmap = pixmap.scaled(self.live_image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.live_image_label.setPixmap(pixmap)
    

app = QtWidgets.QApplication(sys.argv)
myApp = My_App()
myApp.show()
sys.exit(app.exec_())