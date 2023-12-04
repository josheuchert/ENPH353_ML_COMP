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
rospy.init_node('imitation_node')
move_pub = rospy.Publisher('/R1/cmd_vel', Twist, 
  queue_size=10)    
br = CvBridge()
rospy.sleep(1)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

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

        lower_ped = np.array([98,20,22])
        upper_ped = np.array([110,255,116])
        lower_blue = np.array([110,50,50])
        upper_blue= np.array([130,255,255])
        lower_vehicle = np.array([0,0,93])
        upper_vehicle = np.array([10,10,249])
        lower_yoda1 = np.array([165,10,14])
        upper_yoda1 = np.array([186,170,170])

        lower_yoda2 = np.array([0,0,0])
        upper_yoda2 = np.array([9,90,80])

        lower_car = np.array([0,0,93])
        upper_car = np.array([10,10,249])
        img_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        bi = cv2.bilateralFilter(cv_image, 5, 75, 75)
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        im_cut = cv_image[360:720,0:1280]
        im_grey = cv2.cvtColor(im_cut, cv2.COLOR_BGR2GRAY)
        hsv_cut = cv2.cvtColor(im_cut, cv2.COLOR_BGR2HSV)
        lower_road_hsv = np.array([0,0,79])
        upper_road_hsv= np.array([6,6,90])
        mask_road = cv2.inRange(hsv_cut, lower_road_hsv, upper_road_hsv)
        mask_ped = cv2.inRange(hsv, lower_ped, upper_ped)
        mask_vehicle = cv2.inRange(hsv, lower_vehicle, upper_vehicle)
        mask_yoda1 = cv2.inRange(hsv, lower_yoda1, upper_yoda1)
        mask_yoda2 = cv2.inRange(hsv, lower_yoda2, upper_yoda2)

        # Apply background subtraction to get the foreground mask
        fg_mask = bg_subtractor.apply(mask_yoda2)
        print(f'yoda2 cout; {cv2.countNonZero(mask_yoda2)}')

        # Apply additional morphological operations to clean the mask (optional)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Iterate through contours and filter based on size (area)
        filtered_cnts = [contour for contour in contours if cv2.contourArea(contour) > 500]
        for cnts in filtered_cnts:
            print(cv2.contourArea(cnts))
            # Draw bounding rectangle or perform further processing on the detected object
            x, y, w, h = cv2.boundingRect(cnts)
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)



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
        lower_pink= np.array([140,80,105])
        upper_pink= np.array([153,255,255])

        # Threshold the HSV image to get only blue colors
        mask_blue = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)
        mask_white = cv2.bitwise_not(mask_blue)

        if cv2.countNonZero(mask_pink) > 1000:
            print("pink line!")

        # threshold = 180
        # _, binary = cv2.threshold(img_gray,threshold,255,cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(mask_blue, cnts, (255,255,255))
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        mask_clue = cv2.bitwise_and(mask_blue,mask_white)
        num_white_pixels = cv2.countNonZero(mask_clue)
        cnts, _ = cv2.findContours(mask_clue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_cnts = [contour for contour in cnts if cv2.contourArea(contour) > 1000]
        #print(cv2.countNonZero(num_white_pixels))
        print(f'clue cout; {cv2.countNonZero(mask_clue)}')
        if num_white_pixels > 12500 and filtered_cnts:
            for c in filtered_cnts:
                epsilon = 0.08 * cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, epsilon, True)

                #cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 2)
            # print(approx)
        #     dst = cv2.cornerHarris(mask_clue,2,3,0.04)
        #     ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
        #     dst = np.uint8(dst)
        #     ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
        #     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
        #     corners = cv2.cornerSubPix(mask_clue,np.float32(centroids),(5,5),(-1,-1),criteria)
            
        #     #Now draw them
        #     src =corners[1:]
        #     #print(src)
            

            # Rearrange the corners based on the assigned indicesght

            centroid = np.mean(approx, axis=0)[0]
            print(centroid)
            print(centroid.shape)
            #get x cord of centroid 
            x = centroid[0]
            #get y cord of centroid
            y = centroid[1]
            print(x)
            print(y)
            print(approx)
            def sort_key(point):
                angle = np.arctan2(point[0][1] - centroid[1], point[0][0] - centroid[0])
                return (angle + 2 * np.pi) % (2 * np.pi)
            # Sort the source points based on their relative positions to match the destination points format
            sorted_approx = sorted(approx, key=sort_key)
            sorted_approx = np.array(sorted_approx)

        #     # Reorder 'src' points to match the 'dest' format
            # approx
            print(sorted_approx)
            src = np.array([sorted_approx[2], sorted_approx[3], sorted_approx[1], sorted_approx[0]], dtype=np.float32)
            

            width = 600
            height= 400
            dest = np.float32([[0, 0],
                        [width, 0],
                        [0, height],
                        [width , height]])

            M = cv2.getPerspectiveTransform(src,dest)
            clue = cv2.warpPerspective(cv_image,M,(width, height),flags=cv2.INTER_LINEAR)
            cv2.imshow("clue", clue)

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
            cv2.imshow("result", result)

            
            # for corner in src:
            #     x, y = int(corner[0]), int(corner[1])
            #     cv2.circle(cv_image,(x,y), 2, (0,255,0), -1)  # -1 signifies filled circle
        
        # # Get the largest contour (assumed to be the white line)
        # contours, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # largest_contour = max(contours, key=cv2.contourArea)
        
        # # Get orientation of the line using its bounding rectangle
        # rect = cv2.minAreaRect(largest_contour)
        # angle = rect[2]
        # if angle>45:
        #     angle = angle-90
        # # Rotate the robot to align with the line (example, adjust as needed)
        # # Your robot control logic here to adjust orientation based on 'angle'
        # # For simulation purposes, let's print the angle
        # print("Angle to straighten:", angle)
        # if 20>angle>1:
        #     print("RIGHT")

        # elif -20<angle <-1:
        #     print("LEFT")
        # elif -1<angle<1:
        #     print ("STRAIGHT")
        
        #cv2.imshow("gray", mask_yoda2)
        cv2.imshow("cut", mask_clue)
        #cv2.imshow("hsv", cv_image)
        # cv2.imshow("blue", mask_blue)
        # cv2.imshow("out", result)
        # cv2.imshow("white", mask_white)
        # cv2.imshow("image", mask_clue)
        # cv2.imshow("done", result)

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