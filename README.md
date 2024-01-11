# ENPH353 Machine Learning Competition Repository 
### Competition Goal
- Train an agent in a simulated Gazebo environment to traverse the environment and reach the top of the mountain, avoid obstacles, and find clues to solve the crime.
###

| **Skills learned and tools used**              |
|---
| ROS, Linux, Pytorch, OpenCV, Data analysis |

| Sample clueboard | Course |
|----------------------|--------|
|![image](https://github.com/Itaiboss/ENPH353_ML_COMP/assets/90986809/81d7732c-f96a-4bba-856d-a483efe7b932)| <img width="740" alt="Screenshot 2024-01-10 at 6 59 36 PM" src="https://github.com/Itaiboss/ENPH353_controller/assets/90986809/8406af75-e4c8-4dde-8e89-fd3da348be84">||


for an in depth analysis of our approach and implementation read our paper [Fizz Detective Report - Winux.pdf](https://github.com/Itaiboss/ENPH353_controller/files/13897461/Fizz.Detective.Report.-.Winux.pdf). 


### Our Solution 
| Driving | Clue Detection |
|----------------------|--------|
| <ul><li> Wrote utility scripts to record thousands of images from the camera feed and label with corresponding joystick inputs as we navigated the course manually</li><li>Designed model architecture and trained various models on recorded data</li><li>Used OpenCV with background subtraction, masking and other classical techniques to detect obstacles and moving NPC's</li><li>Designed a finite state machine controller in ROS</li> </ul> | <ul><li> Used homography and perspective transforms with OpenCV to align clueboards and homogenous letter cutouts</li><li>Generated randomized blurred/cut letter images. Recorded clueboards during driving runs and cut out letters for a mixed dataset of generated and recorded images</li><li>Trained a CNN to classify letters</li> </ul> |


Below is a video of a sample run

https://github.com/Itaiboss/ENPH353_controller/assets/90986809/14094959-f2cb-4067-b280-243dc0286fe2

### Results 
Placed 1st out of 17 teams with a perfect driving and clue detection score. We were the only imitation learning based team that completed a full run. We also had the fastest lap time overall. 


