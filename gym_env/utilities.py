#!/home/zgc/miniconda3/envs/openmmlab/bin/python
#coding=utf-8
import rospy
from sensor_msgs.msg import Image
from mavros_msgs.srv import CommandBool, SetMode
from cv_bridge import CvBridge
import threading
import math
from ultralytics import YOLO
from message_filters import Subscriber, ApproximateTimeSynchronizer

# MAVROS msgs to use setpoints
from geometry_msgs.msg import Point, PoseStamped, Twist, TwistStamped
from mavros_msgs.msg import *

# MAVROS srv to change modes
from mavros_msgs.srv import CommandBool, SetMode
import numpy as np

# Math functions
from math import sin, cos, sqrt

# ____________________________________________________ Classes ____________________________________________________ #
# Flight modes class
class Drone_Modes:

    def __init__(self):
        pass

    def setArm(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            armService = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
            armService(True)
        except rospy.ServiceException as e:
            print("Service arm call failed: %s"%e)

    def setOffboardMode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flightModeService = rospy.ServiceProxy('mavros/set_mode', SetMode)
            flightModeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("service set_mode call failed: %s. Offboard Mode could not be set."%e)

# Controller class
class Drone_Controller:

    def __init__(self):

        #           Setpoint message for position control
        # ---------------------------------------------------------
        ## Message
        self.sp_pos = PoseStamped()
        # ---------------------------------------------------------
        self.sp_vel = TwistStamped()
        # ---------------------------------------------------------

        # Drone state
        self.state = State()

        # Drone modes
        self.modes = Drone_Modes()

        # Updating rate
        self.rate = rospy.Rate(20)

        # Setpoint_raw publisher for position control
        self.sp_raw_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)

        # Setpoint_velocity publisher for velocity control
        self.sp_vel_pub = rospy.Publisher('mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)

        # Added: Body frame velocity publisher
        self.sp_body_vel_pub = rospy.Publisher('/mavros/setpoint_raw/local', PositionTarget, queue_size=10)

        # Subscribe to drone state
        rospy.Subscriber('mavros/state', State, self.stateCb)

        # Subscribe to drone's local position
        # rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.posCb)


    # ____________ Utility functions____________

    ## Publish target position
    def pubPosition(self):
        self.sp_raw_pub.publish(self.sp_pos)
        self.rate.sleep()

    # Added method: Publish body frame velocity
    def pubBodyVelocity(self, vx, vy, vz, wz):
        vel_msg = PositionTarget()
        vel_msg.header.stamp = rospy.Time.now()

        # Set to body frame (FRAME_BODY_NED)
        vel_msg.coordinate_frame = PositionTarget.FRAME_BODY_NED

        # Ignore position and acceleration, only send velocity and yaw_rate
        vel_msg.type_mask = PositionTarget.IGNORE_PX | \
                            PositionTarget.IGNORE_PY | \
                            PositionTarget.IGNORE_PZ | \
                            PositionTarget.IGNORE_AFX | \
                            PositionTarget.IGNORE_AFY | \
                            PositionTarget.IGNORE_AFZ | \
                            PositionTarget.IGNORE_YAW

        # Set velocity in body frame
        vel_msg.velocity.x = vx
        vel_msg.velocity.y = vy
        vel_msg.velocity.z = vz

        # Set yaw rate (around body z-axis)
        vel_msg.yaw_rate = wz

        # Publish velocity command
        self.sp_body_vel_pub.publish(vel_msg)
        self.rate.sleep()
    ## Publish target position
    def pubstart_position(self, start_position):
        self.sp_raw_pub.publish(start_position)
        self.rate.sleep()

    # Publish velocity
    def pubVelocity(self, action):
        self.sp_vel_pub.publish(action)
        self.rate.sleep()

    ## Arm the drone if not armed
    def armDrone(self):
        if not self.state.armed:
            while not self.state.armed:
                self.modes.setArm()
                self.rate.sleep()

    ## Activate OFFBOARD Mode by sending a few setpoints
    def activateOffboard(self):
        for _ in range(10):
            self.sp_raw_pub.publish(self.sp_pos)
            self.rate.sleep()
        self.modes.setOffboardMode()

    ## Drone State callback
    def stateCb(self, msg):
        self.state = msg

# ____________________________________________________ Classes ____________________________________________________ #
class DepthCameraProcessor:
    """Class for processing depth camera data"""
    def __init__(self, yolo_model_path, score_threshold=0.5):
        self.model = YOLO(yolo_model_path)
        self.model.conf = score_threshold
        self.key_points = (0, 0, 0, 0)
        self.depth_value = None
        self.prev_depth_value = None # Save previous depth value
        self.prev_rect_original = None # Save previous target position info
        self.rect_original = None
        self.rgb_image = None
        self.depth_image = None
        self.result_ready = threading.Event()
        self.bridge = CvBridge()
        self.thread = None  # Save thread reference
        self.threshold = 5  # Depth value jump threshold
        self.num_yichang = 0 # Number of anomalies

    def voc2dota(self, cx, cy, w, h, angle):
        """Convert to rotated bounding box format"""
        angle = float(angle)  # Ensure scalar
        p0x, p0y = self.rotate_point(cx, cy, cx - w / 2, cy - h / 2, -angle)
        p1x, p1y = self.rotate_point(cx, cy, cx + w / 2, cy - h / 2, -angle)
        p2x, p2y = self.rotate_point(cx, cy, cx + w / 2, cy + h / 2, -angle)
        p3x, p3y = self.rotate_point(cx, cy, cx - w / 2, cy + h / 2, -angle)
        return p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y

    def rotate_point(self, xc, yc, xp, yp, theta):
        """Rotate point"""
        xoff = xp - xc
        yoff = yp - yc
        cos_theta = math.cos(float(theta))  # Ensure theta is scalar
        sin_theta = math.sin(float(theta))
        px = cos_theta * xoff + sin_theta * yoff
        py = -sin_theta * xoff + cos_theta * yoff
        return xc + px, yc + py


    def synchronized_callback(self, rgb_msg, depth_msg):
        """Synchronized callback function to process camera images"""
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            results = self.model.predict(self.rgb_image, classes=[0], verbose=False)

            # Check if obb.xywhr is empty
            if results[0].obb.xywhr is not None and results[0].obb.xywhr.shape[0] > 0:
                # If targets exist, select the first one
                angle = results[0].obb.xywhr[0, 4].item()  # Extract scalar
                angle = math.degrees(angle)  # Convert radians to degrees
                h = results[0].obb.xywhr[0, 3].item()
                w = results[0].obb.xywhr[0, 2].item()
                y = results[0].obb.xywhr[0, 1].item()
                x = results[0].obb.xywhr[0, 0].item()

                p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y = self.voc2dota(x, y, w, h, angle)
                x_min, x_max = min(p0x, p1x, p2x, p3x), max(p0x, p1x, p2x, p3x)
                y_min, y_max = min(p0y, p1y, p2y, p3y), max(p0y, p1y, p2y, p3y)
                self.key_points = (x_min, x_max, y_min, y_max)

                if w < h:
                    mid_w = w
                    w = h
                    h = mid_w
                    angle = 90 + angle

                self.rect_original = ((x, y), (w, h), angle)
                # Check if depth_region is empty
                depth_region = self.depth_image[int(y_min):int(y_max), int(x_min):int(x_max)]
                if depth_region.size > 0:
                    self.depth_value = np.min(depth_region)
                else:
                    self.depth_value = None

                self.result_ready.set()
            else:
                self.rect_original = None
                self.depth_value = None
        except Exception as e:
            rospy.logerr(f"Error in synchronized_callback: {e}")

    def get_target_info(self, timeout=None):
        if self.result_ready.wait(timeout):
            self.result_ready.clear()
            if self.rect_original is not None and self.depth_value is not None:
                flag_yichang = False
                # Check if current depth_value has a sudden jump compared to previous
                if self.prev_depth_value is not None and abs(self.depth_value - self.prev_depth_value) > self.threshold:
                    # If abnormal jump, return previous depth/position info
                    depth = self.prev_depth_value
                    rect = self.prev_rect_original
                    self.num_yichang += 1   # Increment anomaly count
                    flag_yichang = True
                else:
                    # Otherwise use current depth/position info
                    depth = self.depth_value
                    rect = self.rect_original

                # Save current depth/position info for next comparison
                self.prev_depth_value = depth
                self.prev_rect_original = rect
                # Limit self.prev_depth_value range to less than 7
                if self.prev_depth_value > 7:
                    self.prev_depth_value = 7
                elif self.prev_depth_value < 0:
                    self.prev_depth_value = 0

                # Retrieve additional info
                x, y = rect[0]
                w, h = rect[1]
                original_angle = rect[2]  # Original angle (degrees)
                # Print anomaly count
                if flag_yichang:
                    rospy.loginfo(f"Target info: x={x}, y={y}, w={w}, h={h}, angle={original_angle}, depth={depth}, anomalies={self.num_yichang}")
                return x, y, original_angle, depth
        return None

    def run(self):
        """Run node"""
        self.rgb_sub = Subscriber("/iris_downward_depth_camera/camera/rgb/image_raw", Image)
        self.depth_sub = Subscriber("/iris_downward_depth_camera/camera/depth/image_raw", Image)
        self.ats = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], queue_size=30, slop=0.05)
        self.ats.registerCallback(self.synchronized_callback)

        self.thread = threading.Thread(target=rospy.spin)
        self.thread.start()


    def stop(self):
        """Stop all activities of DepthCameraProcessor"""
        rospy.loginfo("Stopping DepthCameraProcessor...")
        self.result_ready.clear()  # Clear event
        rospy.signal_shutdown("Shutting down DepthCameraProcessor")  # Stop ROS node
