#!/home/zgc/miniconda3/envs/openmmlab/bin/python
# coding=utf-8
import gym
from gym import spaces
import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
import numpy as np
from gym.envs.classic_control.utilities import Drone_Modes, DepthCameraProcessor, Drone_Controller, LightweightEKF
from gym.envs.registration import register
import math
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from gym.utils import seeding

Wind_Max = 5.0

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self._np_random = None

        # ==============================
        # 1. Environment Hyperparameters
        # ==============================
        self._max_episode_steps = 800  # Max steps per episode
        self.current_step = 0  # Step counter
        self.target_distance = 0.5  # Distance threshold for reset
        self.success_count_total = 0  # Success count
        self.last_observation = None  # Store previous normalized observation

        # Sliding window statistics
        self._window_size = 30
        self._episode_in_window = 0
        self._success_in_window = 0

        # Docking success thresholds (original units)
        self.x_center_threshold = 80  # ±50
        self.y_center_threshold = 80  # ±20
        self.angle_threshold = 10  # ±10 degrees
        self.safe_distance = 1.0  # Ideal docking distance

        # Per-step penalty factor
        self.episode_penalty_factor = 0.05

        self.total_episodes = 0.0

        # Terminal condition counters: "lost_target", "out_of_bound", "max_steps", "success"
        self.terminal_counts = {"lost_target": 0, "out_of_bound": 0, "max_steps": 0, "success": 0}

        # Actual speed thresholds
        self.linear_speed_threshold_actual = 0.075  # Linear speed threshold (m/s)
        self.angular_speed_threshold_actual = 0.05  # Angular speed threshold (rad/s)
        self.speed_penalty_factor = 50.0  # Speed penalty factor
        # Action smoothing factor
        self.lambda_smooth = 0.02  # Tunable hyperparameter

        # ==============================
        # 2. ROS Initialization
        # ==============================
        self.current_state = None
        self.state_sub = rospy.Subscriber('/mavros/state', State, self.state_cb)

        # ---------- Wind speed subscription ----------
        self.wind_world = np.zeros(3, dtype=np.float32)
        rospy.Subscriber("/wind_velocity", Vector3Stamped, self._wind_cb, queue_size=1)

        # Define action space (retain original settings)
        self.action_space = spaces.Box(
            low=np.array([-0.15, -0.15, -0.15, -0.1]),
            high=np.array([0.15, 0.15, 0.15, 0.1]),
            dtype=np.float32
        )

        self.obs_low = np.array([0.0, 0.0, 0.0, 0.0, -3.14, -0.15, -0.15, -0.20, -0.1, -Wind_Max, -Wind_Max, -Wind_Max], dtype=np.float32)
        self.obs_high = np.array([848.0, 848.0, 180.0, 8.0, 3.14, 0.15, 0.15, 0.20, 0.1, Wind_Max, Wind_Max, Wind_Max], dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.obs_low.shape, dtype=np.float32)

        # Normalize to [-1, 1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=self.obs_low.shape, dtype=np.float32)

        # ==============================
        # 3. Initialize Drone Modes / Controller
        # ==============================
        self.drone_modes = Drone_Modes()
        self.controller_object = Drone_Controller()

        # ==============================
        # 4. Initialize YOLO Model
        # ==============================
        self.yolo_model_path = "/home/zgc/software/ultralytics-main0414/ultralytics/runs/obb/train3/weights/best.pt"
        self.depth_processor = DepthCameraProcessor(self.yolo_model_path)
        self.depth_processor.run()

        # Arm + Offboard
        print("Initializing drone...")
        print("Arming the drone...")
        self.controller_object.armDrone()
        print("Activating Offboard...")
        self.controller_object.activateOffboard()

        # Previous shaping value for differential reward calculation (normalized)
        self.prev_shaping = None
        self.prev_action = None  # Clear previous action on reset

    def _wind_cb(self, msg: Vector3Stamped):
        """Save the latest wind speed (world frame)."""
        self.wind_world[0] = msg.vector.x
        self.wind_world[1] = msg.vector.y
        self.wind_world[2] = msg.vector.z

    def get_raw_observation(self, uav_state):
        x, y, theta, distance = uav_state
        yaw, vx, vy, vz, wz = self.get_ekf_data(timeout=5.0)
        raw_obs = np.array([x, y, theta, distance, yaw, vx, vy, vz, wz], dtype=np.float32)
        raw_obs = np.concatenate([raw_obs, self.wind_world], dtype=np.float32)
        if len(raw_obs) != 12 or np.any(np.isnan(raw_obs)) or np.any(np.isinf(raw_obs)):
            rospy.logwarn("Invalid raw observation. Resetting environment.")
            return self.reset()[0]
        return raw_obs

    def normalize_observation(self, raw_obs):
        """
        Normalize raw observation to [-1, 1]
        norm = 2*(raw - low)/(high - low) - 1
        """
        norm_obs = 2.0 * (raw_obs - self.obs_low) / (self.obs_high - self.obs_low) - 1.0
        return norm_obs.astype(np.float32)

    def get_observation(self, uav_state):
        raw_obs = self.get_raw_observation(uav_state)  # raw_obs is 12D
        print("Raw observation:", raw_obs)
        return self.normalize_observation(raw_obs)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        else:
            self._np_random, seed = seeding.np_random(None)
        self.current_step = 0
        self.prev_shaping = None  # Clear previous shaping value on reset
        self.prev_action = None  # Clear previous action on reset
        max_retries = 20
        retries = 0
        while retries < max_retries:
            try:
                if not self.current_state.armed:
                    rospy.logwarn("Drone motors off. Re-arming.")
                    self.controller_object.armDrone()
                if self.current_state.mode != "OFFBOARD":
                    self.controller_object.activateOffboard()
                target_position = PoseStamped()
                target_position.header.stamp = rospy.Time.now()
                target_position.pose.position.x = np.random.uniform(13.5, 15.5)
                target_position.pose.position.y = np.random.uniform(-15.5, -13.5)
                target_position.pose.position.z = np.random.uniform(17.6, 17.9)  # Suggest larger initial distance
                angle_z = np.random.uniform(0, 360)
                # Print initial position and angle
                print(f"Initial position: x={target_position.pose.position.x}, "
                      f"y={target_position.pose.position.y}, z={target_position.pose.position.z}, angle_z={angle_z}°")
                angle_z_rad = math.radians(angle_z)
                target_position.pose.orientation.x = 0.0
                target_position.pose.orientation.y = 0.0
                target_position.pose.orientation.z = math.sin(angle_z_rad / 2)
                target_position.pose.orientation.w = math.cos(angle_z_rad / 2)
                position_reached = False
                yaw_tolerance_rad = math.radians(50.0)
                rate = rospy.Rate(50)
                for _ in range(1000):
                    if not self.current_state.armed:
                        self.controller_object.armDrone()
                    if self.current_state.mode != "OFFBOARD":
                        self.controller_object.activateOffboard()
                    current_pose = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped, timeout=5.0)
                    current_x = current_pose.pose.position.x
                    current_y = current_pose.pose.position.y
                    current_z = current_pose.pose.position.z
                    distance = math.sqrt(
                        (current_x - target_position.pose.position.x) ** 2 +
                        (current_y - target_position.pose.position.y) ** 2 +
                        (current_z - target_position.pose.position.z) ** 2
                    )
                    yaw, vx, vy, vz, wz = self.get_ekf_data(timeout=5.0)
                    linear_speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
                    angular_speed = abs(wz)
                    if (distance < self.target_distance and linear_speed < 0.3 and angular_speed < 0.15):
                        position_reached = True
                        break
                    self.controller_object.pubstart_position(target_position)
                    rate.sleep()
                if not position_reached:
                    raise RuntimeError("UAV failed to reach target position/yaw.")
                # Wait for target information
                uav_state = self.depth_processor.get_target_info(timeout=5.0)
                if not uav_state:
                    raise RuntimeError("Failed to initialize UAV state.")

                # Get initial raw observation (9D: [x, y, theta, distance, yaw, vx, vy, vz, wz])
                raw_obs = self.get_raw_observation(uav_state)
                obs = self.normalize_observation(raw_obs)  # Normalize observation
                return obs, {}
            except RuntimeError as e:
                retries += 1
                rospy.logwarn(f"Reset attempt {retries}/{max_retries} failed: {e}")
                if retries == max_retries:
                    rospy.logerr("Max reset retries reached. Returning default normalized observation.")
                    obs = self.normalize_observation(np.zeros_like(self.obs_low))  # Ensure normalized return value
                    return obs, {}

    def step(self, action):
        if self.current_state.mode == "AUTO.LAND":
            rospy.logerr("Failsafe triggered! Resetting environment.")
            return self.reset()
        if hasattr(action, 'numpy'):
            action = action.numpy()
        if isinstance(action, np.ndarray):
            action = action.tolist()

        # Define actions in body frame
        vx_body, vy_body, vz_body, yaw_rate = action[0], action[1], action[2], action[3]

        # 🔁 New method: publish body frame velocity command
        self.controller_object.pubBodyVelocity(
            vx=vx_body,
            vy=vy_body,
            vz=vz_body,
            wz=yaw_rate
        )
        uav_state = self.depth_processor.get_target_info(timeout=5.0)

        # Lost target case: direct penalty -100, no step penalty
        if not uav_state:
            observation = self.last_observation if self.last_observation is not None else np.zeros_like(
                self.observation_space.low, dtype=np.float32)
            reward = -100.0
            terminated, truncated = True, False
            terminal_cond = 'lost_target'
            info = {'terminal_condition': terminal_cond}
            self.terminal_counts[terminal_cond] += 1
            self.total_episodes += 1
            print(f"Episode terminated. Terminal counts: {self.terminal_counts}")
            return observation, reward, terminated, truncated, info

        # Get normalized observation
        observation = self.get_observation(uav_state)
        print(f"Normalized observation: {observation}")

        # Calculate reward using normalized observation
        reward, done, reason = self._calculate_reward(observation, action)
        self.current_step += 1
        truncated = (self.current_step >= self._max_episode_steps)

        # Determine termination condition: prioritize truncated, then reward condition
        if truncated:
            terminal_cond = 'max_steps'
        elif done:
            terminal_cond = reason
        else:
            terminal_cond = None

        # For "running", "success", "max_steps", add step penalty
        if (done or truncated) and terminal_cond not in {"lost_target", "out_of_bound"}:
            extra_penalty = -self.episode_penalty_factor * self.current_step
            reward += extra_penalty

        # For "out_of_bound", directly set to -100
        if terminal_cond == "out_of_bound":
            reward = -100.0

        # Update terminal counters if episode ends
        if done or truncated:
            if terminal_cond is not None:
                self.terminal_counts[terminal_cond] += 1
            print(f"Episode terminated. Terminal counts: {self.terminal_counts}")

        self.last_observation = observation
        return observation, reward, done, truncated, {'terminal_condition': terminal_cond}

    def _calculate_reward(self, observation, action):
        """
        Calculate reward from normalized state observation = [x, y, theta, d, ...].
        Ideal state (perfect docking): x=0, y=0, theta=0, d=-1 (normalized).
        Use differential reward: reward = current_shaping - prev_shaping.
        If all errors within thresholds, mark docking success, add +100 reward, and terminate episode.
        If denormalized distance > 6.0, consider out_of_bound, direct reward -100.
        """
        x, y, theta, d, yaw, vx, vy, vz, wz = observation
        error_x = abs(x)
        error_y = abs(y)
        error_theta = abs(theta)
        error_d = abs(d - (-0.75))
        shaping = -100 * (error_x + error_y + error_theta + 1.5 * error_d)  # Increase distance weight

        # Differential reward
        if self.prev_shaping is None:
            diff_reward = 0.0
        else:
            diff_reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward = diff_reward

        # 2. Action smoothing penalty
        current_action_norm = self.Action_adapter_reverse(np.array(action), self.action_space.high, self.action_space.low)
        if self.prev_action is None:
            smooth_penalty = 0.0
        else:
            action_diff = np.linalg.norm(current_action_norm - self.prev_action)
            smooth_penalty = self.lambda_smooth * (action_diff ** 2)
        print("Smooth penalty:", smooth_penalty)
        reward -= smooth_penalty
        self.prev_action = current_action_norm

        # Thresholds
        thresh_x = 0.19  # ±(2*80/848)
        thresh_y = 0.19
        thresh_theta = 0.111  # ±(2*10/180)
        thresh_d = 0.05  # Corresponds to actual distance [0.8,1.2]m.

        # Denormalize velocities
        actual_vx = (vx + 1) * (self.obs_high[5] - self.obs_low[5]) / 2 + self.obs_low[5]
        actual_vy = (vy + 1) * (self.obs_high[6] - self.obs_low[6]) / 2 + self.obs_low[6]
        actual_vz = (vz + 1) * (self.obs_high[7] - self.obs_low[7]) / 2 + self.obs_low[7]
        actual_wz = (wz + 1) * (self.obs_high[8] - self.obs_low[8]) / 2 + self.obs_low[8]

        # Compute actual speeds
        linear_speed = math.sqrt(actual_vx ** 2 + actual_vy ** 2 + actual_vz ** 2)
        angular_speed = abs(actual_wz)
        done = False
        reason = "running"

        # Check docking success
        if (error_x <= thresh_x and error_y <= thresh_y and error_theta <= thresh_theta and error_d <= thresh_d) and linear_speed < self.linear_speed_threshold_actual and angular_speed < self.angular_speed_threshold_actual:
                reward += 100.0
                done = True
                reason = "success"
        else:
            done = False
            reason = "running"

        # Denormalize distance and check bounds
        d_original = (d + 1) * (self.obs_high[3] - self.obs_low[3]) / 2 + self.obs_low[3]
        if d_original > 7.5 or d_original < 0.3:
            reward = -100.0
            done = True
            reason = "out_of_bound"

        return reward, done, reason

    def get_ekf_data(self, timeout=5.0):
        # Get orientation
        pose_msg = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped, timeout=timeout)
        q = (pose_msg.pose.orientation.x,
             pose_msg.pose.orientation.y,
             pose_msg.pose.orientation.z,
             pose_msg.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(q)

        # Get body frame velocity
        vel_body_msg = rospy.wait_for_message('/mavros/local_position/velocity_body', TwistStamped, timeout=timeout)
        vx = vel_body_msg.twist.linear.x
        vy = vel_body_msg.twist.linear.y
        vz = vel_body_msg.twist.linear.z

        # Get yaw rate (body z-axis angular velocity)
        imu_msg = rospy.wait_for_message('/mavros/imu/data', Imu, timeout=timeout)
        yaw_rate = imu_msg.angular_velocity.z

        # Optional: clip velocity values
        vx = np.clip(vx, -0.15, 0.15)
        vy = np.clip(vy, -0.15, 0.15)
        vz = np.clip(vz, -0.2, 0.2)
        yaw_rate = np.clip(yaw_rate, -0.1, 0.1)

        return (yaw, vx, vy, vz, yaw_rate)

    def Action_adapter_reverse(self, action, action_space_high, action_space_low):
        # Map from [action_space_low, action_space_high] to [-1,1]
        return 2.0 * (action - action_space_low) / (action_space_high - action_space_low) - 1.0

    def render(self, mode='human'):
        pass

    def close(self):
        self.depth_processor.stop()
        print("Environment closed successfully.")

    def state_cb(self, msg):
        self.current_state = msg
