# Run Crazyflie simulation in openai gym style (example with PPO reinforcement learning)
# Author: Shushuai Li at EPFL LIS Lab
# Based on openai_gym.py in webots and crazyflie_controller_py.py
# To use conda, run webots in conda environment
#   Or create runtime.ini file in this folder with content: [python]
#   COMMAND = /home/shli/anaconda3/envs/tf/bin/python3.10

from operator import truediv
import sys
sys.path.append("/usr/local/webots/lib/controller/python38")
from controller import Supervisor
from math import cos, sin
import time


try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym stable_baselines3"'
    )

# Create gym environment using webots
class CrazyflieInDroneDome(Supervisor, gym.Env):
    def __init__(self, max_episode_steps = 1000):
        super().__init__()
        ## Open AI Gym generic
        # Action space: [forward_velocity, left_velocity, yaw_rate]
        self.action_space = gym.spaces.Box(low = np.array([-0.005, -0.005, -0.0001]),
                                        high = np.array([0.005, 0.005, 0.0001]),
                                        dtype = np.float32)

        # Observation space: [pos_x_to_goal, pos_y_to_goal, range_front,
        #                       range_left, range_back, range_right]
        self.observation_space = gym.spaces.Box(
                                    low = np.array([-20.0, -20.0, 0.0, 0.0, 0.0, 0.0]),
                                    high = np.array([20.0, 20.0, 2.0, 2.0, 2.0, 2.0]),
                                    dtype = np.float32)

        self.state = None

        ## Environment specific
        self.__timestep = int(self.getBasicTimeStep())

        # Actuators
        self.__motors = []

        # Sensors
        self.__imu = []
        self.__gps = []
        self.__gyro = []
        self.__camera = []
        self.__range_front = []
        self.__range_left = []
        self.__range_back = []
        self.__range_right = []

        # History variables
        self.__pastXGlobal = 0
        self.__pastYGlobal = 0
        self.__past_time = self.getTime()

        # Crazyflie velocity PID controller (converted from C code)
        self.__PID_CF = pid_velocity_fixed_height_controller()

        # Task variables
        self.goal_position_x = 4.0
        self.goal_position_y = 0.0
        self.last_action = [0, 0, 0]
        self.step_count = 0
        self.time_last = time.time()

        ## Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)

    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def action_from_keyboard(self):
        forwardVelocityDesired = 0
        leftVelocityDesired = 0
        yawRateDesired = 0
        key = self.keyboard.getKey()
        while key > 0:
            if key == ord('W'):
                forwardVelocityDesired = 0.6
            elif key == ord('S'):
                forwardVelocityDesired = -0.6
            elif key == ord('D'):
                leftVelocityDesired = -0.6
            elif key == ord('A'):
                leftVelocityDesired = 0.6
            elif key == ord('Q'):
                yawRateDesired = 0.8
            elif key == ord('E'):
                yawRateDesired = -0.8
            key = self.keyboard.getKey()
        return [forwardVelocityDesired, leftVelocityDesired, yawRateDesired]

    def observations(self):
        # Get robot position
        xGlobal = self.__gps.getValues()[0]
        yGlobal = self.__gps.getValues()[1]

        # Get robot yaw direction
        actualYaw = self.__imu.getRollPitchYaw()[2]
        cosyaw, sinyaw = cos(actualYaw), sin(actualYaw)

        # Get relative position to the goal
        pos_x_to_goal = self.goal_position_x - xGlobal
        pos_y_to_goal = self.goal_position_y - yGlobal
        pos_x_to_goal_bodyframe = pos_x_to_goal * cosyaw + pos_y_to_goal * sinyaw
        pos_y_to_goal_bodyframe = pos_x_to_goal * cosyaw + pos_y_to_goal * sinyaw

        # Range sensor measurements
        range_front = self.__range_front.getValue() / 1000.0
        range_left = self.__range_left.getValue() / 1000.0
        range_back = self.__range_back.getValue() / 1000.0
        range_right = self.__range_right.getValue() / 1000.0

        return np.array([pos_x_to_goal_bodyframe, pos_y_to_goal_bodyframe,
            range_front, range_left, range_back, range_right]).astype(np.float32)

    def reset(self):
        ## Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)

        # Reser motors
        self.__motors = []
        for name in ['m1_motor', 'm2_motor', 'm3_motor', 'm4_motor']:
            motor = self.getDevice(name)
            motor.setPosition(float('inf'))
            motor.setVelocity(-1)
            self.__motors.append(motor)

        # Reset sensors
        self.__imu = self.getDevice('inertial unit')
        self.__imu.enable(self.__timestep)
        self.__gps = self.getDevice('gps')
        self.__gps.enable(self.__timestep)
        self.__gyro = self.getDevice('gyro')
        self.__gyro.enable(self.__timestep)
        self.__camera = self.getDevice('camera')
        self.__camera.enable(self.__timestep)
        self.__range_front = self.getDevice('range_front')
        self.__range_front.enable(self.__timestep)
        self.__range_left = self.getDevice("range_left")
        self.__range_left.enable(self.__timestep)
        self.__range_back = self.getDevice("range_back")
        self.__range_back.enable(self.__timestep)
        self.__range_right = self.getDevice("range_right")
        self.__range_right.enable(self.__timestep)

        # Reset variables
        self.__pastXGlobal = 0
        self.__pastYGlobal = 0
        self.__past_time = self.getTime()
        self.step_count = 0

        # Reset PID
        self.__PID_CF = pid_velocity_fixed_height_controller()

        # Internals
        super().step(self.__timestep)

        # Uncomment if we want the drone take-off before learning
        stop_time = self.__past_time + 3
        while self.getTime() <= stop_time:
            dt = self.getTime() - self.__past_time
            # Get measurements
            actual_roll = self.__imu.getRollPitchYaw()[0]
            actual_pitch = self.__imu.getRollPitchYaw()[1]
            actual_yaw_rate = self.__gyro.getValues()[2]
            actual_alt = self.__gps.getValues()[2]
            xGlobal = self.__gps.getValues()[0]
            vxGlobal = (xGlobal - self.__pastXGlobal)/dt
            yGlobal = self.__gps.getValues()[1]
            vyGlobal = (yGlobal - self.__pastYGlobal)/dt

            # Get body fixed velocities
            actualYaw = self.__imu.getRollPitchYaw()[2]
            cosyaw = cos(actualYaw)
            sinyaw = sin(actualYaw)
            actual_vx =  vxGlobal * cosyaw + vyGlobal * sinyaw
            actual_vy = -vxGlobal * sinyaw + vyGlobal * cosyaw

            # Low-level PID velocity control with fixed height
            desired_alt = 1.0
            action = [0, 0, 0]
            motorPower = self.__PID_CF.pid(dt, action, desired_alt, actual_roll, actual_pitch,
                                            actual_yaw_rate, actual_alt, actual_vx, actual_vy)

            self.__motors[0].setVelocity(-motorPower[0])
            self.__motors[1].setVelocity(motorPower[1])
            self.__motors[2].setVelocity(-motorPower[2])
            self.__motors[3].setVelocity(motorPower[3])
            
            self.__past_time = self.getTime()
            self.__pastXGlobal = xGlobal
            self.__pastYGlobal = yGlobal

            super().step(self.__timestep)

        # Open AI Gym generic
        return self.observations()

    def step(self, action):
        time_now = time.time()
        print('Step time:', time_now - self.time_last)
        self.time_last = time_now
        dt = self.getTime() - self.__past_time
        # Get measurements
        actual_roll = self.__imu.getRollPitchYaw()[0]
        actual_pitch = self.__imu.getRollPitchYaw()[1]
        actual_yaw_rate = self.__gyro.getValues()[2]
        actual_alt = self.__gps.getValues()[2]
        xGlobal = self.__gps.getValues()[0]
        vxGlobal = (xGlobal - self.__pastXGlobal)/dt
        yGlobal = self.__gps.getValues()[1]
        vyGlobal = (yGlobal - self.__pastYGlobal)/dt

        # Get body fixed velocities
        actualYaw = self.__imu.getRollPitchYaw()[2]
        cosyaw = cos(actualYaw)
        sinyaw = sin(actualYaw)
        actual_vx =  vxGlobal * cosyaw + vyGlobal * sinyaw
        actual_vy = -vxGlobal * sinyaw + vyGlobal * cosyaw

        # Low-level PID velocity control with fixed height
        desired_alt = 1.0
        motorPower = self.__PID_CF.pid(dt, action, desired_alt, actual_roll, actual_pitch,
                                        actual_yaw_rate, actual_alt, actual_vx, actual_vy)

        self.__motors[0].setVelocity(-motorPower[0])
        self.__motors[1].setVelocity(motorPower[1])
        self.__motors[2].setVelocity(-motorPower[2])
        self.__motors[3].setVelocity(motorPower[3])
        
        self.__past_time = self.getTime()
        self.__pastXGlobal = xGlobal
        self.__pastYGlobal = yGlobal

        super().step(self.__timestep)

        # Observation
        observation = self.observations()

        self.step_count += 1
        # Done
        done = bool(
            observation[2] < 0.1 or
            observation[3] < 0.1 or
            observation[4] < 0.1 or
            observation[5] < 0.1
        )

        # Reward
        reward = 0 if done else 10-np.linalg.norm(observation[0:2])

        if self.step_count > 300:
            done = True
            reward = 10-np.linalg.norm(observation[0:2])

        # Debug print
        print(observation, reward, done)

        # Open AI Gym generic
        return observation, reward, done, {}
    
class pid_velocity_fixed_height_controller():
    def __init__(self):
        self.pastVxError = 0
        self.pastVyError = 0
        self.pastAltError = 0
        self.pastPitchError = 0
        self.pastRollError = 0

    def pid(self, dt, action, desired_alt, actual_roll, actual_pitch, actual_yaw_rate,
            actual_alt, actual_vx, actual_vy):
        # Cascaded PID control (coverted from Crazyflie c code)
        gains = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
                "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 50, "kd_z": 5}

        # Actions
        desired_vx, desired_vy, desired_yaw_rate = action[0], action[1], action[2]

        # Horinzontal velocity PID control
        vxError = desired_vx - actual_vx
        vxDeriv = (vxError - self.pastVxError) / dt
        vyError = desired_vy - actual_vy
        vyDeriv = (vyError - self.pastVyError) / dt
        desired_pitch = gains["kp_vel_xy"] * np.clip(vxError, -1, 1) + gains["kd_vel_xy"] * vxDeriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vyError, -1, 1) - gains["kd_vel_xy"] * vyDeriv
        self.pastVxError = vxError
        self.pastVyError = vyError

        # Altitude PID control
        altError = desired_alt - actual_alt
        altDeriv = (altError - self.pastAltError) / dt
        altCommand = gains["kp_z"] * np.clip(altError, -1, 1) + gains["kd_z"] * altDeriv + gains["ki_z"]
        self.pastAltError = altError

        # Attitude PID control
        pitchError = desired_pitch - actual_pitch
        pitchDeriv = (pitchError - self.pastPitchError) / dt
        rollError = desired_roll - actual_roll
        rollDeriv = (rollError - self.pastRollError) / dt
        yawRateError = desired_yaw_rate - actual_yaw_rate
        rollCommand = gains["kp_att_rp"] * np.clip(rollError, -1, 1) + gains["kd_att_rp"] * rollDeriv
        pitchCommand = -gains["kp_att_rp"] * np.clip(pitchError, -1, 1) - gains["kd_att_rp"] * pitchDeriv
        yawCommand = gains["kp_att_y"] * np.clip(yawRateError, -1, 1)
        self.pastPitchError = pitchError
        self.pastRollError = rollError

        # Motor mixing
        m1 =  altCommand - rollCommand + pitchCommand + yawCommand
        m2 =  altCommand - rollCommand - pitchCommand - yawCommand
        m3 =  altCommand + rollCommand - pitchCommand + yawCommand
        m4 =  altCommand + rollCommand + pitchCommand - yawCommand
        return [m1, m2, m3, m4]

def main():
    # Initialize the environment
    env = CrazyflieInDroneDome()
    # check_env(env)

    # Train
    model = PPO('MlpPolicy', env, n_steps=300, batch_size=300*1, verbose=1)
    model.learn(total_timesteps=1e5)
    model.save("ppo_crazyflie")

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()
    for _ in range(100000):
        # action, _states = model.predict(obs)
        # action = env.action_space.sample()
        action = env.action_from_keyboard()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
