# used modules from perturbation drive
from numpy import ndarray, uint8
import matplotlib.patches as patches
import os
import tensorflow as tf
#import cvxpy as cp
from perturbationdrive import (
    PerturbationSimulator,
    ADS,
    Scenario,
    ScenarioOutcome,
    ImageCallBack,
    ImagePerturbation,
    GlobalLog as Gl,
)
import traceback

# used libraries
from examples.udacity.udacity_utils.envs.udacity.udacity_gym_env import (
    UdacityGymEnv_RoadGen,
)
from typing import Union, Tuple
import cv2
import gym
import numpy as np
import time
import math

from udacity.perturbation.perturbationdrive import PerturbationDrive

WAYPOINT_THRESHOLD = 5
ANGLE_THRESHOLD = 0


class Waypoint_control_utils():
        def __init__(self,treshold,angle_treshold):
            self.waypoint_treshold=treshold
            self.angle_treshold=angle_treshold

        def angle_difference(self,a1, a2):
            diff = a1 - a2
            if diff>=180:
                diff-=360

            
            return diff
        
        def convert_waypoints(self,input_string):
            # Split the input string by the '@' symbol to get individual waypoints
            waypoints = input_string.split('@')
            
            # Initialize an empty list to hold the converted waypoints
            waypoint_list = []
            
            # Iterate through each waypoint string
            for waypoint in waypoints:
                # Split the string by the ',' symbol to get x, y, z values
                x, z, y = waypoint.split(',')
                
                # Convert x, y, z to floats and rearrange to [x, y, z]
                waypoint_list.append([float(x), float(y)])

            # waypoint_list,_=self.generate_road_margins(waypoint_list,1)
            
            return waypoint_list[1:]
        
        def generate_road_margins(self,road_points, offset):
            left_margins = []
            right_margins = []

            num_points = len(road_points)

            # Calculate the direction vectors for each road segment
            direction_vectors = []
            for i in range(num_points - 1):
                dx = road_points[i + 1][0] - road_points[i][0]
                dy = road_points[i + 1][1] - road_points[i][1]
                mag = np.sqrt(dx ** 2 + dy ** 2)
                direction_vectors.append((dx / mag, dy / mag))

            # Average neighboring direction vectors to get smoother normals
            averaged_directions = []
            for i in range(num_points - 1):
                if i == 0:
                    averaged_directions.append(direction_vectors[0])
                elif i == num_points - 2:
                    averaged_directions.append(direction_vectors[-1])
                else:
                    averaged_directions.append(((direction_vectors[i][0] + direction_vectors[i - 1][0]) / 2,
                                                (direction_vectors[i][1] + direction_vectors[i - 1][1]) / 2))

            # Calculate normals and generate margins
            for i in range(num_points - 1):
                dx, dy = averaged_directions[i]
                nx = -dy
                ny = dx

                left_x = road_points[i][0] + offset * nx
                left_y = road_points[i][1] + offset * ny
                right_x = road_points[i][0] - offset * nx
                right_y = road_points[i][1] - offset * ny

                left_margins.append([left_x, left_y])
                right_margins.append([right_x, right_y])

            return left_margins, right_margins
    
        def angle_extraction(self, x1, y1, z1, x2, y2, z2):

            # Calculate the distances between points
            dx = x2 - x1
            dy = y2 - y1
            dz = z2 - z1

            # Calculate the angles on each axis
            angle_x_axis = math.atan2(dy, dz)
            angle_y_axis = math.atan2(dx, dz)
            angle_z_axis = math.atan2(dy, dx)

            # Convert angles from radians to degrees
            angle_x_axis_degrees = math.degrees(angle_x_axis)
            angle_y_axis_degrees = math.degrees(angle_y_axis)
            angle_z_axis_degrees = math.degrees(angle_z_axis)
            return angle_x_axis_degrees, angle_y_axis_degrees, angle_z_axis_degrees
        
        def exponential_increase(self,number, factor):
            return factor * (1 - np.exp(-number))

        def calculate_control(self, x_target, y_target, simulator_pose, simulator_orientation):
            x_cur, y_cur, _ = simulator_pose

            #print(f"Position\nx:{x_cur}, y:{y_cur}")
            _, angle_cur, _ = simulator_orientation
            #print(f"Orientation: {round(angle_cur, 3)}, {round(math.radians(angle_cur), 3)} rad")
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            #print(f"dist {distance}")
            _, angle_y_axis_degrees, _=self.angle_extraction(x_cur, 0.0, y_cur, x_target, 0.0, y_target)
            #print(f"Angle to goal: {angle_y_axis_degrees}")
            angle_difference=self.angle_difference(angle_cur,angle_y_axis_degrees)
            # print(f"angle diff {angle_difference}")
            steering = (math.radians(-angle_difference) / math.pi)*4
            # print(f"angle diff: {angle_difference}, steering: {steering}")
            throttle=distance/10

            return steering, throttle, distance, angle_difference,angle_y_axis_degrees
        
        def calculate_distance(self, x_target, y_target, simulator_pose):
            x_cur, y_cur, _ = simulator_pose
            distance = math.sqrt((x_target - x_cur)**2 + (y_target - y_cur)**2)
            return distance

        def is_waypoint_in_back(self,current_pos, current_orientation, waypoint_x,waypoint_y):
            current_x, current_y, _ = current_pos
            _, angle_cur, _ = current_orientation
            current_orientation=math.radians(angle_cur)
            
            # Calculate vector to the next waypoint
            waypoint_vector = (waypoint_x - current_x, waypoint_y - current_y)
            
            # Calculate vehicle forward vector based on its orientation
            vehicle_forward_vector = (math.cos(current_orientation), math.sin(current_orientation))
            
            # Calculate the dot product
            dot_product = waypoint_vector[0] * vehicle_forward_vector[0] + waypoint_vector[1] * vehicle_forward_vector[1]
            
            # If the dot product is positive, the waypoint is in front; otherwise, it's behind
            return dot_product <= 0


class UdacitySimulator(PerturbationSimulator):
    def __init__(
        self,
        simulator_exe_path: str = "./examples/udacity/udacity_utils/sim/udacity_sim.app",
        host: str = "127.0.0.1",
        port: int = 9091,
        show_image_cb=True
    ):
        # udacity road is 8 units wide
        super().__init__(
            max_xte=4.0,
            simulator_exe_path=simulator_exe_path,
            host=host,
            port=port,
            initial_pos=None,
        )
        self.client: Union[UdacityGymEnv_RoadGen, None] = None
        self.logger = Gl("UdacitySimulator")
        self.show_image_cb=show_image_cb
            

    def connect(self):
        super().connect()
        self.client = UdacityGymEnv_RoadGen(
            seed=1,
            exe_path=self.simulator_exe_path,
        )
        self.client.reset()
        time.sleep(2)
        self.logger.info("Connected to Udacity Simulator")
        # set initial pos
        obs, done, info = self.client.observe()
        x, y, z = info["pos"]
        if self.initial_pos is None:
            self.initial_pos = (x, y, z, 2 * self.max_xte)
        self.logger.info(f"Initial pos: {self.initial_pos}")

    def simulate_scanario(
        self,
        agent: Union[ADS,None],
        scenario: Scenario,
        perturbation_controller: Union[ImagePerturbation,None],
        image_size: Tuple[float, float] = (160, 320),
        perturb=False,
        model_drive=False,
        weather="Sun",
        intensity=90
    ) -> bool:
        try:
            waypoints = scenario.waypoints
            height = image_size[0]
            width = image_size[1]
            perturbation_function_string = scenario.perturbation_function
            perturbation_scale = scenario.perturbation_scale
            monitor = ImageCallBack(rows = int(height), cols = int(width))
            monitor.display_waiting_screen()
            self.logger.info(f"{5 * '-'} Starting udacity scenario {5 * '_'}")

            # set up params for saving data
            xte_list = []
            original_image_list=[]
            isSuccess = False
            done = False
            timeout = False

            # reset the scene to match the scenario
            self.client.weather(weather,intensity)

            obs: ndarray[uint8] = self.client.reset(
                skip_generation=False, track_string=waypoints
            )

            obs, done, info = self.client.observe()
            start_time = time.time()
            waypoint_controller = Waypoint_control_utils(WAYPOINT_THRESHOLD, ANGLE_THRESHOLD)

            current_waypoint_index=0
            waypoint_list=waypoint_controller.convert_waypoints(waypoints)
            waypoint_list=waypoint_list
            # print(len(waypoint_list))

            counter=0
            # additional iteration once
            once=True
            data = []
            while True:
                counter+=1
                if time.time() - start_time > 100:
                    self.logger.info("Udacity: Timeout after 100s")
                    timeout = True
                    break
                #obs.shape=(160, 320, 3)
                if obs.shape[:2] != (height, width) :
                    obs = cv2.resize(obs, (width, height), cv2.INTER_NEAREST)
                original_image_list.append(obs)

                if perturb:
                    # perturb the image
                    perturbed_image = perturbation_controller.perturbation(
                        obs, perturbation_function_string, perturbation_scale
                    )
                    image=perturbed_image
                else:
                    image=obs

                rotation=info["orientation_euler"]

                if current_waypoint_index < len(waypoint_list):
                    current_waypoint = waypoint_list[current_waypoint_index]
                x, y = current_waypoint
                steering, throttle, dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)

                if dist <= WAYPOINT_THRESHOLD:
                    current_waypoint_index += 4
                    if  current_waypoint_index < len(waypoint_list):
                        current_waypoint = waypoint_list[current_waypoint_index]
                    x, y = current_waypoint
                    steering,throttle , dist, angl_diff,angle = waypoint_controller.calculate_control(x, y, info["pos"], rotation)

                if done:
                    if once:  # one additional iteration
                        once = False
                    else:  # Exit the loop after the "extra once"
                        break

                data.append({
                    'index': counter,
                    'track': "Road_Generator",
                    'perturb_name': perturbation_function_string,
                    'scale': perturbation_scale,
                    'speed': info['speed'],
                    'steer': steering,
                    'throttle': throttle,
                    'cte': dist,
                    # 'out_of_track': crash.get("out_of_track"),
                    # 'collision': crash.get("collision"),
                    # 'low_speed': crash.get("low_speed"),
                    'is_crashed': not done
                })

                # agent makes a move, the agent also selects the dtype and adds a batch dimension
                actions = agent.action(image)

                # clip action to avoid out of bound errors
                if isinstance(self.client.action_space, gym.spaces.Box):
                    actions = np.clip(
                        actions,
                        self.client.action_space.low,
                        self.client.action_space.high,
                    )
                # if self.show_image_cb:
                monitor.display_img(
                    image,
                    f"{actions[0][0]}",
                    f"{actions[0][1]}",
                    perturbation_function_string,
                )
                # obs is the image, info contains the road and the position of the car
                obs, done, info = self.client.step(actions)

                time.sleep(0.015)

                # log new info
                xte_list.append(info["cte"])

            # determine if we were successful
            isSuccess = max([abs(xte) for xte in xte_list]) < self.max_xte
            if timeout:
                isSuccess=False
            self.logger.info(
                f"{5 * '-'} Finished udacity scenario: {isSuccess} {5 * '_'}"
            )
            monitor.display_disconnect_screen()
            monitor.destroy()

            log_name = f"RoadGen_{perturbation_function_string}_intense{perturbation_scale}_log.csv"
            log_path = os.path.join("./udacity/perturb_logs", log_name)

            # store in log only when ADS drives a not short way but crashed before the end
            if data and data[-1]['index'] > 400 and isSuccess==False:
                PerturbationDrive.perturb_driving_log(log_path=log_path, data=data)

            # reset for the new track
            _ = self.client.reset(skip_generation=False, track_string=waypoints)

            return isSuccess

        except Exception as e:
            # close the simulator
            self.tear_down()
            traceback.print_stack()
            # throw the exception
            raise e

    def tear_down(self):
        self.client.close()

    def name(self) -> str:
        return "UdacitySimualtorAdapter"
