import eventlet
eventlet.monkey_patch()
# ensure compatibility

import os
import csv
import sys
import time
import numpy as np
from udacity.perturbation.imageperturbations import ImagePerturbation
from udacity.perturbation.perturbationdrive import PerturbationDrive
from perturbationdrive import ImageCallBack
from udacity.ase_simulation.agent import SupervisedAgent
from udacity.ase_simulation.gym import UdacityGym
from udacity.ase_simulation.simulator import UdacitySimulator

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def create_perturb_list(perturbation_functions, image_size):
    return ImagePerturbation(funcs=perturbation_functions, attention_map={}, image_size=image_size)

if __name__ == '__main__':
    SIMULATOR_PATH = "./udacity/udacity_sim_tracks/udacity.x86_64"
    HOST, PORT = "127.0.0.1", 4567

    TRACK = "lake"
    DAYTIME, WEATHER = "day", "sunny"
    MODEL_PATH = f"./checkpoints/track1-steer-throttle.h5"
    # MODEL_PATH = "/home/jiaqq/Project-1120/PerturbationDrive/checkpoints/NEB_0.05_400.h5"

    simulator = UdacitySimulator(sim_exe_path=SIMULATOR_PATH, host=HOST, port=PORT)
    env = UdacityGym(simulator=simulator)
    agent = SupervisedAgent(model_path=MODEL_PATH, max_speed=30, min_speed=6, predict_throttle=True)

    IMAGE_SIZE = (160, 320) #Perturbated Image Size, normal same as input size
    PERTURBATIONS = ["scale_image"]
    SCALE = 4 # scale from 0 to 4
    LOW_SPEED_THRESHOLD = 0.01
    LOW_SPEED_LIMIT = 20

    visualize = True
    perturb = True
    simulator.start()
    image_perturbation = create_perturb_list(PERTURBATIONS, IMAGE_SIZE)

    if not perturb:
        perturbations = ["no_perturb"]

    for perturbation in PERTURBATIONS:
        for scale in range(0, SCALE + 1):
            obs, _ = env.reset(track=TRACK, weather=WEATHER, daytime=DAYTIME)

            while not obs or not obs.is_ready():
                obs = env.observe()
                time.sleep(1)

            monitor = ImageCallBack(rows=IMAGE_SIZE[0], cols=IMAGE_SIZE[1]) if visualize else None
            if monitor:
                monitor.display_waiting_screen()

            LOG_FILE = f"{TRACK}_{perturbation}_intense{scale}_log.csv"
            log_path = os.path.join("./udacity/perturb_logs", LOG_FILE)
            data = []
            low_speed_count, csv_index = 0, 0
            keep_running = True
            print(scale)
            while obs.lap == 1 and keep_running:
                image = np.array(obs.input_image) # image为 RGB
                # print(image.shape) #(160, 320, 3)
                #resized_image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)

                if perturb:
                    obs.input_image = image_perturbation.perturbation(image, perturbation, scale)

                actions = agent(obs)
                # print(obs.input_image.shape) # (160, 320, 3)

                if monitor:
                    monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}", perturbation)

                last_obs = obs
                obs, reward, terminated, truncated, crash, info  = env.step(actions)
                csv_index += 1

                if obs.speed <= LOW_SPEED_THRESHOLD and reward <= 4: # 只算道路内主观停止
                    low_speed_count += 1
                else:
                    low_speed_count = 0

                if low_speed_count >= LOW_SPEED_LIMIT: # 连续10帧速度小于0.001
                    crash["low_speed"] += 1
                    keep_running = False

                data.append({
                    'index': csv_index,
                    'track': TRACK,
                    'perturb_name': perturbation,
                    'scale': scale,
                    'lap': obs.lap,
                    'waypoint': obs.sector,
                    'speed': obs.speed,
                    'steer': actions.steering_angle,
                    'throttle': actions.throttle,
                    'cte': reward,
                    'out_of_track': crash.get("out_of_track"),
                    'collision': crash.get("collision"),
                    'low_speed': crash.get("low_speed"),
                    'is_crashed': crash.get("is_crashed")
                })

                env.simulator.sim_state['is_crashed'] = False

                while obs.time == last_obs.time:
                    obs = env.observe()
                    time.sleep(0.05)

            PerturbationDrive.perturb_driving_log(log_path, data)
            print(f"Data saved under {TRACK}_{perturbation}_log.csv!")
            print("Out_of_track Count: ", crash.get("out_of_track"), "; Collision Count: ", crash.get("collision") )

            if monitor:
                monitor.display_disconnect_screen()
                monitor.destroy()

    simulator.close()
    env.close()


