import eventlet
eventlet.monkey_patch()
# ensure compatibility

import os
import sys
import time
import numpy as np
from PIL import Image
from udacity.perturbation.imageperturbations import ImagePerturbation
from udacity.utils import perturb_driving_log
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
    MODEL_NAME = "track1-steer-throttle.h5"

    # MODEL_NAME = "track1_change_dropout_rate_mutated_1.0_layer4.h5"
    MODEL_PATH = os.path.join("./checkpoints", MODEL_NAME)

    simulator = UdacitySimulator(sim_exe_path=SIMULATOR_PATH, host=HOST, port=PORT)
    env = UdacityGym(simulator=simulator)
    agent = SupervisedAgent(model_path=MODEL_PATH, max_speed=30, min_speed=6, predict_throttle=True)

    IMAGE_SIZE = (160, 320) #Perturbated Image Size, normal same as input size
    # "gaussian_noise", "poisson_noise" too large
    # Done "impulse_noise", "defocus_blur", "glass_blur", "motion_blur"
    PERTURBATIONS = ["glass_blur"]
    # PERTURBATIONS = ["zoom_blur","increase_brightness","contrast","elastic","pixelate","jpeg_filter","shear_image"]
    SCALE = 10 # scale from 0 to 4
    LOW_SPEED_THRESHOLD = 0.01
    LOW_SPEED_LIMIT = 20
    # 3-5
    TOTAL_CRASH_LIMIT = (3,5) if TRACK == "lake" else (3,6) if TRACK == "mountain" else (1,4)

    visualize = True
    perturb = True
    simulator.start()
    image_perturbation = create_perturb_list(PERTURBATIONS, IMAGE_SIZE)

    if not perturb:
        perturbations = ["no_perturb"]

    for perturbation in PERTURBATIONS:
        scale = 0
        while True:
            print("---------------------------------------")
            print("Start perturbation: ", perturbation)
            print("Current scale of testing: ", scale)

            obs, _ = env.reset(track=TRACK, weather=WEATHER, daytime=DAYTIME)

            while not obs or not obs.is_ready():
                obs = env.observe()
                time.sleep(1)

            monitor = ImageCallBack(rows=IMAGE_SIZE[0], cols=IMAGE_SIZE[1]) if visualize else None
            if monitor:
                monitor.display_waiting_screen()

            # LOG_PATH here should be ABSOLUTE, info is required in csv log file
            LOG_NAME = f"{TRACK}_{perturbation}_intense{scale}_log.csv"
            LOG_PATH = f"/home/jiaqq/Project-1120/PerturbationDrive/udacity/perturb_logs/{TRACK}_{perturbation}_intense{scale}_log"

            data = []
            low_speed_count, frame, total_crash = 0, 0, 0
            keep_running = True

            # First condition: keep testing in one round
            # Second: Ensure ADS doesn't stop in the middle
            # Third: Ensure ADS can still perform its driving skills
            while obs.lap == 1 and keep_running and total_crash <= TOTAL_CRASH_LIMIT[1]:
                frame += 1 # used to store images and csv_data

                image_folder = os.path.join(LOG_PATH, "image_logs")
                if not os.path.exists(image_folder):
                    os.makedirs(image_folder)
                image_path = os.path.join(image_folder, f"{frame}.png")

                image = np.array(obs.input_image) # image为 RGB, shape (160, 320, 3)
                if perturb:
                    obs.input_image = image_perturbation.perturbation(image, perturbation, scale)

                image = Image.fromarray(obs.input_image)
                image.save(image_path)  # plt.imsave(image_path, obs.input_image)

                actions = agent(obs)
                # print(obs.input_image.shape) # (160, 320, 3)

                if monitor:
                    monitor.display_img(obs.input_image, f"{actions.steering_angle}", f"{actions.throttle}", perturbation)

                last_obs = obs
                obs, reward, terminated, truncated, crash, info  = env.step(actions)
                # print(obs.throttle,"   ,", obs.speed)

                if obs.speed <= LOW_SPEED_THRESHOLD and reward <= 4: # 只算道路内主观停止
                    low_speed_count += 1
                else:
                    low_speed_count = 0
                #     def ————（speed, threshold, cte）3种情况 停止 倒车 卡住（cte相关）

                if low_speed_count >= LOW_SPEED_LIMIT: # 连续20帧速度小于0.001
                    crash["low_speed"] += 1
                    keep_running = False


                data.append({
                    'index': frame,
                    'track': TRACK,
                    'model': MODEL_NAME,
                    'perturb_name': perturbation,
                    'scale': scale,
                    'image_path': image_path,
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

                if crash.get("is_crashed"):
                    total_crash += 1

                env.simulator.sim_state['is_crashed'] = False

                while obs.time == last_obs.time:
                    obs = env.observe()
                    time.sleep(0.05)

            if monitor:
                monitor.display_disconnect_screen()
                monitor.destroy()

            if TOTAL_CRASH_LIMIT[0] <= total_crash <= TOTAL_CRASH_LIMIT[1]:
                perturb_driving_log(os.path.join(LOG_PATH, LOG_NAME), data)

                print(f"Data saved under {TRACK}_{perturbation}_scale{scale}_log.csv!")
                print("Out_of_track Count: ", crash.get("out_of_track"), "; Collision Count: ",
                      crash.get("collision"))

                break  # jump out of current perturbation

            elif total_crash <= TOTAL_CRASH_LIMIT[0]:
                print(f"ADS drives perfect! Skipping record of {TRACK}_{perturbation}_scale{scale}_log.csv")
            else:
                print(f"Too many crashes! Skipping record of {TRACK}_{perturbation}_scale{scale}_log.csv")

            scale += 1
            if scale > 10:
                break

    simulator.close()
    env.close()


