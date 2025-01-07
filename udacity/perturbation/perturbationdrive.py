from concurrent.futures import ThreadPoolExecutor

from tensorflow.python.eager.profiler_client import monitor

from perturbationdrive.Simulator.Simulator import PerturbationSimulator
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.imageperturbations import ImagePerturbation, get_functions_from_module
from perturbationdrive.Simulator.Scenario import Scenario, ScenarioOutcome, OfflineScenarioOutcome
from perturbationdrive.RoadGenerator.RoadGenerator import RoadGenerator
from udacity.utils import *
from perturbationdrive.utils.logger import ScenarioOutcomeWriter, OfflineScenarioOutcomeWriter

from typing import List, Union, Dict, Tuple
import os, json, cv2, copy, csv, time


class PerturbationDrive:
    """
    Simulator independent ADS robustness benchmarking
    """

    def __init__(
        self,
        simulator: PerturbationSimulator,
        ads: Union[ADS, None],
    ):
        assert isinstance(
            simulator, PerturbationSimulator
        ), "Simulator must be a subclass of PerturbationSimulator"
        if ads is not None:
            assert isinstance(ads, ADS), "ADS must be a subclass of ADS"
        self.simulator = simulator
        self.ads = ads

    def setADS(self, ads: ADS):
        assert isinstance(ads, ADS), "ADS must be a subclass of ADS"
        self.ads = ads

    def setSimulator(self, simulator: PerturbationSimulator):
        assert isinstance(
            simulator, PerturbationSimulator
        ), "Simulator must be a subclass of PerturbationSimulator"
        self.simulator = simulator

    def grid_seach(
            self,
            perturbation_functions: List[str],
            attention_map: Dict = {},
            road_number: int = 0,
            road_generator: Union[RoadGenerator, None] = None,
            road_angles: List[int] = None,
            road_segments: List[int] = None,
            image_size: Tuple[float, float] = (160, 320),
            test_model: bool = False,
            perturb: bool = False,
            monitor: bool = False,
            scale_limit: int = 4,
            weather: Union[str, None] = "Sun",
            weather_intensity: Union[int, None] = 90
    ):
        """
        Basically, what we have done in image perturbations up until now but in a single nice function wrapped

        If log_dir is none, we return the scenario outcomes
        """
        if perturb:
            image_perturbation = ImagePerturbation(
                funcs=perturbation_functions,
                attention_map=attention_map,
                image_size=image_size,
            )

        else:
            image_perturbation = None

        scale = 0
        perturbations: List[str] = []

        if perturb:
            perturbations: List[str] = copy.deepcopy(perturbation_functions)

        # we append the empty perturbation here
        # perturbations.append("")

        # set up simulator
        self.simulator.connect()
        # wait 1 second for connection to build up
        time.sleep(1)

        # set up initial road
        waypoints = None
        if not road_generator is None:
            # TODO: Insert here all kwargs needed for specific generator
            waypoints = road_generator.generate(starting_pos=self.simulator.initial_pos, angles=road_angles,
                                                seg_lengths=road_segments)

        # grid search loop
        while True:
            perturbation = perturbations[0]
            print(
                f"{5 * '-'} Running Scenario: Perturbation {perturbation} on scale: {scale} {5 * '-'}"
            )

            scenario = Scenario(
                waypoints=waypoints,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )

            LOG_NAME = f"roadGen_{perturbation}_road{road_number}_scale{scale}_log.csv"
            LOG_PATH = f"/home/jiaqq/Project-1120/PerturbationDrive/udacity/perturb_logs/roadGen/{perturbation}/roadGen_{perturbation}_road{road_number}_scale{scale}_log"
            image_folder = os.path.join(LOG_PATH, "image_logs")

            # simulate the scenario
            isSuccess, temporary_images, data = self.simulator.simulate_scanario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation, perturb=perturb, visualize=monitor,
                model_drive=test_model, weather=weather, intensity=weather_intensity, image_folder = image_folder
            )

            if len(perturbations) == 0:
                # all perturbations resulted in failures
                # we will still have one perturbation here because we never
                # drop the empty perturbation
                break

            if isSuccess: # no crashed in this turn, so iterate into the next scale_level
                print(f"No crash in current scale: {scale}, increasing scale.")
                scale += 1

            else:
                if len(temporary_images) > 50:# crash happens in current scale, record the image and data, jump out to the next perturbation
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [executor.submit(save_image, image_path, image) for image_path, image in temporary_images]
                    # 等待所有任务完成
                    for future in futures:
                        try:
                            future.result()  # 检查任务状态
                        except Exception as e:
                            print(f"Error during image saving: {e}")

                    perturb_driving_log(os.path.join(LOG_PATH, LOG_NAME), data)
                    print(f"Data saved under {LOG_NAME}!")

                else:
                    print("Driving performance bad, too short! Data is not saving!")

                scale = 0
                perturbations.remove(perturbation)
                if len(perturbations) == 0:
                    break
                data.clear()
                futures.clear()
                temporary_images.clear()
                time.sleep(2)
                print("Data has been cleared!")

            if scale > scale_limit:
                # we went through all scales
                print("Drives perfect in all scales! Going into the next perturbation!")
                break

        # TODO: print command line summary of benchmarking process
        del image_perturbation
        del scenario
        del road_generator

        # tear down the simulator
        self.simulator.tear_down()
