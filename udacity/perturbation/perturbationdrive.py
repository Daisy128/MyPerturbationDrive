from perturbationdrive.Simulator.Simulator import PerturbationSimulator
from perturbationdrive.AutomatedDrivingSystem.ADS import ADS
from perturbationdrive.imageperturbations import ImagePerturbation, get_functions_from_module
from perturbationdrive.Simulator.Scenario import Scenario, ScenarioOutcome, OfflineScenarioOutcome
from perturbationdrive.RoadGenerator.RoadGenerator import RoadGenerator
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
            road_generator: Union[RoadGenerator, None] = None,
            road_angles: List[int] = None,
            road_segments: List[int] = None,
            image_size: Tuple[float, float] = (160, 320),
            test_model: bool = False,
            perturb: bool = False,
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
        index = 0
        perturbations: List[str] = []
        lack_perturb_scale = []
        is_crashed = False

        if perturb:
            perturbations: List[str] = copy.deepcopy(perturbation_functions)
            # populate all perturbations
            if len(perturbations) == 0:
                perturbation_fns = get_functions_from_module(
                    "perturbationdrive.perturbationfuncs"
                )
                perturbations = list(map(lambda f: f.__name__, perturbation_fns))

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
            perturbation = perturbations[index]
            print(
                f"{5 * '-'} Running Scenario: Perturbation {perturbation} on {scale} {5 * '-'}"
            )

            scenario = Scenario(
                waypoints=waypoints,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )

            # simulate the scenario
            isSuccess = self.simulator.simulate_scanario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation, perturb=perturb,
                model_drive=test_model, weather=weather, intensity=weather_intensity
            )

            if not isSuccess:
                is_crashed = True

            # check if we drop the scenario, we never remove the empty perturbation
            # for comparison reasons
            # print("outcome.isSuccess is: ", outcome.isSuccess)
            # if not outcome.isSuccess or perturb==False:
            if perturb == False:
                perturbations.remove(perturbation)
            else:
                index += 1

            if len(perturbations) == 0:
                # all perturbations resulted in failures
                # we will still have one perturbation here because we never
                # drop the empty perturbation
                break
            if index == len(perturbations):
                # we increment the scale, so start with the first perturbation again
                index = 0
                scale += 1

            if scale > 4:
                # we went through all scales
                break

        if not is_crashed: # isSuccess: no crash from all perturbation
            lack_perturb_scale.append(perturbation)
            print("For those types that do not have sufficient perturbation effects: ", lack_perturb_scale)

        # TODO: print command line summary of benchmarking process
        del image_perturbation
        del scenario
        del road_generator

        # tear down the simulator
        self.simulator.tear_down()
