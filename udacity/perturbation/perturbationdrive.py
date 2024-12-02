from .Simulator.Simulator import PerturbationSimulator
from .AutomatedDrivingSystem.ADS import ADS
from .imageperturbations import ImagePerturbation, get_functions_from_module
from .Simulator.Scenario import Scenario, ScenarioOutcome, OfflineScenarioOutcome
from .RoadGenerator.RoadGenerator import RoadGenerator
from .utils.logger import ScenarioOutcomeWriter, OfflineScenarioOutcomeWriter

from typing import List, Union, Dict, Tuple
import copy
import os
import json
import cv2
import time

from ..ase_simulation.agent import SupervisedAgent


class PerturbationDrive:
    """
    Simulator independent ADS robustness benchmarking
    """

    def __init__(
        self,
        simulator: PerturbationSimulator,
        ads: Union[SupervisedAgent, None],
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

    def grid_search_tracks(
            self,
            perturbation_functions: List[str],
            attention_map: Dict = {},
            simulator: Union[str, None] = "",
            log_dir: Union[str, None] = "logs.json",
            overwrite_logs: bool = True,
            image_size: Tuple[float, float] = (240, 320),
            test_model: bool = False,
            perturb: bool = False,
            track: Union[str, None] = "lake",
            weather: Union[str, None] = "sunny",
            daytime: Union[str, None] = "day"
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Basically, what we have done in image perturbations up until now but in a single nice function wrapped

        If log_dir is none, we return the scenario outcomes
        """
        image_perturbation = ImagePerturbation(
            funcs=perturbation_functions,
            attention_map=attention_map,
            image_size=image_size,
        )
        scale = 0
        index = 0
        # outcomes: List[ScenarioOutcome] = []
        # if log_dir is None:
        #     print("No log directory")
        # else:
        #     scenario_writer = ScenarioOutcomeWriter(log_dir, overwrite_logs)

        perturbations: List[str] = []

        perturbations: List[str] = copy.deepcopy(perturbation_functions)
        # populate all perturbations
        if len(perturbations) == 0:
            perturbation_fns = get_functions_from_module(
                "perturbationdrive.perturbationfuncs"
            )
            perturbations = list(map(lambda f: f.__name__, perturbation_fns))

        # we append the empty perturbation here
        perturbations.append("")

        # set up simulator
        self.simulator.connect()
        # wait 1 second for connection to build up
        time.sleep(1)

        # grid search loop
        while True:
            #print(perturbations)
            perturbation = perturbations[index]
            print(
                f"{5 * '-'} Running Scenario: Perturbation {perturbation} on {scale} {5 * '-'}"
            )

            scenario = Scenario(
                waypoints=simulator,
                perturbation_function=perturbation,
                perturbation_scale=scale,
            )

            # simulate the scenario
            outcome = self.simulator.simulate_scenario_tracks(
                self.ads,
                scenario=scenario,
                perturbation_controller=image_perturbation,
                perturb=perturb,
                model_drive=test_model,
                track=track,
                weather=weather,
                daytime=daytime,
            )

            # check if we drop the scenario, we never remove the empty perturbation
            # for comparison reasons
            if not outcome.isSuccess or perturb == False:
                perturbations.remove(perturbation)
            else:
                index += 1
            scenario_writer.write([outcome])

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

        # TODO: print command line summary of benchmarking process
        del image_perturbation
        del scenario

        # tear down the simulator
        self.simulator.tear_down()

    def simulate_scenarios(
        self,
        scenarios: List[Scenario],
        attention_map: Dict = {},
        log_dir: Union[str, None] = "logs.json",
        overwrite_logs: bool = True,
        image_size: Tuple[float, float] = (240, 320),
    ) -> Union[None, List[ScenarioOutcome]]:
        """
        Basically, what we have done with open sbt

        If log_dir is none, we return the scenario outcomes
        """
        #print(attention_map)

        # get all perturbations to set up this object
        perturbations: List[str] = []
        for scenario in scenarios:
            perturbations.append(scenario.perturbation_function)

        image_perturbation = ImagePerturbation(
            funcs=perturbations, attention_map=attention_map, image_size=image_size
        )
        # sim is setup in main to get starting pos

        outcomes: List[ScenarioOutcome] = []
        # iterate over all scenarios
        for scenario in scenarios:
            outcome = self.simulator.simulate_scenario(
                self.ads, scenario=scenario, perturbation_controller=image_perturbation, perturb=True, model_drive=True
            )
            outcomes.append(outcome)
            time.sleep(2.0)

        del image_perturbation
        del perturbations
        # tear sim down
        if log_dir is not None:
            scenario_writer = ScenarioOutcomeWriter(log_dir, overwrite_logs)
            scenario_writer.write(outcomes)
            del scenario_writer
        return outcomes
