from perturbationdrive import PerturbationDrive, CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

if __name__ == '__main__':
    simulator = UdacitySimulator(
        simulator_exe_path="./udacity/udacity_sim_road_generator/udacity_sim_linux.x86_64",
        host="127.0.0.1",
        port=9091
    )
    ads = Dave2Agent(
        model_path="./checkpoints/original_extended3_extended5.h5")

    benchmarking_obj = PerturbationDrive(simulator, ads)

    road_angles = [10, 20, 30, 40, 0, -10, -20, -30, 0, 10, 40, 10, 0, -10, -40, -10]
    road_segments = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles))

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model = ads.model
    attention_map = {
        "map": "grad_cam",  # `grad_cam` or `vanilla`
        "model": model,
        "threshold": 0.1,
        "layer": "conv2d_5",
    }

    # "Sun", "Rain", "Snow", "Fog"
    weather = "Sun"
    intens = 20

    try:
        # start the benchmarking
        benchmarking_obj.grid_search(
            perturbation_functions=[],
            attention_map={},
            road_generator=road_generator,
            road_angles=road_angles,
            road_segments=road_segments,
            log_dir=f"/home/jiaqq/Documents/PerturbationDrive-Replication/output_logs/model_nominal/udacity_logs_{time}.json",
            overwrite_logs=True,
            image_size=(240, 320),
            test_model=True,
            perturb=False,
            weather=weather,
            weather_intensity=intens
        )
        print(f"{5 * '#'} Finished Running Udacity Sim {5 * '#'}")

    except Exception as e:
        print(
            f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
        )