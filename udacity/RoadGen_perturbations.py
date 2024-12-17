from udacity.perturbation.perturbationdrive import PerturbationDrive
from perturbationdrive.RoadGenerator.CustomRoadGenerator import CustomRoadGenerator
from examples.udacity.udacity_simulator import UdacitySimulator
from examples.models.dave2_agent import Dave2Agent
import traceback
from datetime import datetime

# road_angles_list=[]
# road_segments_list=[]
# road_types_list=[]
#
# def generate_random_roads(angles_number=20):
#     road_angles_list = []
#     road_segments_list = []
#     road_types_list=[]
#     # for i in range(10, 1, -1):
#     i=7
#     road_angles = [random.randint(-2 * i, 2 * i) for _ in range(angles_number)]
#     road_segments = [random.randint(20, 30) for _ in range(angles_number)]
#     road_angles_list.append(road_angles)
#     road_segments_list.append(road_segments)
#     road_types_list.append("random")
#     if i != 0:
#         road_angles = [random.randint(-4 * i, 4 * i) for _ in range(angles_number)]
#         road_segments = [random.randint(20, 30) for _ in range(angles_number)]
#         road_angles_list.append(road_angles)
#         road_segments_list.append(road_segments)
#         road_types_list.append("random")
#
#         road_angles = [random.randint(-6 * i, 6 * i) for _ in range(angles_number)]
#         road_segments = [random.randint(20, 30) for _ in range(angles_number)]
#         road_angles_list.append(road_angles)
#         road_segments_list.append(road_segments)
#         road_types_list.append("random")
#     return road_angles_list, road_segments_list, road_types_list
#
# road_angles_list,road_segments_list,road_types_list=generate_random_roads()
# print(len(road_angles_list))

road_angles_list=[[-37, 44, 53, -60, 69, 17, -1, -32, -43, 57], [-52, 41, -20, 21, 63, 50, -22, -29, 8, -6], [8, 27, -35, -2, -28, -2, -13, 49, 2, 11], [35, -20, -23, 9, -24, -14, 43, 26, -23, -3], [9, -9, -5, -26, -28, -20, 9, -27, 5, -34], [36, 32, 28, 24, 20, 16, 12, 8, 4, 0], [-36, -32, -28, -24, -20, -16, -12, -8, -4, 0], [28, 24, 20, 16, 12, 8, 4, 0, -4, -8], [-28, -24, -20, -16, -12, -8, -4, 0, 4, 8], [35, 35, 35, 35, 35, 35, 35, 35], [-35, -35, -35, -35, -35, -35, -35, -35], [25, 25, 25, 25, 25, 25, 25, 25], [-25, -25, -25, -25, -25, -25, -25, -25], [45, -45, 45, -45, 45, -45, 45, -45, 45, -45], [35, -35, 35, -35, 35, -35, 35, -35, 35, -35]]
road_segments_list=[[20, 22, 29, 24, 23, 21, 26, 27, 21, 20], [20, 26, 21, 21, 23, 20, 23, 29, 24, 27], [27, 28, 22, 20, 30, 25, 23, 23, 29, 26], [22, 29, 26, 27, 30, 27, 24, 25, 20, 27], [25, 21, 28, 20, 30, 21, 26, 29, 25, 30], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [10, 10, 10, 10, 10, 10, 10, 10], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20], [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]]
road_types_list=['random', 'random', 'random', 'random', 'random', 'curvy', 'curvy', 'curvy', 'curvy', 'simple', 'simple', 'simple', 'simple', 'swiggly', 'swiggly']


try:
    i=1
    simulator = UdacitySimulator(
        simulator_exe_path="./udacity/road_generator_simulator/binary.x86_64",
        host="127.0.0.1",
        port=9091
    )
    road_generator = CustomRoadGenerator(num_control_nodes=len(road_angles_list[i]))
    ads = Dave2Agent(model_path="./checkpoints/roadGen_trained.h5")
    model = ads.model
    attention_map = {
        "map": "grad_cam",
        "model": model,
        "threshold": 0.1,
        "layer": "conv2d_5",
    }
    # functions=['canny_edges_mapping', 'contrast', 'cutout_filter', 'defocus_blur', 'dotted_lines_mapping', 'dynamic_lightning_filter', 'dynamic_object_overlay', 'dynamic_rain_filter', 'dynamic_raindrop_filter', 'dynamic_smoke_filter', 'dynamic_snow_filter', 'dynamic_sun_filter', 'effects_attention_regions', 'elastic', 'fog_filter', 'frost_filter', 'gaussian_blur', 'gaussian_noise', 'glass_blur', 'grayscale_filter', 'histogram_equalisation', 'impulse_noise', 'increase_brightness', 'jpeg_filter', 'low_pass_filter', 'motion_blur', 'object_overlay',  'phase_scrambling', 'pixelate', 'poisson_noise', 'posterize_filter', 'reflection_filter', 'rotate_image', 'sample_pairing_filter', 'saturation_decrease_filter', 'saturation_filter', 'scale_image', 'sharpen_filter', 'shear_image' '']
    functions = ['contrast']
    print("All: ", len(functions))
    functions_done=[]
    functions_todo = [element for element in functions if element not in functions_done]
    print("Remaining: ",len(functions_todo)," names: ", functions_todo)

    benchmarking_obj = PerturbationDrive(simulator, ads)
    print(f"{5 * '#'} Testing road {i} {5 * '#'}")
    time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    benchmarking_obj.grid_seach(
        perturbation_functions=functions,
        attention_map=attention_map,
        road_generator=road_generator,
        road_angles=road_angles_list[i],
        road_segments=road_segments_list[i],
        image_size=(160, 320), # model is trained under this shape of images
        test_model=True,
        perturb=True,
    )
    print(f"{5 * '#'} Finished Testing road {i} {5 * '#'}")
except Exception as e:
    print(
        f"{5 * '#'} Udacity Error: Exception type: {type(e).__name__}, \nError message: {e}\nTract {traceback.print_exc()} {5 * '#'} "
    )