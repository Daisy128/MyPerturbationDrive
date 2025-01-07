import os
import csv
import cv2

def perturb_driving_log(csv_path, data):
    with open(csv_path, 'w', newline='') as csvfile:
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"{csv_path} will be overwritten")


    with open(csv_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if data:
            writer.writerow(data[0].keys())  # column names
            for row in data:
                writer.writerow(row.values())

def save_image(image_path, image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(image_path, image_bgr)
    # print("Path is : "+image_path)
    # if isinstance(image, np.ndarray):
    #     image = Image.fromarray(image)
    #
    # if not isinstance(image, Image.Image):
    #     raise ValueError("Image is not a valid PIL.Image object!")
    #
    # image.save(image_path)