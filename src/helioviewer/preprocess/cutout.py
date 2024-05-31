from PIL import Image
import os

from numpy import full
import glob
from tqdm import tqdm
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


def crop_image(input_path, output_path, target_size):
    # Skip if output path already exists
    if os.path.exists(output_path):
        return
    # Open the input image
    img = Image.open(input_path)

    # Get dimensions of the input image
    width, height = img.size

    # Calculate the coordinates to crop the center of the image
    left = (width - target_size) / 2
    top = (height - target_size) / 2
    right = (width + target_size) / 2
    bottom = (height + target_size) / 2

    # Crop the image
    cropped_img = img.crop((left, top, right, bottom))

    # Create subfolders for the output path if necessary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the cropped image
    cropped_img.save(output_path)


def run():
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_path, "../../../data")
    # print(data_dir)
    input_folder = "flare_images"
    output_folder = "flare_images_preprocessed"
    target_size = 636 * 2

    # Get a list of all image files in the flare_images folder and subfolders
    image_files = glob.glob(
        os.path.join(data_dir, input_folder, "**/*.png"), recursive=True
    )

    # Define the number of processes to use
    num_processes = multiprocessing.cpu_count()

    # Create a thread pool executor
    with ThreadPoolExecutor(max_workers=num_processes) as executor:
        # Define a function to process each image file
        def process_image(image_file):
            # Get the relative path of the image file
            relative_path = os.path.relpath(
                image_file, os.path.join(data_dir, input_folder)
            )

            # Create the corresponding output path
            output_path = os.path.join(data_dir, output_folder, relative_path)

            # Crop the image
            crop_image(image_file, output_path, target_size)

        # Use the thread pool executor to process the image files in parallel
        list(tqdm(executor.map(process_image, image_files), total=len(image_files)))


if __name__ == "__main__":
    run()
