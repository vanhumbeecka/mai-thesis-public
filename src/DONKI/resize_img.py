import os
from venv import logger
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm

from common.utils import init_logger



def resize_image(flare_id, source, input_folder, output_folder):
    logger = init_logger(flare_id)
    logger.info(f"Resizing images for {flare_id}...")
    images = os.listdir(os.path.join(input_folder, flare_id, source))

    for image_filename in images:
        os.makedirs(os.path.join(output_folder, flare_id, source), exist_ok=True)
        output_path = os.path.join(output_folder, flare_id, source, image_filename)
        if not os.path.exists(output_path):
            # Resize the image to 512x512
            image_path = os.path.join(input_folder, flare_id, source, image_filename)
            try:
                image = Image.open(image_path)
                resized_image = image.resize((512, 512))
                # Save the resized image to the output folder
                resized_image.save(output_path)
                # Close the image
                image.close()
            except Exception as e:
                logger.error(f"Error resizing {image_path}: {str(e)}")
            

def resize_images_parallel(input_folder: str, output_folder: str, sources: list[str]):
    all_flare_ids = os.listdir(input_folder)

    for source in sources:
        Parallel(n_jobs=-1)(
            delayed(resize_image)(flare_id, source, input_folder, output_folder)
            for flare_id in all_flare_ids
        )
    print("Done.")

# def resize_images(input_folder: str, output_folder: str, sources: list[str]):
#     """Resize all images in the input folder and save them to the output folder"""
#     for flare_id in tqdm(os.listdir(input_folder)):
#         for source in sources:
#             images: list[str] = os.listdir(os.path.join(input_folder, flare_id, source))

#             for image_filename in images:
#                 os.makedirs(os.path.join(output_folder, flare_id, source), exist_ok=True)
#                 output_path = os.path.join(
#                     output_folder, flare_id, source, image_filename
#                 )
#                 if not os.path.exists(output_path):
#                     # Resize the image to 512x512
#                     image_path = os.path.join(
#                         input_folder, flare_id, source, image_filename
#                     )
#                     image = Image.open(image_path)
#                     resized_image = image.resize((512, 512))
#                     # Save the resized image to the output folder
#                     resized_image.save(output_path)
#                     # Close the image
#                     image.close()


if __name__ == "__main__":
    current_dir = os.path.abspath(os.path.dirname(__file__))
    # data_path = os.path.join(current_dir, "../../data")
    data_path = "/mnt/d/datasets/helioviewer"
    image_path = os.path.join(data_path, "images")
    output_folder = os.path.join(data_path, "images_resized")
    os.makedirs(output_folder, exist_ok=True)

    sources = ["source_19"]

    resize_images_parallel(image_path, output_folder, sources)
