from PIL import Image
import os


def display_jp2_image(image_path):
    try:
        # Open the JP2 image file
        img = Image.open(image_path)

        # Display the image
        img.show()

    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
    except Exception as e:
        print("An error occurred:", e)

    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jp2_image_path",
        type=str,
        help="Path to the JP2 image file",
        default=os.path.join(
            os.path.expanduser("~"), "helioviewer", "20101201T000000Z_source_19.jp2"
        ),
    )
    args = parser.parse_args()

    display_jp2_image(args.jp2_image_path)
