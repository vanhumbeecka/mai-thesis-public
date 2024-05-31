from hvpy.utils import create_layers
from hvpy.datasource import DataSource
import math

from hvpy import createScreenshot, DataSource, create_events, create_layers, EventType
from datetime import datetime, timedelta

from matplotlib import cm


def print_layer():
    layer_string = create_layers([(DataSource.HMI_MAG, 0)])
    print(layer_string)


def screenshot(w, filename, layers=create_layers([(DataSource.HMI_MAG, 100)])):
    screenshot_location = createScreenshot(
        date=datetime.now(),
        layers=layers,
        eventLabels=False,
        watermark=False,
        imageScale=1,
        x1=-1 * w,
        y1=-1 * w,
        x2=w,
        y2=w,
        filename=filename,
        overwrite=True,
    )
    print(screenshot_location)


def combine_images(images, spacing):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, len(images), figsize=(len(images) * 5, 5))

    for i, image_name in enumerate(images):
        image = plt.imread(image_name)
        ax[i].imshow(image, cmap="gray")
        ax[i].axis("off")

    fig.subplots_adjust(wspace=spacing)
    fig.tight_layout()
    plt.savefig("combined_images.png")
    # plt.show()


def run(download=True):
    if download:
        w1 = 1000
        w2 = round(math.sin(45 * math.pi / 180) * w1)
        w3 = round(math.sin(45 * math.pi / 180) * w1 * 0.9) # 636 arcseconds
        screenshot(w1, "screenshot_1")
        screenshot(w2, "screenshot_2")
        screenshot(w3, "screenshot_3")

    images = ["screenshot_1.png", "screenshot_2.png", "screenshot_3.png"]
    spacing = 0.1

    combine_images(images, spacing)


if __name__ == "__main__":
    # run(True)
    
    w = 1200
    screenshot(w, "screenshot_hmi_mag", layers=create_layers([(DataSource.HMI_MAG, 100)]))
    screenshot(w, "screenshot_hmi_cont", layers=create_layers([(DataSource.HMI_INT, 100)]))
    screenshot(w, "screenshot_aia_171", layers=create_layers([(DataSource.AIA_171, 100)]))
    screenshot(w, "screenshot_aia_335", layers=create_layers([(DataSource.AIA_335, 100)]))

    combine_images(["screenshot_hmi_mag.png", "screenshot_hmi_cont.png", "screenshot_aia_171.png", "screenshot_aia_335.png"], 0.1)

