# mai-thesis-public
Public resources related to my thesis for Master in Artificial Intelligence.
It contains the most important scripts and data used in my thesis.

# Local setup

## Conda

Make sure you have `conda` or `miniconda` installed on your system. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

The `conda.yml` contains a snapshot of the environment that was used for all the python scripts during this thesis. You can create the environment by running the following command:

```bash
conda env create -f conda.yml
```

You might need to adjust the `prefix` in the `conda.yml` file to match your system. Some package may not be compatible with your system, in that case you can remove them from the `conda.yml` file.

## Hardware

The conda environment is set up to use `pytorch` with `cuda` support. 
Make sure you have a compatible GPU and the necessary drivers installed.

# Sections

## 1. Fits files
This section shows how you can download and visualize fits files from the GOES satellites.

You can download the fits files discussed in the thesis with the script located at `src/fits/download_fits.py`.
You can run it with the following commands, and it will store it in a local sqlite database called `fits.db`.

```bash
conda activate ml

# Prints help, showing possible arguments
python src/fits/download_fits.py --help

# Download all fits files between 2012-01-01 and 2012-01-30, from GOES satellite 15
python src/fits/download_fits.py --start_date 2012-01-01 --end 2012-01-30 --satellite 15
```

Next, you can uses the jupyter notebook `src/fits/fits_db.ipynb` to visualize the fits files stored in the sqlite database you populated with the previous script.

## 2. MNIST experiments
This section shows how you can train and evaluate beta-VAE models on the MNIST dataset.
The `src/mnist` folder contains all scripts for training and evaluating the MNIST experiments discussed in the thesis.

## 2.1 Training

The `src/mnist/main.py` will download & train the burgess model on the MNIST dataset. 

```bash
conda activate ml

# show help
python src/mnist/main.py --help

# train the model with options
python src/mnist/main.py --epochs 10 --beta 4
```

Training the model will output 

* the model weights in the `data/mnist-expirements` folder as `*.pt` files.
* KL losses during training which can be used for latent variable visualization in `.txt` files
* reconstruction and sample images after each epoch

This file uses the model defined in `src/mnist/model_burgess.py`. This file can itself be run as a standalone script, to get more context about this model. See the script for more details.

## 2.2 Visualization

The `src/mnist/plot_beta.ipynb` jupiter notebook allows you to visualize the latent space of the trained model. It uses the KL losses stored in the `data/mnist-experiments` folder to plot the latent space of the model. More details can be found in comments in the notebook.

## 3. DONKI flare database

Scripts for downloading and visualizing the DONKI flare database are located in the `src/DONKI` folder.
You can run the `src/DONKI/donki.ipynb` jupyter notebook 

* to explore the flare database
* download HMI magnetograms for the these flare events
* visualize some magnetograms

The `src/DONKI/pytorch_dataset.py` contains helper functions for constructing datasets from magnetograms, as well as the `HelioViewerDataset` class that can be used to create a structured pytorch dataset from the downloaded magnetograms, which are also used in future sections.

## 4. HEK flare database

The `src/sunpy_flares` folder contains scripts for downloading and visualizing the HEK flare database. The `src/sunpy_flares/GOES_flair.ipynb` jupyter notebook will download the HEK flare database and write it to a CSV file called `flares_GOES.csv`.

## 5. Helioviewer scripts

Several scripts allow you to download solar images from the Helioviewer API. These scripts are located in the `src/helioviewer` folder.

### 5.1 Downloading jp2 images for every hour

`src/helioviewer/download_jp2.py` downloads jp2 images from the Helioviewer API. You can call the script with `--help` to see the possible arguments. (!) Beware, this will download a lot of data. Also, it doesn't take into account the flare event timestamps. It downloads an image for every hour in the given time range.

```bash
conda activate ml

# show help
python src/helioviewer/download_jp2.py --help

# download jp2 images for source ids 18 and 19
python src/helioviewer/download_jp2.py --source_ids 18 19
```

### 5.2 Downloading jp2 images based on HEK flare events

The `src/helioviewer/download_cutouts.ipynb` jupiter notebook does 2 things:

* it will do some preprocessing on the images: Based on the previous constructed csv file `flares_GOES.csv`, it looks if enough image data is available for each flare event by querying the Helioviewer API. Based on the results, it will construct a `valid_flares` datastructure and store it in a pickle file called `valid_flares_10h.pickle`. This file is used in the next section.

* it will download jp2 images from the Helioviewer API based on this 'valid flares' datastructure. It will download images for each flare event, between 10 hours before peak flare event up until peak flare. The images will be stored in your `data/flare_images` directory in the right structure: `<flare_name>/<source_id>/<timestamp>.jp2`. This structure is necessary for the `HelioViewerDataset` class to process the images in a PyTorch dataset.

* Next, you can use the `src/helioviewer/preprocess/cutout.py` standalone script to further preprocess these images stored in `data/flare_images`. The script in particular will crop the images (as discussed in the thesis) to a fixed size and store them in a new directory `data/flare_images_preprocessed`. 


## 6. PyTorch model training


