# mai-thesis-public
Public resources related to my thesis for Master in Artificial Intelligence.
It contains the most important scripts and data used in the thesis.

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

## 2. DONKI flare database

Scripts for downloading and visualizing the DONKI flare database are located in the `src/DONKI` folder.
You can run the `src/DONKI/donki.ipynb` jupyter notebook 

* to explore the flare database
* download HMI magnetograms for the these flare events
* visualize some magnetograms

The `src/DONKI/pytorch_dataset.py` contains helper functions for constructing datasets from magnetograms, as well as the `HelioViewerDataset` class that can be used to create a structured pytorch dataset from the downloaded magnetograms.

