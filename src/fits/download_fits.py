# Description: Download FITS files from the NASA GOES database and store the data in a SQLite database.

import numpy as np
from astropy.io import fits
from datetime import datetime, timedelta
from typing import Iterable, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import os
from contextlib import contextmanager
from joblib import Parallel, delayed

from common.utils import generate_datetime_range, init_logger
from common.db import sqlite_connection

logger = init_logger(__name__)

BASE_URL = "https://umbra.nascom.nasa.gov/goes/fits/"
current_directory = os.path.dirname(os.path.abspath(__file__))
SQLITE_DB = os.path.join(current_directory, "fits.db")


def get_file_url(date: datetime, satellite: str = "15") -> str:
    fits_url = BASE_URL + f"{date.year}/"
    fits_url += f"go{satellite}{date.year:04d}{date.month:02d}{date.day:02d}.fits"
    return fits_url


def plot_fits_values(values: Iterable[tuple[datetime, float]]) -> None:
    # Extract the timestamps and flux values from the iterable
    timestamps, fluxes = zip(*values)

    # Convert the timestamps to hours since the first timestamp
    timestamps = np.array(timestamps)
    timestamps = (timestamps - timestamps[0]).astype("timedelta64[m]").astype(int) / 60

    # Plot the flux values
    plt.plot(timestamps, fluxes)
    plt.xlabel("Time")
    plt.ylabel("Flux (W/m^2)")
    plt.title("Flux over Time from FITS File")
    plt.show()


def get_flux_values(fits_url) -> Iterable[tuple[datetime, float]]:
    # Download the FITS file
    try:
        with fits.open(fits_url, use_fsspec=True, mode="readonly") as hdul:
            # Extract date and time strings from FITS header
            date_obs_str = hdul[0].header["DATE-OBS"]
            time_obs_str = hdul[0].header["TIME-OBS"]
            datetime_obs = datetime.strptime(
                date_obs_str + " " + time_obs_str, "%d/%m/%Y %H:%M:%S.%f"
            )

            # Extract time data from FITS file
            timestamps_utc = []
            time_data = hdul["FLUXES"].data["TIME"][0]
            for time in time_data:
                timestamps_utc.append(datetime_obs + timedelta(seconds=time))

            # Extract flux data from FITS file
            flux_data = hdul["FLUXES"].data["FLUX"][0, :, 0]

        return zip(timestamps_utc, flux_data)
    except Exception as e:
        logger.error("Failed to download FITS file: %s", fits_url)
        return []


def store(data: pd.DataFrame, filename: str, conn: sqlite3.Connection):
    """Insert the data into the table"""
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR IGNORE INTO metadata (filename, last_updated)
        VALUES (?, ?)
        """,
        (filename, datetime.now()),
    )
    last_rowid = cur.lastrowid

    data["metadata_rowid"] = last_rowid
    data.to_sql("fits", conn, if_exists="append", index=False)

    # Commit the changes
    conn.commit()
    cur.close()


def init_db(conn: sqlite3.Connection):
    # Create the table if it does not exist
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            filename TEXT,
            last_updated DATETIME,
            UNIQUE(filename)
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS fits (
            timestamp DATETIME,
            flux REAL,
            metadata_rowid INTEGER,
            UNIQUE(timestamp),
            FOREIGN KEY(metadata_rowid) REFERENCES metadata(rowid)
        )
        """
    )

    conn.commit()


def run_single_date(date: datetime, satellite: str, conn: sqlite3.Connection):
    log = init_logger(__name__)

    # Get the URL of the FITS file
    fits_url = get_file_url(date, satellite)

    # Check if the data is already in the database
    cur = conn.cursor()
    result = cur.execute(
        """
        SELECT filename FROM metadata WHERE filename = ?
        """,
        (fits_url,),
    )
    if result.fetchone():
        log.info("Data already exists: %s", date)
        return
    cur.close()

    data = get_flux_values(fits_url)
    if not data:
        log.warning("No data was returned for %s", date)
        return

    df = pd.DataFrame(data, columns=["timestamp", "flux"])

    # resample to 1 minute
    resampled = df.resample("1min", on="timestamp").mean()

    # filter out values of different days to prevent overlap
    resampled_filtered = resampled.loc[date.date().strftime("%Y-%m-%d")]
    resampled_filtered.reset_index(inplace=True)

    store(resampled_filtered, fits_url, conn)  # type: ignore
    log.info("Data stored: %s", date)


def single_day(date: datetime, satellite: str):
    with sqlite_connection(SQLITE_DB) as conn:
        run_single_date(date, satellite, conn)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        type=str,
        help="Start date of the FITS file to download",
        default="2012-01-01",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        help="End date of the FITS file to download",
        default="2012-01-30",
    )
    parser.add_argument(
        "--satellite",
        type=str,
        help="Satellite of the FITS file to download",
        default="15",
    )
    args = parser.parse_args()
    start_date: datetime = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date: datetime = datetime.strptime(args.end_date, "%Y-%m-%d")

    date_range = generate_datetime_range(start_date, end_date, timedelta(days=1))

    with sqlite_connection(SQLITE_DB) as conn:
        logger.info("Initializing database")
        init_db(conn)

    logger.info("Downloading FITS files")
    Parallel(n_jobs=8)(delayed(single_day)(d, args.satellite) for d in date_range)
    logger.info("Done")
