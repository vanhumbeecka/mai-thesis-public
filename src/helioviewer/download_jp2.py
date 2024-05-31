from enum import unique
from math import perm
import requests
from datetime import date, datetime, timedelta, timezone
import logging
import os
from joblib import Parallel, delayed
from multiprocessing import Manager
import random
import itertools
from itertools import permutations

from common.utils import generate_datetime_range, init_logger

base_url = "https://api.helioviewer.org/v2/getJP2Image/?date={}&sourceId={}&json={}"
logger = init_logger(__name__)


def save_downloaded_files(file_list, record_file, record_file_lock):
    with record_file_lock:
        with open(record_file, "a") as f:
            for file_name in file_list:
                f.write(file_name + "\n")


def create_file_if_not_exists(file_name):
    if not os.path.exists(file_name):
        try:
            with open(file_name, "w") as f:
                logger.info("Created file: %s", file_name)
        except IOError as e:
            logger.error("Failed to create file: %s", file_name)
            raise e


def check_if_file_downloaded(file_name, record_file, record_file_lock):
    with record_file_lock:
        with open(record_file, "r") as f:
            downloaded_files = f.read().splitlines()
            if file_name in downloaded_files:
                logger.info(f"File {file_name.split('/')[-1]} already downloaded")
                return True


def _format_date(date: datetime) -> str:
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def _format_date_filename_friendly(date: datetime) -> str:
    return date.strftime("%Y%m%dT%H%M%SZ")


def download_image(
    date: datetime, source_id: int, destination_folder: str, file_lock, is_json=False
) -> None:
    logger = init_logger(__name__)
    hour = date.hour
    year = date.year
    record_file = os.path.join(destination_folder, f"downloaded_files_{year}.txt")
    new_date = date.replace(hour=hour, minute=0, second=0, microsecond=0)
    formatted_date = _format_date(new_date)
    formatted_date_filename_friendly = _format_date_filename_friendly(new_date)

    # File naming based on date and sourceId
    filename = f"{formatted_date_filename_friendly}_source_{source_id}.jp2"
    destination_path = os.path.join(destination_folder, filename)

    if check_if_file_downloaded(
        destination_path, record_file=record_file, record_file_lock=file_lock
    ):
        return

    api_url = base_url.format(formatted_date, source_id, "true" if is_json else "false")
    logger.debug('GET "%s"', api_url)
    response = requests.get(api_url)

    if response.status_code == 200:
        with open(destination_path, "wb") as file:
            file.write(response.content)
        logger.info(f"Saved {filename}")
        save_downloaded_files(
            [destination_path], record_file=record_file, record_file_lock=file_lock
        )
    else:
        logger.error(
            f"Failed GET request for date: {formatted_date} and sourceId: {source_id} and status code: {response.status_code}"
        )


def generate_unique_combinations(
    date_range: list[datetime], source_ids: list[int]
) -> list[tuple[datetime, int]]:
    return list(itertools.product(date_range, source_ids))


def get_years_between(start_date: datetime, end_date: datetime) -> list[int]:
    start_year = start_date.year
    end_year = end_date.year

    # Generate a list of years between start and end (inclusive)
    years_between = [year for year in range(start_year, end_year + 1)]

    return years_between


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Helioviewer image downloader\nhttps://api.helioviewer.org/docs/v2/appendix/data_sources.html"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date in YYYY-MM-DDTHH:MM:SS format",
        default="2010-12-01T00:00:00",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date in YYYY-MM-DDTHH:MM:SS format",
        default="2018-12-01T01:00:00",
    )
    parser.add_argument("--num-jobs", type=int, help="Number of jobs", default=4)
    parser.add_argument(
        "--destination-folder",
        type=str,
        help="Destination folder path",
        default=os.path.join(os.path.expanduser("~"), "helioviewer"),
    )
    parser.add_argument(
        "--source_ids",
        type=int,
        nargs="+",
        help="Source IDs to download. 18 and 19 are continuum and magnetogram images respectively. See https://api.helioviewer.org/docs/v2/appendix/data_sources.html",
    )

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    end_date = datetime.strptime(args.end_date, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    num_jobs = args.num_jobs
    destination_folder = args.destination_folder

    logger.info("destination_folder: %s", destination_folder)
    logger.info("starting date: %s", _format_date(start_date))
    logger.info("ending date: %s", _format_date(end_date))
    logger.info("number of jobs: %d", num_jobs)

    date_range = generate_datetime_range(start_date, end_date)
    # source_ids = list(range(8, 20))  # sourceId range from 8 to 19
    source_ids = args.source_ids

    logger.info("number of dates: %d", len(date_range))
    logger.info("number of source_ids: %d", len(source_ids))

    combinations: list[tuple[datetime, int]] = generate_unique_combinations(
        date_range, source_ids
    )
    logger.info("number of combinations: %d", len(combinations))

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    years = get_years_between(start_date, end_date)
    for year in years:
        record_file = os.path.join(destination_folder, f"downloaded_files_{year}.txt")
        create_file_if_not_exists(record_file)

    logger.info("starting download")
    manager = Manager()
    lock = manager.Lock()
    Parallel(n_jobs=num_jobs)(
        delayed(download_image)(*c, destination_folder=destination_folder, file_lock=lock)
        for c in combinations
    )

    logger.info("Done downloading images.")


if __name__ == "__main__":
    main()
