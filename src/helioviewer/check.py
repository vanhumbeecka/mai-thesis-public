import os
from datetime import datetime, timedelta
from turtle import st
from typing import Iterable
from tqdm import tqdm
import itertools


def generate_datetime_range(start_date: datetime, end_date: datetime) -> list[datetime]:
    delta = timedelta(hours=1)
    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date)
        current_date += delta

    return date_list


def generate_unique_combinations(
    date_range: list[datetime], source_ids: list[int]
) -> Iterable[tuple[datetime, int]]:
    return itertools.product(date_range, source_ids)


def check_missing_files(directory_path):
    missing_files = []

    # Loop through dates from 2012-01-01 to 2018-12-31
    start_date = datetime(2012, 12, 1)
    end_date = datetime(2018, 12, 1)
    delta = timedelta(hours=1)

    combos = list(
        generate_unique_combinations(
            generate_datetime_range(start_date=start_date, end_date=end_date), [18, 19]
        )
    )

    current_date = start_date
    for current_date, source in tqdm(combos):
        file_name = current_date.strftime(
            "%Y%m%dT%H0000Z_source_{:02d}.jp2".format(source)
        )
        file_path = os.path.join(directory_path, file_name)

        if not os.path.exists(file_path):
            missing_files.append(file_name)

    return missing_files


if __name__ == "__main__":
    # Replace this path with your specific directory path
    directory_to_check = "/Volumes/home/ml/datasets/helioviewer"

    missing_files_list = check_missing_files(directory_to_check)

    if missing_files_list:
        print("Missing files:")
        for missing_file in missing_files_list:
            print(missing_file)
    else:
        print("All files are present within the specified range.")
