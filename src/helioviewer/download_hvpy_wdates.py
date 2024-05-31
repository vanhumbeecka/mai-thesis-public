from hvpy.utils import create_layers
from hvpy.datasource import DataSource
import math
import numpy as np

from hvpy import (
    takeScreenshot,
    createScreenshot,
    DataSource,
    create_events,
    create_layers,
    EventType,
)
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from hvpy import getClosestImage

from typing import List, Tuple


@dataclass
class FlareEvent:
    archive_id: str
    start_time: datetime
    peak_time: datetime
    end_time: datetime
    flare_class: str

def extract_datetime(response) -> datetime:
    return datetime.strptime(response["date"], "%Y-%m-%d %H:%M:%S")

def calculate_timestamps_for_flare(
    flare: FlareEvent, source_id: int, delta_before_start: timedelta = timedelta(hours=2)
) -> Tuple[bool, List[datetime]]:
    # Get the flare event
    frame_start = flare.start_time - delta_before_start

    closest_flair_start_response = getClosestImage(
        date=flare.start_time,
        sourceId=source_id,
    )
    closest_flair_start = extract_datetime(closest_flair_start_response)
    closest_flair_peak_response = getClosestImage(
        date=flare.peak_time,
        sourceId=source_id,
    )
    closest_flair_peak = extract_datetime(closest_flair_peak_response)
    closest_frame_start_response = getClosestImage(
        date=frame_start,
        sourceId=source_id,
    )
    closest_frame_start = extract_datetime(closest_frame_start_response)

    if (flare.start_time - closest_flair_start) > timedelta(minutes=30):
        return False, []

    if (flare.peak_time - closest_flair_peak) > timedelta(minutes=30):
        return False, []

    if (frame_start - closest_frame_start) > timedelta(minutes=30):
        return False, []

    delta = closest_flair_peak - closest_frame_start
    if delta < timedelta(minutes=30):
        return False, []

    time_values = np.linspace(0, 1, 6)
    date_values = [
        closest_frame_start + (closest_flair_peak - closest_frame_start) * t
        for t in time_values
    ]

    result = []
    result.append(date_values[0])
    for date in date_values[1:-1]:
        res = getClosestImage(date=date, sourceId=source_id)
        cl = extract_datetime(res)

        if abs(date - cl) > timedelta(minutes=30):
            return False, []
        result.append(cl)
    result.append(date_values[-1])

    return True, result


def screenshot(w, filename):
    res = takeScreenshot(
        date=datetime.now(),
        layers=create_layers([(DataSource.HMI_MAG, 100)]),
        eventLabels=False,
        watermark=False,
        imageScale=1,
        x1=-1 * w,
        y1=-1 * w,
        x2=w,
        y2=w,
    )
    print(res)


if __name__ == "__main__":
    event: FlareEvent = FlareEvent(
        archive_id="ivo://helio-informatics.org/FL_SECstandard_20120111_192534_20111001085600",
        start_time=datetime.strptime("2011-10-01 08:56:00.000", "%Y-%m-%d %H:%M:%S.%f"),
        peak_time=datetime.strptime("2011-10-01 09:59:00.000", "%Y-%m-%d %H:%M:%S.%f"),
        end_time=datetime.strptime("2011-10-01 10:17:00.000", "%Y-%m-%d %H:%M:%S.%f"),
        flare_class="M1.2",
    )
    valid, result = calculate_timestamps_for_flare(event, 19)
    print(valid, result)
