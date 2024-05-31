from datetime import timedelta
import pandas as pd
import os
from typing import Tuple, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FlareEvent:
    archive_id: str
    start_time: datetime
    peak_time: datetime
    end_time: datetime
    flare_class: str


@dataclass
class FlareRecord:
    flare_event: FlareEvent
    timestamps: List[datetime]


class SolarConfig:
    """
    Configuration for the solar flare data. This class is used to store the configuration for the solar flare data.
    """

    def __init__(
        self,
        local_data_path: str,
        image_data_path: str,
        image_dimension: Tuple[int, int, int],
        channels_name: List[str],
        timedimension_max_timedelta: timedelta,
        flare_data_file: str = "flare_data.pkl",
    ) -> None:
        self.data_path = local_data_path
        self.image_path = image_data_path
        self.image_dimension = image_dimension
        self.channels_name = channels_name
        self.timedimension_max_timedelta = timedimension_max_timedelta
        self.flare_data_file = flare_data_file

        if os.path.exists(self.data_path) is False:
            raise ValueError(f"Invalid data path: {self.data_path}")
        if os.path.exists(self.image_path) is False:
            raise ValueError(f"Invalid image path: {self.image_path}")
        if os.path.exists(self.annotations_path) is False:
            raise ValueError(f"Cannot find annotations file path: {self.data_path}")

    @property
    def annotations_path(self) -> str:
        return os.path.join(self.data_path, self.flare_data_file)

    @property
    def depth_dimension(self) -> int:
        return self.image_dimension[0]
    
    @property
    def pl_path(self) -> str:
        return os.path.join(self.data_path, "lightning_logs")

    def get_annotations(self) -> List[FlareEvent]:
        raw = pd.read_pickle(self.annotations_path)
        if isinstance(raw, pd.DataFrame):
            return [
                FlareEvent(
                    archive_id=str(index),
                    start_time=row["beginTime"].to_pydatetime(),
                    peak_time=row["peakTime"].to_pydatetime(),
                    end_time=row["endTime"].to_pydatetime(),
                    flare_class=row["classType"],
                )
                for index, row in raw.iterrows()
            ]
        # Convert to DataFrame: List[Tuple[FlareEvent, List[datetime]]]
        if isinstance(raw, list):
            return [evts for evts, _ in raw]

        raise ValueError(f"Invalid annotations file format: {self.annotations_path}")

    def get_image_dir(self) -> str:
        return self.image_path

    # @property
    # def tensorboard_log_dir(self) -> str:
    #     return os.path.join(self.data_path, "logs")

    @property
    def data_dir(self) -> str:
        return self.data_path

    @property
    def root_dir(self) -> str:
        return self.data_path
