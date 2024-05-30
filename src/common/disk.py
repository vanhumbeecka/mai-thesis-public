import os
from venv import logger
from diskcache import FanoutCache, Disk
from common.utils import init_logger
from joblib import Memory


logger = init_logger(__name__)


def getCache(scope: str) -> FanoutCache:
    abs_path = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.join(abs_path, "../../data/disk-cache")
    cache_path = os.path.join(base_path, scope)
    logger.info(f"Cache path: {cache_path}")
    return FanoutCache(cache_path, shards=64, disk=Disk, timeout=1, size_limit=3e11)


def getMemCache(scope: str) -> Memory:
    abs_path = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.join(abs_path, "../../data/mem-cache")
    cache_path = os.path.join(base_path, scope)
    return Memory(cache_path, verbose=0)
