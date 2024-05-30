import sqlite3
import os
from contextlib import contextmanager
import numpy as np

@contextmanager
def sqlite_connection(db_name: str):
    conn = sqlite3.connect(db_name)
    sqlite3.register_adapter(np.int64, lambda val: int(val))
    sqlite3.register_adapter(np.float64, lambda val: float(val))
    sqlite3.register_adapter(np.float32, lambda val: float(val))
    try:
        yield conn
    finally:
        conn.close()