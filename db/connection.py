import os
import threading
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = pool.ThreadedConnectionPool(1, 5, os.environ["DATABASE_URL"])
    return _pool


def get_conn():
    return _get_pool().getconn()


def return_conn(conn):
    _get_pool().putconn(conn)


@contextmanager
def db_conn():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        if conn.closed:
            _get_pool().putconn(conn, close=True)
        else:
            return_conn(conn)
