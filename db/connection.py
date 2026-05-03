import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

_pool = None


def _get_pool():
    global _pool
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
        conn.rollback()
        raise
    finally:
        return_conn(conn)
