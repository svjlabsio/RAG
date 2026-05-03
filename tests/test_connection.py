import os
import pytest
from db.connection import get_conn, return_conn

def test_connection_executes_query():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS val")
            row = cur.fetchone()
        assert row[0] == 1
    finally:
        return_conn(conn)


def test_connection_pool_returns_connection():
    conn1 = get_conn()
    return_conn(conn1)
    conn2 = get_conn()  # must succeed; proves conn1 was returned to pool
    try:
        with conn2.cursor() as cur:
            cur.execute("SELECT 1")
            assert cur.fetchone()[0] == 1
    finally:
        return_conn(conn2)
