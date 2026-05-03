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
