from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS stages (
    item_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (item_id, stage)
);
"""


class StateDB:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def done(self, item_id: str, stage: str) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT status FROM stages WHERE item_id=? AND stage=?",
                (item_id, stage),
            ).fetchone()
        return bool(row and row[0] == "done")

    def set(self, item_id: str, stage: str, status: str, detail: str | None = None) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO stages(item_id, stage, status, detail)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(item_id, stage) DO UPDATE SET
                    status=excluded.status,
                    detail=excluded.detail,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (item_id, stage, status, detail),
            )
