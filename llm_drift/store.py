from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import aiosqlite

from llm_drift.fingerprint import Fingerprint

_CREATE_RUNS = """
    CREATE TABLE IF NOT EXISTS runs (
        run_id    TEXT PRIMARY KEY,
        suite     TEXT NOT NULL,
        created_at TEXT NOT NULL,
        data      TEXT NOT NULL
    )
"""

_CREATE_RESULTS = """
    CREATE TABLE IF NOT EXISTS results (
        run_id      TEXT PRIMARY KEY,
        suite       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        drift_score REAL NOT NULL,
        drifted     INTEGER NOT NULL
    )
"""

_CREATE_RUN_OUTPUTS = """
    CREATE TABLE IF NOT EXISTS run_outputs (
        run_id     TEXT NOT NULL,
        suite      TEXT NOT NULL,
        probe_id   TEXT NOT NULL,
        raw_output TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
"""


class BaselineStore(ABC):
    @abstractmethod
    async def save(self, suite_name: str, fingerprints: List[Fingerprint]) -> str:
        """Persist fingerprints and return a new run_id."""

    @abstractmethod
    async def load_latest(self, suite_name: str) -> Optional[List[Fingerprint]]:
        """Return the most recent baseline fingerprints, or None if none exist."""

    @abstractmethod
    async def list_runs(self, suite_name: str) -> List[dict]:
        """Return run metadata newest-first: [{run_id, created_at}, ...]."""

    @abstractmethod
    async def save_result(self, suite_name: str, run_id: str, drift_score: float, drifted: bool) -> None:
        """Persist a drift run result."""

    @abstractmethod
    async def list_results(self, suite_name: str) -> List[dict]:
        """Return drift results newest-first: [{run_id, created_at, drift_score, drifted}, ...]."""

    @abstractmethod
    async def save_run_outputs(self, suite_name: str, run_id: str, fingerprints: List[Fingerprint]) -> None:
        """Persist the raw probe outputs produced during a drift run (for diffing)."""

    @abstractmethod
    async def load_run_outputs(self, suite_name: str, run_id: Optional[str] = None) -> Optional[List[dict]]:
        """Return [{probe_id, raw_output}, ...] for a run (latest run if run_id is None)."""


class SQLiteStore(BaselineStore):
    def __init__(self, path: str | Path = ".llm-drift/baselines.db"):
        self.path = Path(path)

    def _open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return aiosqlite.connect(self.path)

    async def _ensure_schema(self, db: aiosqlite.Connection) -> None:
        await db.execute(_CREATE_RUNS)
        await db.execute(_CREATE_RESULTS)
        await db.execute(_CREATE_RUN_OUTPUTS)
        await db.commit()

    async def save(self, suite_name: str, fingerprints: List[Fingerprint]) -> str:
        run_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        data = json.dumps([fp.__dict__ for fp in fingerprints])
        async with self._open() as db:
            await self._ensure_schema(db)
            await db.execute(
                "INSERT INTO runs (run_id, suite, created_at, data) VALUES (?, ?, ?, ?)",
                (run_id, suite_name, created_at, data),
            )
            await db.commit()
        return run_id

    async def load_latest(self, suite_name: str) -> Optional[List[Fingerprint]]:
        async with self._open() as db:
            await self._ensure_schema(db)
            async with db.execute(
                "SELECT data FROM runs WHERE suite = ? ORDER BY created_at DESC LIMIT 1",
                (suite_name,),
            ) as cur:
                row = await cur.fetchone()
        if row is None:
            return None
        return [Fingerprint(**r) for r in json.loads(row[0])]

    async def list_runs(self, suite_name: str) -> List[dict]:
        async with self._open() as db:
            await self._ensure_schema(db)
            async with db.execute(
                "SELECT run_id, created_at FROM runs WHERE suite = ? ORDER BY created_at DESC",
                (suite_name,),
            ) as cur:
                rows = await cur.fetchall()
        return [{"run_id": r[0], "created_at": r[1]} for r in rows]

    async def save_result(self, suite_name: str, run_id: str, drift_score: float, drifted: bool) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        async with self._open() as db:
            await self._ensure_schema(db)
            await db.execute(
                "INSERT INTO results (run_id, suite, created_at, drift_score, drifted) VALUES (?, ?, ?, ?, ?)",
                (run_id, suite_name, created_at, drift_score, int(drifted)),
            )
            await db.commit()

    async def list_results(self, suite_name: str) -> List[dict]:
        async with self._open() as db:
            await self._ensure_schema(db)
            async with db.execute(
                "SELECT run_id, created_at, drift_score, drifted FROM results WHERE suite = ? ORDER BY created_at DESC",
                (suite_name,),
            ) as cur:
                rows = await cur.fetchall()
        return [
            {"run_id": r[0], "created_at": r[1], "drift_score": r[2], "drifted": bool(r[3])}
            for r in rows
        ]

    async def save_run_outputs(self, suite_name: str, run_id: str, fingerprints: List[Fingerprint]) -> None:
        created_at = datetime.now(timezone.utc).isoformat()
        rows = [
            (run_id, suite_name, fp.probe_id, fp.raw_output, created_at)
            for fp in fingerprints
        ]
        async with self._open() as db:
            await self._ensure_schema(db)
            await db.executemany(
                "INSERT INTO run_outputs (run_id, suite, probe_id, raw_output, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            await db.commit()

    async def load_run_outputs(self, suite_name: str, run_id: Optional[str] = None) -> Optional[List[dict]]:
        async with self._open() as db:
            await self._ensure_schema(db)
            if run_id is None:
                async with db.execute(
                    "SELECT run_id FROM run_outputs WHERE suite = ? ORDER BY created_at DESC LIMIT 1",
                    (suite_name,),
                ) as cur:
                    row = await cur.fetchone()
                if row is None:
                    return None
                run_id = row[0]
            async with db.execute(
                "SELECT probe_id, raw_output FROM run_outputs WHERE suite = ? AND run_id = ?",
                (suite_name, run_id),
            ) as cur:
                rows = await cur.fetchall()
        if not rows:
            return None
        return [{"probe_id": r[0], "raw_output": r[1]} for r in rows]
