import pytest
from pathlib import Path

from llm_drift.fingerprint import Fingerprint
from llm_drift.store import BaselineStore, SQLiteStore


def _fp(text: str = "hello") -> Fingerprint:
    return Fingerprint(
        embedding=[0.1, 0.2, 0.3],
        token_count=1,
        format="plain",
        raw_output=text,
    )


# ---------------------------------------------------------------------------
# SQLiteStore — save / load
# ---------------------------------------------------------------------------

async def test_sqlite_store_save_and_load(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    fp = _fp("test output")
    await store.save("my-suite", [fp])
    loaded = await store.load_latest("my-suite")
    assert loaded is not None
    assert loaded[0].raw_output == "test output"
    assert loaded[0].token_count == fp.token_count


async def test_sqlite_store_load_returns_none_when_empty(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    result = await store.load_latest("nonexistent-suite")
    assert result is None


async def test_sqlite_store_list_runs_ordered(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    await store.save("s", [_fp("a")])
    await store.save("s", [_fp("b")])
    await store.save("s", [_fp("c")])
    runs = await store.list_runs("s")
    assert len(runs) == 3
    # newest first
    assert runs[0]["created_at"] >= runs[1]["created_at"] >= runs[2]["created_at"]


async def test_sqlite_store_creates_file_on_first_use(tmp_path: Path):
    db_path = tmp_path / "subdir" / "baselines.db"
    assert not db_path.exists()
    store = SQLiteStore(db_path)
    await store.save("s", [_fp()])
    assert db_path.exists()


# ---------------------------------------------------------------------------
# SQLiteStore — results
# ---------------------------------------------------------------------------

async def test_sqlite_store_save_and_list_results(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    run_id = await store.save("s", [_fp()])
    await store.save_result("s", run_id, drift_score=0.12, drifted=False)
    results = await store.list_results("s")
    assert len(results) == 1
    assert results[0]["drift_score"] == pytest.approx(0.12)
    assert results[0]["drifted"] is False


async def test_sqlite_store_list_results_empty(tmp_path: Path):
    store = SQLiteStore(tmp_path / "test.db")
    results = await store.list_results("no-suite")
    assert results == []


# ---------------------------------------------------------------------------
# ABC enforcement
# ---------------------------------------------------------------------------

def test_custom_store_satisfies_abc():
    class MyStore(BaselineStore):
        async def save(self, suite_name, fingerprints): return "id"
        async def load_latest(self, suite_name): return None
        async def list_runs(self, suite_name): return []
        async def save_result(self, suite_name, run_id, drift_score, drifted): pass
        async def list_results(self, suite_name): return []

    assert MyStore()  # instantiates without error


def test_incomplete_store_raises_type_error():
    with pytest.raises(TypeError):
        class BadStore(BaselineStore):
            pass
        BadStore()
