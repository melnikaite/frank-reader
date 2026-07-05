import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  source_type TEXT NOT NULL,
  source_name TEXT NOT NULL,
  target_lang TEXT NOT NULL,
  force_vision INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL,
  total_pages INTEGER,
  error TEXT
);

CREATE TABLE IF NOT EXISTS pages (
  job_id TEXT NOT NULL REFERENCES jobs(id),
  page_number INTEGER NOT NULL,
  status TEXT NOT NULL,
  cache_key TEXT,
  result_json TEXT,
  error TEXT,
  PRIMARY KEY (job_id, page_number)
);

CREATE TABLE IF NOT EXISTS llm_cache (
  cache_key TEXT PRIMARY KEY,
  result_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.data_dir / "frank.db", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(
            "PRAGMA journal_mode=WAL;"
            "PRAGMA synchronous=NORMAL;"
            "PRAGMA foreign_keys=ON;"
            "PRAGMA busy_timeout=5000;"
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def job_dir(self, job_id: str) -> Path:
        d = self.data_dir / "jobs" / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def create_job(
        self,
        job_id: str,
        source_type: str,
        source_name: str,
        target_lang: str,
        force_vision: bool = False,
    ) -> None:
        self._conn.execute(
            "INSERT INTO jobs (id, created_at, source_type, source_name, target_lang, "
            "force_vision, status, total_pages, error) VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL, NULL)",
            (job_id, _now(), source_type, source_name, target_lang, int(force_vision)),
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_jobs(self) -> list[dict]:
        rows = self._conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def set_job_status(
        self,
        job_id: str,
        status: str,
        error: str | None = None,
        total_pages: int | None = None,
    ) -> None:
        fields = ["status = ?"]
        params: list = [status]
        if error is not None:
            fields.append("error = ?")
            params.append(error)
        if total_pages is not None:
            fields.append("total_pages = ?")
            params.append(total_pages)
        params.append(job_id)
        self._conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)
        self._conn.commit()

    def upsert_page(
        self,
        job_id: str,
        page_number: int,
        status: str,
        result_json: str | None = None,
        error: str | None = None,
        cache_key: str | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO pages (job_id, page_number, status, cache_key, result_json, error)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(job_id, page_number) DO UPDATE SET
                 status=excluded.status, cache_key=excluded.cache_key,
                 result_json=excluded.result_json, error=excluded.error""",
            (job_id, page_number, status, cache_key, result_json, error),
        )
        self._conn.commit()

    def get_pages(self, job_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM pages WHERE job_id = ? ORDER BY page_number", (job_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def cache_get(self, cache_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT result_json FROM llm_cache WHERE cache_key = ?", (cache_key,)
        ).fetchone()
        return row["result_json"] if row else None

    def cache_put(self, cache_key: str, result_json: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, result_json, created_at) VALUES (?, ?, ?)",
            (cache_key, result_json, _now()),
        )
        self._conn.commit()

    def mark_running_as_interrupted(self) -> None:
        self._conn.execute("UPDATE jobs SET status = 'interrupted' WHERE status = 'running'")
        self._conn.commit()
