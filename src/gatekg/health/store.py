from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gatekg.paths import runtime_paths

HEALTH_SCHEMA_VERSION = "kg-health-v3"
DEFAULT_HEALTH_DB_PATH = runtime_paths().cache / "kg_health" / "snapshots.sqlite3"


class KGHealthSnapshotStore:
    def __init__(self, db_path: str | Path = DEFAULT_HEALTH_DB_PATH):
        self.db_path = Path(db_path)

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM health_snapshots WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchone()
        return self._row_to_snapshot(row)

    def get_baseline_snapshot(self, base_group_id: str) -> dict[str, Any] | None:
        return self.get_snapshot(_baseline_snapshot_id(base_group_id))

    def list_snapshots(
        self,
        *,
        base_group_id: str | None = None,
        write_group_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        where_clauses: list[str] = []
        params: list[Any] = []
        if base_group_id is not None:
            where_clauses.append("base_group_id = ?")
            params.append(base_group_id)
        if write_group_id is not None:
            where_clauses.append("write_group_id = ?")
            params.append(write_group_id)

        query = "SELECT * FROM health_snapshots"
        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)
        query += " ORDER BY created_at ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [snapshot for row in rows if (snapshot := self._row_to_snapshot(row)) is not None]

    def get_metric_history(
        self,
        *,
        base_group_id: str,
        write_group_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        history: dict[str, list[dict[str, Any]]] = {}
        for snapshot in self.list_snapshots(
            base_group_id=base_group_id,
            write_group_id=write_group_id,
        ):
            metric_scores = snapshot["metric_scores"]
            metric_raw_values = snapshot["metric_raw_values"]
            for metric_name, health_score in metric_scores.items():
                history.setdefault(metric_name, []).append(
                    {
                        "snapshot_id": snapshot["snapshot_id"],
                        "snapshot_kind": snapshot["snapshot_kind"],
                        "created_at": snapshot["created_at"],
                        "health_score": health_score,
                        "raw_value": metric_raw_values.get(metric_name),
                    }
                )
        return history

    def upsert_baseline_snapshot(
        self,
        *,
        base_group_id: str,
        group_ids: list[str],
        report: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        write_group_id: str | None = None,
    ) -> dict[str, Any]:
        snapshot_id = _baseline_snapshot_id(base_group_id)
        timestamp = _utc_now()
        metadata_payload = metadata or {}
        metric_scores = _metric_scores(report)
        metric_raw_values = _metric_raw_values(report)
        health_score_list = _health_score_list(report)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO health_snapshots (
                    snapshot_id,
                    snapshot_kind,
                    base_group_id,
                    write_group_id,
                    group_ids_json,
                    average_health_score,
                    metric_scores_json,
                    metric_raw_values_json,
                    health_score_list_json,
                    report_json,
                    metadata_json,
                    evaluator_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    write_group_id = excluded.write_group_id,
                    group_ids_json = excluded.group_ids_json,
                    average_health_score = excluded.average_health_score,
                    metric_scores_json = excluded.metric_scores_json,
                    metric_raw_values_json = excluded.metric_raw_values_json,
                    health_score_list_json = excluded.health_score_list_json,
                    report_json = excluded.report_json,
                    metadata_json = excluded.metadata_json,
                    evaluator_version = excluded.evaluator_version,
                    updated_at = excluded.updated_at
                """,
                (
                    snapshot_id,
                    "baseline",
                    base_group_id,
                    write_group_id,
                    _dump_json(group_ids),
                    float(report["average_health_score"]),
                    _dump_json(metric_scores),
                    _dump_json(metric_raw_values),
                    _dump_json(health_score_list),
                    _dump_json(report),
                    _dump_json(metadata_payload),
                    HEALTH_SCHEMA_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return self.get_snapshot(snapshot_id) or {}

    def insert_run_snapshot(
        self,
        *,
        base_group_id: str,
        write_group_id: str,
        group_ids: list[str],
        report: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        snapshot_id = f"run:{uuid.uuid4().hex}"
        timestamp = _utc_now()
        metric_scores = _metric_scores(report)
        metric_raw_values = _metric_raw_values(report)
        health_score_list = _health_score_list(report)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO health_snapshots (
                    snapshot_id,
                    snapshot_kind,
                    base_group_id,
                    write_group_id,
                    group_ids_json,
                    average_health_score,
                    metric_scores_json,
                    metric_raw_values_json,
                    health_score_list_json,
                    report_json,
                    metadata_json,
                    evaluator_version,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    "run",
                    base_group_id,
                    write_group_id,
                    _dump_json(group_ids),
                    float(report["average_health_score"]),
                    _dump_json(metric_scores),
                    _dump_json(metric_raw_values),
                    _dump_json(health_score_list),
                    _dump_json(report),
                    _dump_json(metadata or {}),
                    HEALTH_SCHEMA_VERSION,
                    timestamp,
                    timestamp,
                ),
            )
            connection.commit()
        return self.get_snapshot(snapshot_id) or {}

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS health_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                snapshot_kind TEXT NOT NULL,
                base_group_id TEXT NOT NULL,
                write_group_id TEXT,
                group_ids_json TEXT NOT NULL,
                average_health_score REAL NOT NULL,
                metric_scores_json TEXT NOT NULL DEFAULT '{}',
                metric_raw_values_json TEXT NOT NULL DEFAULT '{}',
                health_score_list_json TEXT NOT NULL DEFAULT '[]',
                report_json TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                evaluator_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._ensure_column(
            connection,
            column_name="metric_scores_json",
            column_sql="TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            connection,
            column_name="metric_raw_values_json",
            column_sql="TEXT NOT NULL DEFAULT '{}'",
        )
        self._ensure_column(
            connection,
            column_name="health_score_list_json",
            column_sql="TEXT NOT NULL DEFAULT '[]'",
        )
        self._backfill_metric_fields(connection)
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_health_snapshots_kind_group
            ON health_snapshots (snapshot_kind, base_group_id, write_group_id, created_at)
            """
        )
        return connection

    def _row_to_snapshot(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        report_payload = json.loads(row["report_json"])
        return {
            "snapshot_id": row["snapshot_id"],
            "snapshot_kind": row["snapshot_kind"],
            "base_group_id": row["base_group_id"],
            "write_group_id": row["write_group_id"],
            "group_ids": json.loads(row["group_ids_json"]),
            "average_health_score": row["average_health_score"],
            "metric_scores": _load_json_value(
                row["metric_scores_json"],
                fallback=_metric_scores(report_payload),
                prefer_fallback_when_empty=True,
            ),
            "metric_raw_values": _load_json_value(
                row["metric_raw_values_json"],
                fallback=_metric_raw_values(report_payload),
                prefer_fallback_when_empty=True,
            ),
            "health_score_list": _load_json_value(
                row["health_score_list_json"],
                fallback=_health_score_list(report_payload),
                prefer_fallback_when_empty=True,
            ),
            "report": report_payload,
            "metadata": json.loads(row["metadata_json"]),
            "evaluator_version": row["evaluator_version"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        *,
        column_name: str,
        column_sql: str,
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(health_snapshots)").fetchall()
        }
        if column_name in columns:
            return
        connection.execute(f"ALTER TABLE health_snapshots ADD COLUMN {column_name} {column_sql}")
        connection.commit()

    def _backfill_metric_fields(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT snapshot_id, report_json, metric_scores_json, metric_raw_values_json, health_score_list_json
            FROM health_snapshots
            """
        ).fetchall()

        updates: list[tuple[str, str, str, str]] = []
        for row in rows:
            report_payload = json.loads(row["report_json"])
            metric_scores = _metric_scores(report_payload)
            metric_raw_values = _metric_raw_values(report_payload)
            health_score_list = _health_score_list(report_payload)

            resolved_metric_scores = _load_json_value(
                row["metric_scores_json"],
                fallback=metric_scores,
                prefer_fallback_when_empty=True,
            )
            resolved_metric_raw_values = _load_json_value(
                row["metric_raw_values_json"],
                fallback=metric_raw_values,
                prefer_fallback_when_empty=True,
            )
            resolved_health_score_list = _load_json_value(
                row["health_score_list_json"],
                fallback=health_score_list,
                prefer_fallback_when_empty=True,
            )

            if (
                resolved_metric_scores == _safe_json_load(row["metric_scores_json"])
                and resolved_metric_raw_values == _safe_json_load(row["metric_raw_values_json"])
                and resolved_health_score_list == _safe_json_load(row["health_score_list_json"])
            ):
                continue

            updates.append(
                (
                    _dump_json(resolved_metric_scores),
                    _dump_json(resolved_metric_raw_values),
                    _dump_json(resolved_health_score_list),
                    row["snapshot_id"],
                )
            )

        if not updates:
            return

        connection.executemany(
            """
            UPDATE health_snapshots
            SET
                metric_scores_json = ?,
                metric_raw_values_json = ?,
                health_score_list_json = ?
            WHERE snapshot_id = ?
            """,
            updates,
        )
        connection.commit()


def _baseline_snapshot_id(base_group_id: str) -> str:
    return f"baseline:{base_group_id}"


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json_value(
    value: str | None,
    *,
    fallback: Any,
    prefer_fallback_when_empty: bool = False,
) -> Any:
    if value is None:
        return fallback
    parsed_value = json.loads(value)
    if (
        prefer_fallback_when_empty
        and _is_empty_json_value(parsed_value)
        and not _is_empty_json_value(fallback)
    ):
        return fallback
    return parsed_value


def _metric_scores(report: dict[str, Any]) -> dict[str, float]:
    metric_scores = report.get("metric_scores")
    if isinstance(metric_scores, dict):
        return {str(name): float(score) for name, score in metric_scores.items()}
    return {
        str(metric["name"]): float(metric["health_score"])
        for metric in report.get("metrics", [])
        if "name" in metric and "health_score" in metric
    }


def _metric_raw_values(report: dict[str, Any]) -> dict[str, float]:
    metric_raw_values = report.get("metric_raw_values")
    if isinstance(metric_raw_values, dict):
        return {str(name): float(raw_value) for name, raw_value in metric_raw_values.items()}
    return {
        str(metric["name"]): float(metric["raw_value"])
        for metric in report.get("metrics", [])
        if "name" in metric and "raw_value" in metric
    }


def _health_score_list(report: dict[str, Any]) -> list[float]:
    health_score_list = report.get("health_score_list")
    if isinstance(health_score_list, list):
        return [float(score) for score in health_score_list]
    return [
        float(metric["health_score"])
        for metric in report.get("metrics", [])
        if "health_score" in metric
    ]


def _safe_json_load(value: str | None) -> Any:
    if value is None:
        return None
    return json.loads(value)


def _is_empty_json_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False
