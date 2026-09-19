from __future__ import annotations

import pytest

import gatekg.health.tracking as tracking
from gatekg.health.evaluator import KGHealthEvaluator
from gatekg.health.store import HEALTH_SCHEMA_VERSION, KGHealthSnapshotStore


def test_health_snapshot_store_round_trip(tmp_path):
    store = KGHealthSnapshotStore(tmp_path / "gatekg.health.sqlite3")

    baseline_snapshot = store.upsert_baseline_snapshot(
        base_group_id="mcp-cache_base",
        group_ids=["mcp-cache_base"],
        report={
            "average_health_score": 0.8,
            "metrics": [
                {"name": "density", "health_score": 0.8, "raw_value": 0.2, "details": {}},
                {"name": "duplicate_ratio", "health_score": 1.0, "raw_value": 0.0, "details": {}},
            ],
            "metric_scores": {"density": 0.8, "duplicate_ratio": 1.0},
            "metric_raw_values": {"density": 0.2, "duplicate_ratio": 0.0},
            "health_score_list": [0.8, 1.0],
            "raw_summary": {"total_nodes": 3},
        },
        metadata={"cache_model_key": "google:kg"},
    )

    assert baseline_snapshot["snapshot_kind"] == "baseline"
    assert baseline_snapshot["group_ids"] == ["mcp-cache_base"]
    assert baseline_snapshot["report"]["raw_summary"]["total_nodes"] == 3
    assert baseline_snapshot["metric_scores"] == {"density": 0.8, "duplicate_ratio": 1.0}
    assert baseline_snapshot["health_score_list"] == [0.8, 1.0]

    updated_baseline = store.upsert_baseline_snapshot(
        base_group_id="mcp-cache_base",
        group_ids=["mcp-cache_base"],
        report={
            "average_health_score": 0.9,
            "metrics": [
                {"name": "density", "health_score": 0.9, "raw_value": 0.25, "details": {}},
            ],
            "metric_scores": {"density": 0.9},
            "metric_raw_values": {"density": 0.25},
            "health_score_list": [0.9],
            "raw_summary": {"total_nodes": 4},
        },
        metadata={"cache_model_key": "google:kg"},
    )

    assert updated_baseline["snapshot_id"] == baseline_snapshot["snapshot_id"]
    assert updated_baseline["average_health_score"] == 0.9
    assert updated_baseline["report"]["raw_summary"]["total_nodes"] == 4
    assert updated_baseline["metric_scores"] == {"density": 0.9}

    run_snapshot = store.insert_run_snapshot(
        base_group_id="mcp-cache_base",
        write_group_id="mcp-work_run",
        group_ids=["mcp-work_run", "mcp-cache_base"],
        report={
            "average_health_score": 0.7,
            "metrics": [
                {"name": "density", "health_score": 0.7, "raw_value": 0.15, "details": {}},
                {"name": "duplicate_ratio", "health_score": 0.6, "raw_value": 0.04, "details": {}},
            ],
            "metric_scores": {"density": 0.7, "duplicate_ratio": 0.6},
            "metric_raw_values": {"density": 0.15, "duplicate_ratio": 0.04},
            "health_score_list": [0.7, 0.6],
            "raw_summary": {"total_nodes": 5},
        },
        metadata={"model_key": "google:gemini"},
    )

    assert run_snapshot["snapshot_kind"] == "run"
    assert run_snapshot["write_group_id"] == "mcp-work_run"
    assert run_snapshot["group_ids"] == ["mcp-work_run", "mcp-cache_base"]
    metric_history = store.get_metric_history(base_group_id="mcp-cache_base")
    assert [entry["health_score"] for entry in metric_history["density"]] == [0.9, 0.7]
    assert [entry["health_score"] for entry in metric_history["duplicate_ratio"]] == [0.6]


def test_ensure_baseline_health_snapshot_reuses_existing_snapshot(tmp_path, monkeypatch):
    store = KGHealthSnapshotStore(tmp_path / "gatekg.health.sqlite3")
    evaluate_calls: list[list[str]] = []

    class FakeKG:
        async def get_entity_nodes_by_group_ids(self, group_ids):
            return []

        async def get_entity_edges_by_group_ids(self, group_ids):
            return []

    async def fake_evaluate_kg_health(kg, *, group_ids, **kwargs):
        evaluate_calls.append(list(group_ids))
        return {
            "average_health_score": 0.75,
            "metrics": [],
            "metric_scores": {"density": 0.75},
            "metric_raw_values": {"density": 0.22},
            "health_score_list": [0.75],
            "raw_summary": {"group_ids": list(group_ids)},
        }

    monkeypatch.setattr(tracking, "evaluate_kg_health", fake_evaluate_kg_health)

    fake_kg = FakeKG()
    first_snapshot = _run(
        tracking.ensure_baseline_health_snapshot(
            kg=fake_kg,
            base_group_id="mcp-cache_base",
            cache_model_key="google:kg",
            context_chunk_count=2,
            store=store,
        )
    )
    second_snapshot = _run(
        tracking.ensure_baseline_health_snapshot(
            kg=fake_kg,
            base_group_id="mcp-cache_base",
            cache_model_key="google:kg",
            context_chunk_count=2,
            store=store,
        )
    )
    forced_snapshot = _run(
        tracking.ensure_baseline_health_snapshot(
            kg=fake_kg,
            base_group_id="mcp-cache_base",
            cache_model_key="google:kg",
            context_chunk_count=2,
            force=True,
            store=store,
        )
    )

    assert first_snapshot is not None
    assert second_snapshot is not None
    assert forced_snapshot is not None
    assert evaluate_calls == [["mcp-cache_base"], ["mcp-cache_base"]]


def test_record_run_health_snapshot_persists_combined_group_ids(tmp_path, monkeypatch):
    store = KGHealthSnapshotStore(tmp_path / "gatekg.health.sqlite3")

    class FakeKG:
        async def get_entity_nodes_by_group_ids(self, group_ids):
            return []

        async def get_entity_edges_by_group_ids(self, group_ids):
            return []

    async def fake_evaluate_kg_health(kg, *, group_ids, **kwargs):
        return {
            "average_health_score": 0.61,
            "metrics": [],
            "metric_scores": {"density": 0.61, "duplicate_ratio": 1.0},
            "metric_raw_values": {"density": 0.18, "duplicate_ratio": 0.0},
            "health_score_list": [0.61, 1.0],
            "raw_summary": {"group_ids": list(group_ids)},
        }

    monkeypatch.setattr(tracking, "evaluate_kg_health", fake_evaluate_kg_health)

    snapshot = _run(
        tracking.record_run_health_snapshot(
            kg=FakeKG(),
            base_group_id="mcp-cache_base",
            write_group_id="mcp-work_run",
            model_key="google:gemini",
            kg_model_key="google:kg",
            question="question",
            prompt="prompt",
            prediction="prediction",
            store=store,
        )
    )

    assert snapshot is not None
    assert snapshot["snapshot_kind"] == "run"
    assert snapshot["group_ids"] == ["mcp-work_run", "mcp-cache_base"]
    assert snapshot["metadata"]["question"] == "question"
    assert snapshot["metadata"]["prompt_char_count"] == 6
    assert snapshot["metadata"]["prediction_char_count"] == 10
    assert snapshot["metric_scores"] == {"density": 0.61, "duplicate_ratio": 1.0}


def test_ensure_baseline_health_snapshot_recomputes_stale_snapshot_version(tmp_path, monkeypatch):
    import sqlite3

    store = KGHealthSnapshotStore(tmp_path / "gatekg.health.sqlite3")
    evaluate_calls: list[list[str]] = []

    class FakeKG:
        async def get_entity_nodes_by_group_ids(self, group_ids):
            return []

        async def get_entity_edges_by_group_ids(self, group_ids):
            return []

    store.upsert_baseline_snapshot(
        base_group_id="mcp-cache_base",
        group_ids=["mcp-cache_base"],
        report={
            "average_health_score": 0.25,
            "metrics": [],
            "metric_scores": {"density": 0.25},
            "metric_raw_values": {"density": 0.4},
            "health_score_list": [0.25],
            "raw_summary": {"group_ids": ["mcp-cache_base"]},
        },
        metadata={"cache_model_key": "google:kg"},
    )

    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE health_snapshots SET evaluator_version = ? WHERE snapshot_id = ?",
            ("kg-health-v1", "baseline:mcp-cache_base"),
        )
        connection.commit()

    async def fake_evaluate_kg_health(kg, *, group_ids, **kwargs):
        evaluate_calls.append(list(group_ids))
        return {
            "average_health_score": 0.75,
            "metrics": [],
            "metric_scores": {"density": 0.75},
            "metric_raw_values": {"density": 0.22},
            "health_score_list": [0.75],
            "raw_summary": {"group_ids": list(group_ids)},
        }

    monkeypatch.setattr(tracking, "evaluate_kg_health", fake_evaluate_kg_health)

    snapshot = _run(
        tracking.ensure_baseline_health_snapshot(
            kg=FakeKG(),
            base_group_id="mcp-cache_base",
            cache_model_key="google:kg",
            context_chunk_count=2,
            store=store,
        )
    )

    assert snapshot is not None
    assert evaluate_calls == [["mcp-cache_base"]]
    assert snapshot["average_health_score"] == 0.75
    assert snapshot["evaluator_version"] == HEALTH_SCHEMA_VERSION


def test_store_backfills_metric_fields_for_partial_rows(tmp_path):
    import json
    import sqlite3

    db_path = tmp_path / "partial-health.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE health_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            snapshot_kind TEXT NOT NULL,
            base_group_id TEXT NOT NULL,
            write_group_id TEXT,
            group_ids_json TEXT NOT NULL,
            average_health_score REAL NOT NULL,
            report_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            evaluator_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO health_snapshots (
            snapshot_id,
            snapshot_kind,
            base_group_id,
            write_group_id,
            group_ids_json,
            average_health_score,
            report_json,
            metadata_json,
            evaluator_version,
            created_at,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "baseline:mcp-cache_base",
            "baseline",
            "mcp-cache_base",
            None,
            json.dumps(["mcp-cache_base"]),
            0.5,
            json.dumps(
                {
                    "average_health_score": 0.5,
                    "metrics": [
                        {
                            "name": "density",
                            "health_score": 0.5,
                            "raw_value": 0.1,
                            "details": {},
                        }
                    ],
                    "raw_summary": {"total_nodes": 2},
                }
            ),
            json.dumps({}),
            "kg-health-v0",
            "2026-04-09T00:00:00+00:00",
            "2026-04-09T00:00:00+00:00",
        ),
    )
    connection.commit()
    connection.close()

    store = KGHealthSnapshotStore(db_path)
    snapshot = store.get_baseline_snapshot("mcp-cache_base")

    assert snapshot is not None
    assert snapshot["metric_scores"] == {"density": 0.5}
    assert snapshot["metric_raw_values"] == {"density": 0.1}
    assert snapshot["health_score_list"] == [0.5]


def test_duplicate_ratio_falls_back_to_normalized_names_without_gds():
    class FakeKG:
        async def get_entity_nodes_by_group_ids(self, group_ids):
            return [
                {"uuid": "n1", "group_id": "group-a", "name": "José Luis Picardo"},
                {"uuid": "n2", "group_id": "group-a", "name": "Jose Luis Picardo"},
                {"uuid": "n3", "group_id": "group-a", "name": "Trinidad de Ribera"},
            ]

        async def get_entity_edges_by_group_ids(self, group_ids):
            return []

    report = _run(
        KGHealthEvaluator(FakeKG()).evaluate(
            group_ids=["group-a"],
            duplicate_similarity_threshold=0.95,
        )
    )

    duplicate_metric = next(metric for metric in report.metrics if metric.name == "duplicate_ratio")
    assert duplicate_metric.raw_value == 1 / 3
    assert duplicate_metric.health_score == 0.0
    assert duplicate_metric.details["method"] == "normalized_name_exact"
    assert duplicate_metric.details["duplicate_name_group_count"] == 1
    assert duplicate_metric.details["threshold_applied"] is False

    coverage_metric = next(
        metric for metric in report.metrics if metric.name == "name_embedding_coverage"
    )
    assert coverage_metric.raw_value == 0.0
    assert coverage_metric.health_score == 0.0

    semantic_duplicate_metric = next(
        metric for metric in report.metrics if metric.name == "semantic_duplicate_ratio"
    )
    assert semantic_duplicate_metric.raw_value == 0.0
    assert semantic_duplicate_metric.health_score == 0.0
    assert semantic_duplicate_metric.details["reason"] == "insufficient_name_embeddings"


def test_semantic_metrics_use_name_embeddings_when_available():
    class FakeKG:
        async def get_entity_nodes_by_group_ids(self, group_ids, with_embeddings=False):
            nodes = [
                {
                    "uuid": "n1",
                    "group_id": "group-a",
                    "name": "Alice Johnson",
                    "name_embedding": [1.0, 0.0],
                },
                {
                    "uuid": "n2",
                    "group_id": "group-a",
                    "name": "Alicia Johnson",
                    "name_embedding": [0.999, 0.001],
                },
                {
                    "uuid": "n3",
                    "group_id": "group-a",
                    "name": "Bob Smith",
                    "name_embedding": [0.0, 1.0],
                },
            ]
            if with_embeddings:
                return nodes
            return [
                {key: value for key, value in node.items() if key != "name_embedding"}
                for node in nodes
            ]

        async def get_entity_edges_by_group_ids(self, group_ids, with_embeddings=False):
            return []

    report = _run(KGHealthEvaluator(FakeKG()).evaluate(group_ids=["group-a"]))

    coverage_metric = next(
        metric for metric in report.metrics if metric.name == "name_embedding_coverage"
    )
    assert coverage_metric.raw_value == 1.0
    assert coverage_metric.health_score == 1.0
    assert coverage_metric.details["embedded_entity_count"] == 3

    semantic_duplicate_metric = next(
        metric for metric in report.metrics if metric.name == "semantic_duplicate_ratio"
    )
    assert semantic_duplicate_metric.raw_value == pytest.approx(1 / 3)
    assert semantic_duplicate_metric.health_score == 0.0
    assert semantic_duplicate_metric.details["duplicate_pairs"] == 1
    assert semantic_duplicate_metric.details["embedded_entity_count"] == 3
    assert semantic_duplicate_metric.details["compared_pairs"] == 3
    assert (
        semantic_duplicate_metric.details["duplicate_candidates"][0]["source_name"]
        == "Alice Johnson"
    )
    assert (
        semantic_duplicate_metric.details["duplicate_candidates"][0]["target_name"]
        == "Alicia Johnson"
    )


def _run(coro):
    import asyncio

    return asyncio.run(coro)
