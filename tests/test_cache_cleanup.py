from gatekg.utils.cache_cleanup import build_cache_cleanup_plan
from gatekg.utils.config import (
    Config,
    GoogleConfig,
    GraphitiRuntimeConfig,
    LoggingConfig,
    OllamaConfig,
)
from gatekg.utils.context_cache import build_context_group_id
from gatekg.utils.eval_runner import ExperimentEntry


def _dummy_config() -> Config:
    return Config(
        provider="google",
        google=GoogleConfig(
            llm_model_name="gemini-2.5-pro",
            kg_llm_model_name="gemini-3-flash-preview",
            emb_model_name="gemini-embedding-001",
        ),
        ollama=OllamaConfig(
            llm_model_name="qwen3.5:35b",
            emb_model_name="nomic-embed-text:latest",
        ),
        logging=LoggingConfig(),
        graphiti=GraphitiRuntimeConfig(),
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        neo4j_database=None,
    )


def test_build_cache_cleanup_plan_uses_context_group_id(monkeypatch):
    entries = [
        ExperimentEntry(
            entry_id="a",
            question="q1",
            gold="g1",
            context_chunks=["alpha", "beta"],
        ),
        ExperimentEntry(
            entry_id="b",
            question="q2",
            gold="g2",
            context_chunks=["gamma"],
        ),
    ]
    monkeypatch.setattr(
        "gatekg.utils.cache_cleanup._load_entries_for_task",
        lambda task_name, entry_limit=0: entries,
    )

    plan = build_cache_cleanup_plan(
        tasks=["loogle"],
        llms=["gemini-3-flash-preview"],
        config=_dummy_config(),
    )

    assert plan.task_filters == ("loogle",)
    assert plan.cache_model_keys == ("google:gemini-3-flash-preview",)
    assert len(plan.selections) == 1
    assert plan.selections[0].group_ids == (
        build_context_group_id(
            cache_model_key="google:gemini-3-flash-preview",
            context_chunks=["alpha", "beta"],
        ),
        build_context_group_id(
            cache_model_key="google:gemini-3-flash-preview",
            context_chunks=["gamma"],
        ),
    )


def test_build_cache_cleanup_plan_preserves_full_cache_model_key(monkeypatch):
    entries = [
        ExperimentEntry(
            entry_id="a",
            question="q1",
            gold="g1",
            context_chunks=["alpha"],
        ),
    ]
    monkeypatch.setattr(
        "gatekg.utils.cache_cleanup._load_entries_for_task",
        lambda task_name, entry_limit=0: entries,
    )

    plan = build_cache_cleanup_plan(
        tasks=["hotpotqa"],
        llms=["google:gemini-2.5-pro"],
        config=_dummy_config(),
    )

    assert plan.cache_model_keys == ("google:gemini-2.5-pro",)
    assert plan.group_ids == (
        build_context_group_id(
            cache_model_key="google:gemini-2.5-pro",
            context_chunks=["alpha"],
        ),
    )


def test_build_cache_cleanup_plan_creates_cross_product_of_tasks_and_llms(monkeypatch):
    entries_by_task = {
        "hotpotqa": [
            ExperimentEntry(
                entry_id="a",
                question="q1",
                gold="g1",
                context_chunks=["alpha"],
            ),
        ],
        "loogle": [
            ExperimentEntry(
                entry_id="b",
                question="q2",
                gold="g2",
                context_chunks=["beta"],
            ),
        ],
    }
    monkeypatch.setattr(
        "gatekg.utils.cache_cleanup._load_entries_for_task",
        lambda task_name, entry_limit=0: entries_by_task[task_name],
    )

    plan = build_cache_cleanup_plan(
        tasks=["hotpotqa,loogle"],
        llms=["gemini-3-flash-preview", "google:gemini-2.5-pro"],
        config=_dummy_config(),
    )

    assert len(plan.selections) == 4
    assert plan.cache_model_keys == (
        "google:gemini-3-flash-preview",
        "google:gemini-2.5-pro",
    )
