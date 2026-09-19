from types import SimpleNamespace

import graphiti_core.helpers as graphiti_helpers

import gatekg.api.kg as kg_module


def test_kg_applies_graphiti_concurrency_limit(monkeypatch):
    original_limit = graphiti_helpers.SEMAPHORE_LIMIT

    class FakeGraphiti:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeNeo4jDriver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    try:
        fake_config = SimpleNamespace(
            graphiti=SimpleNamespace(max_coroutines=3),
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
        )
        monkeypatch.setattr(kg_module, "config", fake_config)
        monkeypatch.setattr(kg_module, "Graphiti", FakeGraphiti)
        monkeypatch.setattr(kg_module, "Neo4jDriver", FakeNeo4jDriver)
        monkeypatch.setattr(kg_module, "build_graphiti_llm_client", lambda: "llm")
        monkeypatch.setattr(kg_module, "build_graphiti_embedder", lambda: "embedder")
        monkeypatch.setattr(kg_module, "build_graphiti_cross_encoder", lambda: None)
        monkeypatch.setattr(kg_module, "default_graph_search_config", lambda: "search-config")

        kg = kg_module.KG()

        assert graphiti_helpers.SEMAPHORE_LIMIT == 3
        assert kg.client.kwargs["max_coroutines"] == 3
    finally:
        graphiti_helpers.SEMAPHORE_LIMIT = original_limit
