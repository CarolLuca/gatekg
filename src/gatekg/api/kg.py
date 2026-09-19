import asyncio
from datetime import datetime
from typing import Any

import graphiti_core.helpers as graphiti_helpers
from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from gatekg.utils.config import Config, load_config

from .providers import (
    build_graphiti_cross_encoder,
    build_graphiti_embedder,
    build_graphiti_llm_client,
    default_graph_search_config,
)

config: Config = load_config()


def format_fact_result(edge) -> dict[str, Any]:
    result = edge.model_dump(
        mode="json",
        exclude={
            "fact_embedding",
        },
    )
    result.get("attributes", {}).pop("fact_embedding", None)
    return result


class GraphConnection:
    """Shared Graphiti connection lifecycle without operation-specific methods."""

    def __init__(self):
        self._indices_initialized = False
        self._initialize_lock: asyncio.Lock | None = None
        if config.graphiti.max_coroutines is not None:
            graphiti_helpers.SEMAPHORE_LIMIT = config.graphiti.max_coroutines

        if not config.neo4j_uri:
            raise ValueError("NEO4J_URI is required for graph access")
        self.graph_driver = Neo4jDriver(
            uri=config.neo4j_uri,
            user=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database or "neo4j",
        )
        self.client = Graphiti(
            graph_driver=self.graph_driver,
            llm_client=build_graphiti_llm_client(),
            embedder=build_graphiti_embedder(),
            cross_encoder=build_graphiti_cross_encoder(),
            max_coroutines=config.graphiti.max_coroutines,
        )
        self.default_search_config = default_graph_search_config()

    async def initialize(self) -> None:
        if self._indices_initialized:
            return

        if self._initialize_lock is None:
            self._initialize_lock = asyncio.Lock()

        async with self._initialize_lock:
            if self._indices_initialized:
                return
            await self.client.build_indices_and_constraints()
            self._indices_initialized = True

    async def close(self) -> None:
        await self.graph_driver.close()


class KG(GraphConnection):
    """Convenience API for direct graph access."""

    async def init_knowledge(self, supporting_facts: list[str]) -> None:
        for i, fact in enumerate(supporting_facts):
            await self.add_memory(
                memory_name=f"Memory number {i}",
                memory_body=fact,
                memory_description="Supporting facts",
            )

    async def clear_graph(self) -> None:
        await clear_data(self.client.driver)

    async def add_memory(
        self,
        memory_name: str,
        memory_body: str,
        memory_description: str = "",
    ) -> None:
        await self.client.add_episode(
            name=memory_name,
            episode_body=memory_body,
            source_description=memory_description,
            source=EpisodeType.text,
            reference_time=datetime.now(),
        )

    async def search_nodes(
        self,
        query: str,
        max_nodes: int = 10,
    ) -> list[dict]:
        results = await self.client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
        )
        nodes = results.nodes[:max_nodes] if results.nodes else []
        return [node.model_dump() for node in nodes]

    async def search_memory_facts(
        self,
        query: str,
        max_facts: int = 10,
    ) -> list[dict]:
        relevant_edges = await self.client.search(
            query=query,
            num_results=max_facts,
        )
        facts = [format_fact_result(edge) for edge in relevant_edges]
        return facts
