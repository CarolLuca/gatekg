from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.graphiti import AddBulkEpisodeResults, AddEpisodeResults, AddTripletResults
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchConfig, SearchResults
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.bulk_utils import RawEpisode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from .kg import GraphConnection, format_fact_result

logger = logging.getLogger(__name__)


class MCP_KG(GraphConnection):
    def __init__(self):
        super().__init__()
        self.active_group_id: str | None = None
        self.active_group_ids: list[str] | None = None

    def set_active_group(self, group_id: str | None) -> None:
        self.active_group_id = group_id

    def set_active_groups(self, group_ids: list[str] | None) -> None:
        if group_ids is None:
            self.active_group_ids = None
            return
        normalized_group_ids = [group_id for group_id in group_ids if group_id]
        self.active_group_ids = normalized_group_ids or None

    def clear_active_group(self) -> None:
        self.active_group_id = None
        self.active_group_ids = None

    def _resolve_group_id(self, group_id: str | None) -> str | None:
        if self.active_group_id is not None and group_id not in (None, self.active_group_id):
            raise ValueError("Write is outside the active graph group")
        return group_id or self.active_group_id

    def _normalize_search_group_ids(self, group_ids: list[str] | None) -> list[str] | None:
        if group_ids is None:
            if self.active_group_ids is not None:
                return list(self.active_group_ids)
            return [self.active_group_id] if self.active_group_id else None
        allowed = self.active_group_ids or ([self.active_group_id] if self.active_group_id else [])
        if allowed and (not group_ids or not set(group_ids).issubset(allowed)):
            raise ValueError("Read is outside the active graph groups")
        return group_ids or None

    def _normalize_write_group_ids(self, group_ids: list[str] | None) -> list[str] | None:
        if self.active_group_id is not None:
            if group_ids is not None and group_ids != [self.active_group_id]:
                raise ValueError("Write is outside the active graph group")
            return [self.active_group_id]
        return group_ids or None

    async def _check_episode_ownership(self, uuids: Sequence[str]) -> None:
        if self.active_group_id is not None and uuids:
            episodes = await EpisodicNode.get_by_uuids(self.client.driver, list(uuids))
            if {episode.uuid for episode in episodes} != set(uuids):
                raise ValueError("Unknown episode UUID")
            for episode in episodes:
                self._resolve_group_id(episode.group_id)

    def _coerce_episode_type(self, source: EpisodeType | str | None) -> EpisodeType | None:
        if source is None or isinstance(source, EpisodeType):
            return source

        normalized_source = source.strip().lower()
        try:
            return EpisodeType.from_str(normalized_source)
        except NotImplementedError as exc:
            raise ValueError(f"Unsupported episode source: {source}") from exc

    def _coerce_search_filters(
        self, search_filter: SearchFilters | dict[str, Any] | None
    ) -> SearchFilters | None:
        if search_filter is None or isinstance(search_filter, SearchFilters):
            return search_filter
        return SearchFilters(**search_filter)

    def _coerce_raw_episode(self, episode: RawEpisode | dict[str, Any]) -> RawEpisode:
        if isinstance(episode, RawEpisode):
            return episode

        raw_episode = dict(episode)
        raw_episode["source"] = self._coerce_episode_type(raw_episode["source"])
        return RawEpisode(**raw_episode)

    def _coerce_entity_node(self, node: EntityNode | dict[str, Any]) -> EntityNode:
        if isinstance(node, EntityNode):
            return node
        return EntityNode(**node)

    def _coerce_entity_edge(self, edge: EntityEdge | dict[str, Any]) -> EntityEdge:
        if isinstance(edge, EntityEdge):
            return edge
        return EntityEdge(**edge)

    def _sort_items_by_uuid(self, items: list[Any], uuids: list[str]) -> list[Any]:
        order = {uuid: index for index, uuid in enumerate(uuids)}
        return sorted(items, key=lambda item: order.get(item.uuid, len(order)))

    def _serialize_entity_node(
        self, node: EntityNode, include_embeddings: bool = False
    ) -> dict[str, Any]:
        self._normalize_search_group_ids([node.group_id])
        exclude = {"name_embedding"} if not include_embeddings else None
        result = node.model_dump(mode="json", exclude=exclude)
        attributes = result.get("attributes")
        if isinstance(attributes, dict) and not include_embeddings:
            result["attributes"] = {
                key: value for key, value in attributes.items() if "embedding" not in key.lower()
            }
        return result

    def _serialize_community_node(
        self, node: CommunityNode, include_embeddings: bool = False
    ) -> dict[str, Any]:
        self._normalize_search_group_ids([node.group_id])
        exclude = {"name_embedding"} if not include_embeddings else None
        return node.model_dump(mode="json", exclude=exclude)

    def _serialize_episode(self, episode: EpisodicNode) -> dict[str, Any]:
        self._normalize_search_group_ids([episode.group_id])
        return episode.model_dump(mode="json")

    def _serialize_entity_edge(
        self, edge: EntityEdge, include_embeddings: bool = False
    ) -> dict[str, Any]:
        self._normalize_search_group_ids([edge.group_id])
        if include_embeddings:
            return edge.model_dump(mode="json")
        return format_fact_result(edge)

    def _serialize_episodic_edge(self, edge: EpisodicEdge) -> dict[str, Any]:
        self._normalize_search_group_ids([edge.group_id])
        return edge.model_dump(mode="json")

    def _serialize_community_edge(self, edge: CommunityEdge) -> dict[str, Any]:
        self._normalize_search_group_ids([edge.group_id])
        return edge.model_dump(mode="json")

    def _serialize_add_episode_results(
        self, results: AddEpisodeResults, include_embeddings: bool = False
    ) -> dict[str, Any]:
        return {
            "episode": self._serialize_episode(results.episode),
            "episodic_edges": [
                self._serialize_episodic_edge(edge) for edge in results.episodic_edges
            ],
            "nodes": [
                self._serialize_entity_node(node, include_embeddings=include_embeddings)
                for node in results.nodes
            ],
            "edges": [
                self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
                for edge in results.edges
            ],
            "communities": [
                self._serialize_community_node(node, include_embeddings=include_embeddings)
                for node in results.communities
            ],
            "community_edges": [
                self._serialize_community_edge(edge) for edge in results.community_edges
            ],
        }

    def _serialize_add_bulk_episode_results(
        self, results: AddBulkEpisodeResults, include_embeddings: bool = False
    ) -> dict[str, Any]:
        return {
            "episodes": [self._serialize_episode(episode) for episode in results.episodes],
            "episodic_edges": [
                self._serialize_episodic_edge(edge) for edge in results.episodic_edges
            ],
            "nodes": [
                self._serialize_entity_node(node, include_embeddings=include_embeddings)
                for node in results.nodes
            ],
            "edges": [
                self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
                for edge in results.edges
            ],
            "communities": [
                self._serialize_community_node(node, include_embeddings=include_embeddings)
                for node in results.communities
            ],
            "community_edges": [
                self._serialize_community_edge(edge) for edge in results.community_edges
            ],
        }

    def _serialize_add_triplet_results(
        self, results: AddTripletResults, include_embeddings: bool = False
    ) -> dict[str, Any]:
        return {
            "nodes": [
                self._serialize_entity_node(node, include_embeddings=include_embeddings)
                for node in results.nodes
            ],
            "edges": [
                self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
                for edge in results.edges
            ],
        }

    def _serialize_search_results(
        self, results: SearchResults, include_embeddings: bool = False
    ) -> dict[str, Any]:
        return {
            "edges": [
                self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
                for edge in results.edges
            ],
            "edge_reranker_scores": results.edge_reranker_scores,
            "nodes": [
                self._serialize_entity_node(node, include_embeddings=include_embeddings)
                for node in results.nodes
            ],
            "node_reranker_scores": results.node_reranker_scores,
            "episodes": [self._serialize_episode(episode) for episode in results.episodes],
            "episode_reranker_scores": results.episode_reranker_scores,
            "communities": [
                self._serialize_community_node(node, include_embeddings=include_embeddings)
                for node in results.communities
            ],
            "community_reranker_scores": results.community_reranker_scores,
        }

    async def close(self) -> None:
        await self.client.close()

    async def init_knowledge(
        self,
        supporting_facts: list[str],
        group_id: str | None = None,
    ) -> None:
        normalized_facts = [fact for fact in supporting_facts if fact.strip()]
        if not normalized_facts:
            logger.info(
                "Skipping KG initialization for empty supporting facts. group_id=%s", group_id
            )
            return

        logger.info(
            "Initializing KG knowledge. group_id=%s facts=%d",
            group_id,
            len(normalized_facts),
        )
        reference_time = datetime.now()
        bulk_episodes = [
            RawEpisode(
                name=f"Memory number {index}",
                content=fact,
                source_description="Supporting facts",
                source=EpisodeType.text,
                reference_time=reference_time,
            )
            for index, fact in enumerate(normalized_facts)
        ]
        await self.add_episode_bulk(
            bulk_episodes=bulk_episodes,
            group_id=group_id,
        )

    async def has_group_data(self, group_id: str | None = None) -> bool:
        resolved_group_id = self._resolve_group_id(group_id)
        if resolved_group_id is None:
            logger.info("KG cache lookup skipped because no group_id is active.")
            return False
        episodes = await EpisodicNode.get_by_group_ids(
            self.client.driver,
            group_ids=[resolved_group_id],
            limit=1,
        )
        has_data = bool(episodes)
        logger.info("KG cache lookup for group_id=%s -> %s", resolved_group_id, has_data)
        return has_data

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        await self.client.build_indices_and_constraints(delete_existing=delete_existing)

    async def add_memory(
        self,
        name: str,
        episode_body: str,
        group_id: str | None = None,
        source: EpisodeType | str = EpisodeType.text,
        source_description: str = "",
        uuid: str | None = None,
        reference_time: datetime | None = None,
        update_communities: bool = False,
        previous_episode_uuids: list[str] | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | None = None,
        saga_previous_episode_uuid: str | None = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        resolved_group_id = self._resolve_group_id(group_id)
        if saga is not None and not isinstance(saga, str):
            raise ValueError("Expected a saga name")
        refs = [ref for ref in (uuid, saga_previous_episode_uuid) if ref is not None]
        await self._check_episode_ownership(refs + list(previous_episode_uuids or []))
        results = await self.client.add_episode(
            name=name,
            episode_body=episode_body,
            source_description=source_description,
            source=self._coerce_episode_type(source) or EpisodeType.text,
            reference_time=reference_time or datetime.now(),
            group_id=resolved_group_id,
            uuid=uuid,
            update_communities=update_communities,
            previous_episode_uuids=previous_episode_uuids,
            custom_extraction_instructions=custom_extraction_instructions,
            saga=saga,
            saga_previous_episode_uuid=saga_previous_episode_uuid,
        )
        return self._serialize_add_episode_results(results, include_embeddings=include_embeddings)

    async def add_episode_bulk(
        self,
        bulk_episodes: Sequence[RawEpisode | dict[str, Any]],
        group_id: str | None = None,
        custom_extraction_instructions: str | None = None,
        saga: str | None = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        resolved_group_id = self._resolve_group_id(group_id)
        normalized_episodes = [self._coerce_raw_episode(episode) for episode in bulk_episodes]
        if saga is not None and not isinstance(saga, str):
            raise ValueError("Expected a saga name")
        await self._check_episode_ownership(
            [ep.uuid for ep in normalized_episodes if ep.uuid is not None]
        )
        logger.info(
            "Submitting add_episode_bulk. group_id=%s episodes=%d saga=%s custom_instructions=%s",
            resolved_group_id,
            len(normalized_episodes),
            bool(saga),
            bool(custom_extraction_instructions),
        )
        start = perf_counter()
        results = await self.client.add_episode_bulk(
            bulk_episodes=normalized_episodes,
            group_id=resolved_group_id,
            custom_extraction_instructions=custom_extraction_instructions,
            saga=saga,
        )
        duration_ms = (perf_counter() - start) * 1000
        logger.info(
            "Completed add_episode_bulk. group_id=%s episodes=%d nodes=%d edges=%d episodic_edges=%d duration_ms=%.0f",
            resolved_group_id,
            len(results.episodes),
            len(results.nodes),
            len(results.edges),
            len(results.episodic_edges),
            duration_ms,
        )
        return self._serialize_add_bulk_episode_results(
            results, include_embeddings=include_embeddings
        )

    async def add_triplet(
        self,
        source_node: EntityNode | dict[str, Any],
        edge: EntityEdge | dict[str, Any],
        target_node: EntityNode | dict[str, Any],
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        defaults = {"group_id": self.active_group_id} if self.active_group_id else {}
        source = self._coerce_entity_node(
            {**defaults, **source_node} if isinstance(source_node, dict) else source_node
        )
        target = self._coerce_entity_node(
            {**defaults, **target_node} if isinstance(target_node, dict) else target_node
        )
        if isinstance(edge, dict):
            edge = {
                **defaults,
                "source_node_uuid": source.uuid,
                "target_node_uuid": target.uuid,
                "created_at": datetime.now(UTC),
                **edge,
            }
        relation = self._coerce_entity_edge(edge)
        if (relation.source_node_uuid, relation.target_node_uuid) != (source.uuid, target.uuid):
            raise ValueError("Triplet endpoints must match the supplied nodes")
        if self.active_group_id is not None:
            source = source.model_copy(deep=True)
            target = target.model_copy(deep=True)
            relation = relation.model_copy(deep=True)
            for item in (source, target, relation):
                self._resolve_group_id(item.group_id)
                if item.attributes.keys() & (
                    type(item).model_fields.keys() | {"source_uuid", "target_uuid"}
                ):
                    raise ValueError("Triplet attributes cannot override graph fields")
                try:
                    existing = await type(item).get_by_uuid(self.client.driver, item.uuid)
                except (NodeNotFoundError, EdgeNotFoundError):
                    # Concurrent work groups can propose the same new UUID. Graphiti
                    # merges by UUID, so give new objects a group- and type-local ID.
                    item.uuid = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"gatekg/{self.active_group_id}/{type(item).__name__}/{item.uuid}",
                        )
                    )
                    continue
                self._resolve_group_id(existing.group_id)
            relation.source_node_uuid, relation.target_node_uuid = source.uuid, target.uuid
            await self._check_episode_ownership(relation.episodes)
        results = await self.client.add_triplet(
            source_node=source, edge=relation, target_node=target
        )
        return self._serialize_add_triplet_results(results, include_embeddings=include_embeddings)

    async def build_communities(
        self, group_ids: list[str] | None = None, include_embeddings: bool = False
    ) -> dict[str, Any]:
        if self.active_group_id is not None:
            raise ValueError("Graphiti community rebuild requires unscoped administrative access")
        communities, community_edges = await self.client.build_communities(
            group_ids=self._normalize_write_group_ids(group_ids)
        )
        return {
            "communities": [
                self._serialize_community_node(node, include_embeddings=include_embeddings)
                for node in communities
            ],
            "community_edges": [self._serialize_community_edge(edge) for edge in community_edges],
        }

    async def clear_graph(self, group_ids: list[str] | None = None) -> None:
        resolved_group_ids = self._normalize_write_group_ids(group_ids)
        if not resolved_group_ids:
            raise ValueError("Explicit graph groups are required for tool-driven deletion")
        logger.info("Clearing KG groups=%s", resolved_group_ids)
        await clear_data(self.client.driver, group_ids=resolved_group_ids)

    async def search_nodes(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
        entity_types: list[str] | None = None,
        search_filter: SearchFilters | dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        filters = self._coerce_search_filters(search_filter)
        if entity_types is not None:
            filter_payload = filters.model_dump(exclude_none=True) if filters else {}
            filter_payload["node_labels"] = entity_types
            filters = SearchFilters(**filter_payload)

        results = await self.client.search_(
            query=query,
            config=NODE_HYBRID_SEARCH_RRF,
            group_ids=self._normalize_search_group_ids(group_ids),
            search_filter=filters,
        )
        nodes = results.nodes[:max_nodes] if results.nodes else []
        return [
            self._serialize_entity_node(node, include_embeddings=include_embeddings)
            for node in nodes
        ]

    async def search_memory_facts(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
        search_filter: SearchFilters | dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        relevant_edges = await self.client.search(
            query=query,
            group_ids=self._normalize_search_group_ids(group_ids),
            num_results=max_facts,
            center_node_uuid=center_node_uuid,
            search_filter=self._coerce_search_filters(search_filter),
        )
        return [
            self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
            for edge in relevant_edges
        ]

    async def search_graph(
        self,
        query: str,
        group_ids: list[str] | None = None,
        config: SearchConfig | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | dict[str, Any] | None = None,
        include_embeddings: bool = False,
    ) -> dict[str, Any]:
        results = await self.client.search_(
            query=query,
            config=config or self.default_search_config,
            group_ids=self._normalize_search_group_ids(group_ids),
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            search_filter=self._coerce_search_filters(search_filter),
        )
        return self._serialize_search_results(results, include_embeddings=include_embeddings)

    async def get_nodes_and_edges_by_episode(
        self, episode_uuids: list[str], include_embeddings: bool = False
    ) -> dict[str, Any]:
        results = await self.client.get_nodes_and_edges_by_episode(episode_uuids)
        return self._serialize_search_results(results, include_embeddings=include_embeddings)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: EpisodeType | str | None = None,
        saga: str | None = None,
    ) -> list[dict[str, Any]]:
        episodes = await self.client.retrieve_episodes(
            reference_time=reference_time,
            last_n=last_n,
            group_ids=self._normalize_search_group_ids(group_ids),
            source=self._coerce_episode_type(source),
            saga=saga,
        )
        return [self._serialize_episode(episode) for episode in episodes]

    async def get_entity_edge(self, uuid: str, include_embeddings: bool = False) -> dict[str, Any]:
        entity_edge = await EntityEdge.get_by_uuid(self.client.driver, uuid)
        return self._serialize_entity_edge(entity_edge, include_embeddings=include_embeddings)

    async def get_entity_edges(
        self, uuids: list[str], include_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        entity_edges = await EntityEdge.get_by_uuids(self.client.driver, uuids)
        entity_edges = self._sort_items_by_uuid(entity_edges, uuids)
        return [
            self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
            for edge in entity_edges
        ]

    async def get_entity_edges_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        entity_edges = await EntityEdge.get_by_group_ids(
            self.client.driver,
            group_ids=self._normalize_search_group_ids(group_ids) or [],
            limit=limit,
            uuid_cursor=uuid_cursor,
            with_embeddings=with_embeddings,
        )
        return [
            self._serialize_entity_edge(edge, include_embeddings=with_embeddings)
            for edge in entity_edges
        ]

    async def get_entity_edges_between_nodes(
        self,
        source_node_uuid: str,
        target_node_uuid: str,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        entity_edges = await EntityEdge.get_between_nodes(
            self.client.driver,
            source_node_uuid=source_node_uuid,
            target_node_uuid=target_node_uuid,
        )
        return [
            self._serialize_entity_edge(edge, include_embeddings=include_embeddings)
            for edge in entity_edges
        ]

    async def delete_entity_edge(self, uuid: str) -> None:
        entity_edge = await EntityEdge.get_by_uuid(self.client.driver, uuid)
        self._resolve_group_id(entity_edge.group_id)
        await entity_edge.delete(self.client.driver)

    async def delete_entity_edges(self, uuids: list[str]) -> None:
        entity_edges = await EntityEdge.get_by_uuids(self.client.driver, uuids)
        for entity_edge in entity_edges:
            self._resolve_group_id(entity_edge.group_id)
        for entity_edge in entity_edges:
            await entity_edge.delete(self.client.driver)

    async def get_episode(self, uuid: str) -> dict[str, Any]:
        episode = await EpisodicNode.get_by_uuid(self.client.driver, uuid)
        return self._serialize_episode(episode)

    async def get_episodes(
        self,
        group_ids: list[str] | None = None,
        max_episodes: int = 10,
        uuid_cursor: str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_group_ids = self._normalize_search_group_ids(group_ids)
        if normalized_group_ids is None:
            cursor_filter = "WHERE e.uuid < $uuid_cursor" if uuid_cursor is not None else ""
            async with self.client.driver.session() as session:
                result = await session.run(
                    f"""
                    MATCH (e:Episodic)
                    {cursor_filter}
                    RETURN e.uuid AS uuid
                    ORDER BY e.uuid DESC
                    LIMIT $limit
                    """,
                    limit=max_episodes,
                    uuid_cursor=uuid_cursor,
                )
                episode_uuids = [record["uuid"] async for record in result]
            episodes = await EpisodicNode.get_by_uuids(self.client.driver, episode_uuids)
            episodes = self._sort_items_by_uuid(episodes, episode_uuids)
        else:
            episodes = await EpisodicNode.get_by_group_ids(
                self.client.driver,
                group_ids=normalized_group_ids,
                limit=max_episodes,
                uuid_cursor=uuid_cursor,
            )
        return [self._serialize_episode(episode) for episode in episodes]

    async def get_episodes_by_uuids(self, uuids: list[str]) -> list[dict[str, Any]]:
        episodes = await EpisodicNode.get_by_uuids(self.client.driver, uuids)
        episodes = self._sort_items_by_uuid(episodes, uuids)
        return [self._serialize_episode(episode) for episode in episodes]

    async def delete_episode(self, uuid: str) -> None:
        await self.delete_episodes([uuid])

    async def delete_episodes(self, uuids: list[str]) -> None:
        await self._check_episode_ownership(uuids)
        for uuid in dict.fromkeys(uuids):
            await self.client.remove_episode(uuid)

    async def get_entity_node(self, uuid: str, include_embeddings: bool = False) -> dict[str, Any]:
        node = await EntityNode.get_by_uuid(self.client.driver, uuid)
        if include_embeddings:
            await node.load_name_embedding(self.client.driver)
        return self._serialize_entity_node(node, include_embeddings=include_embeddings)

    async def get_entity_nodes(
        self, uuids: list[str], include_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        nodes = await EntityNode.get_by_uuids(self.client.driver, uuids)
        nodes = self._sort_items_by_uuid(nodes, uuids)
        if include_embeddings:
            for node in nodes:
                await node.load_name_embedding(self.client.driver)
        return [
            self._serialize_entity_node(node, include_embeddings=include_embeddings)
            for node in nodes
        ]

    async def get_entity_nodes_by_group_ids(
        self,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
        with_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        nodes = await EntityNode.get_by_group_ids(
            self.client.driver,
            group_ids=self._normalize_search_group_ids(group_ids) or [],
            limit=limit,
            uuid_cursor=uuid_cursor,
            with_embeddings=with_embeddings,
        )
        return [
            self._serialize_entity_node(node, include_embeddings=with_embeddings) for node in nodes
        ]

    async def get_community_node(
        self, uuid: str, include_embeddings: bool = False
    ) -> dict[str, Any]:
        node = await CommunityNode.get_by_uuid(self.client.driver, uuid)
        if include_embeddings:
            await node.load_name_embedding(self.client.driver)
        return self._serialize_community_node(node, include_embeddings=include_embeddings)

    async def get_communities(
        self,
        uuids: list[str] | None = None,
        group_ids: list[str] | None = None,
        limit: int | None = None,
        uuid_cursor: str | None = None,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_group_ids = self._normalize_search_group_ids(group_ids)
        if uuids is not None:
            communities = await CommunityNode.get_by_uuids(self.client.driver, uuids)
            communities = self._sort_items_by_uuid(communities, uuids)
        elif normalized_group_ids is not None:
            communities = await CommunityNode.get_by_group_ids(
                self.client.driver,
                group_ids=normalized_group_ids,
                limit=limit,
                uuid_cursor=uuid_cursor,
            )
        else:
            cursor_filter = "WHERE c.uuid < $uuid_cursor" if uuid_cursor is not None else ""
            async with self.client.driver.session() as session:
                result = await session.run(
                    f"""
                    MATCH (c:Community)
                    {cursor_filter}
                    RETURN c.uuid AS uuid
                    ORDER BY c.uuid DESC
                    LIMIT $limit
                    """,
                    limit=limit or 10,
                    uuid_cursor=uuid_cursor,
                )
                community_uuids = [record["uuid"] async for record in result]
            communities = await CommunityNode.get_by_uuids(self.client.driver, community_uuids)
            communities = self._sort_items_by_uuid(communities, community_uuids)

        if include_embeddings:
            for community in communities:
                await community.load_name_embedding(self.client.driver)

        return [
            self._serialize_community_node(node, include_embeddings=include_embeddings)
            for node in communities
        ]

    async def get_status(self) -> dict[str, Any]:
        if hasattr(self.graph_driver, "health_check"):
            await self.graph_driver.health_check()
        else:
            async with self.graph_driver.session() as session:
                result = await session.run("RETURN 1 AS ok")
                _ = [record async for record in result]

        return {
            "status": "ok",
            "database_provider": self.graph_driver.provider.value,
            "database": getattr(self.graph_driver, "_database", None),
        }
