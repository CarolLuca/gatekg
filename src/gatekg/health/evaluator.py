from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from math import sqrt
from statistics import mean, pstdev
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    name: str
    raw_value: float
    health_score: float
    details: dict[str, Any]


@dataclass
class KGHealthReport:
    metrics: list[MetricResult]
    average_health_score: float
    raw_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        metric_scores = {metric.name: metric.health_score for metric in self.metrics}
        metric_raw_values = {metric.name: metric.raw_value for metric in self.metrics}
        return {
            "metrics": [asdict(metric) for metric in self.metrics],
            "average_health_score": self.average_health_score,
            "metric_scores": metric_scores,
            "metric_raw_values": metric_raw_values,
            "health_score_list": list(metric_scores.values()),
            "raw_value_list": list(metric_raw_values.values()),
            "raw_summary": self.raw_summary,
        }


class KGHealthEvaluator:
    def __init__(self, kg: Any):
        self.kg = kg

    async def evaluate(
        self,
        *,
        group_ids: Sequence[str],
        shacl_shapes_graph: Any | None = None,
        duplicate_similarity_threshold: float = 0.95,
        semantic_duplicate_similarity_threshold: float = 0.98,
        max_expected_isolate_ratio: float = 0.20,
        max_expected_fragment_ratio: float = 0.20,
    ) -> KGHealthReport:
        normalized_group_ids = _normalize_group_ids(group_ids)
        if not normalized_group_ids:
            raise ValueError("KG health evaluation requires at least one group_id.")

        nodes, edges = await self._load_group_entities(
            normalized_group_ids,
            with_embeddings=True,
        )
        graph = self._build_entity_graph_as_networkx(nodes, edges)
        metric_results: list[MetricResult] = []

        shacl_result = await self._metric_shacl_conformance(
            normalized_group_ids,
            shacl_shapes_graph,
        )
        if shacl_result is not None:
            metric_results.append(shacl_result)

        metric_results.append(self._metric_density(graph))
        metric_results.append(
            self._metric_isolate_ratio(
                graph,
                max_expected_ratio=max_expected_isolate_ratio,
            )
        )
        metric_results.append(
            self._metric_fragmentation(
                graph,
                max_expected_ratio=max_expected_fragment_ratio,
            )
        )
        metric_results.append(self._metric_degree_dispersion(graph))
        metric_results.append(self._metric_name_embedding_coverage(nodes))
        metric_results.append(
            self._metric_semantic_duplicate_ratio(
                nodes,
                similarity_threshold=semantic_duplicate_similarity_threshold,
            )
        )

        duplicate_result = await self._metric_duplicate_ratio(
            group_ids=normalized_group_ids,
            graph=graph,
            similarity_threshold=duplicate_similarity_threshold,
        )
        metric_results.append(duplicate_result)

        average_health_score = (
            mean(metric.health_score for metric in metric_results) if metric_results else 0.0
        )
        metric_scores = {metric.name: metric.health_score for metric in metric_results}
        metric_raw_values = {metric.name: metric.raw_value for metric in metric_results}
        raw_summary = {
            "group_ids": normalized_group_ids,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "metric_scores": metric_scores,
            "metric_raw_values": metric_raw_values,
            "health_score_list": list(metric_scores.values()),
            "raw_value_list": list(metric_raw_values.values()),
            "metrics": {
                metric.name: {
                    "raw_value": metric.raw_value,
                    "health_score": metric.health_score,
                    **metric.details,
                }
                for metric in metric_results
            },
        }
        return KGHealthReport(
            metrics=metric_results,
            average_health_score=average_health_score,
            raw_summary=raw_summary,
        )

    async def _metric_shacl_conformance(
        self,
        group_ids: list[str],
        shapes_graph: Any | None,
    ) -> MetricResult | None:
        if shapes_graph is None:
            return None

        try:
            from pyshacl import validate
        except ImportError as exc:
            raise RuntimeError(
                "pySHACL is required for SHACL validation: pip install pyshacl"
            ) from exc

        data_graph = await self._export_rdf_graph(group_ids)
        conforms, _, results_text = validate(
            data_graph,
            shacl_graph=shapes_graph,
            inference="rdfs",
            abort_on_first=False,
            allow_infos=True,
            allow_warnings=True,
        )
        violation_count = results_text.count("Constraint Violation")
        health_score = 1.0 if conforms else 1.0 / (1.0 + violation_count)
        return MetricResult(
            name="shacl_conformance",
            raw_value=float(violation_count),
            health_score=health_score,
            details={
                "conforms": bool(conforms),
                "violation_count": violation_count,
            },
        )

    def _metric_density(self, graph: nx.DiGraph) -> MetricResult:
        raw_value = float(nx.density(graph))
        health_score = min(raw_value / 0.30, 1.0)
        return MetricResult(
            name="density",
            raw_value=raw_value,
            health_score=health_score,
            details={},
        )

    def _metric_isolate_ratio(
        self,
        graph: nx.DiGraph,
        *,
        max_expected_ratio: float,
    ) -> MetricResult:
        if graph.number_of_nodes() == 0:
            return MetricResult(
                name="isolate_ratio",
                raw_value=1.0,
                health_score=0.0,
                details={"reason": "graph empty"},
            )

        undirected_graph = graph.to_undirected()
        isolates = list(nx.isolates(undirected_graph))
        ratio = len(isolates) / undirected_graph.number_of_nodes()
        health_score = max(0.0, 1.0 - (ratio / max_expected_ratio))
        return MetricResult(
            name="isolate_ratio",
            raw_value=ratio,
            health_score=min(1.0, health_score),
            details={
                "isolated_nodes": len(isolates),
                "total_nodes": undirected_graph.number_of_nodes(),
            },
        )

    def _metric_fragmentation(
        self,
        graph: nx.DiGraph,
        *,
        max_expected_ratio: float,
    ) -> MetricResult:
        if graph.number_of_nodes() == 0:
            return MetricResult(
                name="fragmentation",
                raw_value=1.0,
                health_score=0.0,
                details={"reason": "graph empty"},
            )

        if graph.is_directed():
            component_count = nx.number_weakly_connected_components(graph)
        else:
            component_count = nx.number_connected_components(graph)
        fragment_ratio = (component_count - 1) / graph.number_of_nodes()
        health_score = max(0.0, 1.0 - (fragment_ratio / max_expected_ratio))
        return MetricResult(
            name="fragmentation",
            raw_value=fragment_ratio,
            health_score=min(1.0, health_score),
            details={
                "component_count": component_count,
                "total_nodes": graph.number_of_nodes(),
            },
        )

    def _metric_degree_dispersion(self, graph: nx.DiGraph) -> MetricResult:
        if graph.number_of_nodes() <= 1:
            return MetricResult(
                name="degree_dispersion",
                raw_value=0.0,
                health_score=0.0,
                details={"reason": "graph too small"},
            )

        values = list(nx.degree_centrality(graph).values())
        dispersion = float(pstdev(values))
        health_score = min(dispersion / 0.15, 1.0)
        return MetricResult(
            name="degree_dispersion",
            raw_value=dispersion,
            health_score=health_score,
            details={"mean_degree_centrality": sum(values) / len(values)},
        )

    def _metric_name_embedding_coverage(
        self,
        nodes: Sequence[dict[str, Any]],
    ) -> MetricResult:
        entity_count = len(nodes)
        if entity_count == 0:
            return MetricResult(
                name="name_embedding_coverage",
                raw_value=0.0,
                health_score=0.0,
                details={"reason": "graph empty"},
            )

        embedded_entity_count = sum(
            1 for node in nodes if _normalize_embedding(node.get("name_embedding")) is not None
        )
        coverage = embedded_entity_count / entity_count
        return MetricResult(
            name="name_embedding_coverage",
            raw_value=coverage,
            health_score=coverage,
            details={
                "embedded_entity_count": embedded_entity_count,
                "entity_count": entity_count,
            },
        )

    def _metric_semantic_duplicate_ratio(
        self,
        nodes: Sequence[dict[str, Any]],
        *,
        similarity_threshold: float,
    ) -> MetricResult:
        entity_count = len(nodes)
        if entity_count <= 1:
            return MetricResult(
                name="semantic_duplicate_ratio",
                raw_value=0.0,
                health_score=1.0,
                details={
                    "duplicate_pairs": 0,
                    "entity_count": entity_count,
                    "embedded_entity_count": entity_count,
                    "similarity_threshold": similarity_threshold,
                    "reason": "graph too small",
                },
            )

        embedded_nodes: list[dict[str, Any]] = []
        for node in nodes:
            normalized_embedding = _normalize_embedding(node.get("name_embedding"))
            if normalized_embedding is None:
                continue
            embedded_nodes.append(
                {
                    "uuid": node.get("uuid"),
                    "name": node.get("name"),
                    "embedding": normalized_embedding,
                }
            )

        embedded_entity_count = len(embedded_nodes)
        coverage = embedded_entity_count / entity_count if entity_count else 0.0
        if embedded_entity_count < 2:
            return MetricResult(
                name="semantic_duplicate_ratio",
                raw_value=0.0,
                health_score=coverage,
                details={
                    "duplicate_pairs": 0,
                    "entity_count": entity_count,
                    "embedded_entity_count": embedded_entity_count,
                    "similarity_threshold": similarity_threshold,
                    "reason": "insufficient_name_embeddings",
                },
            )

        duplicate_pairs = 0
        compared_pairs = 0
        duplicate_candidates: list[dict[str, Any]] = []
        for index, source in enumerate(embedded_nodes):
            for target in embedded_nodes[index + 1 :]:
                similarity = _cosine_similarity(
                    source["embedding"],
                    target["embedding"],
                )
                if similarity is None:
                    continue
                compared_pairs += 1
                if similarity < similarity_threshold:
                    continue
                duplicate_pairs += 1
                duplicate_candidates.append(
                    {
                        "source_uuid": source["uuid"],
                        "source_name": source["name"],
                        "target_uuid": target["uuid"],
                        "target_name": target["name"],
                        "similarity": round(similarity, 6),
                    }
                )

        if compared_pairs == 0:
            return MetricResult(
                name="semantic_duplicate_ratio",
                raw_value=0.0,
                health_score=coverage,
                details={
                    "duplicate_pairs": 0,
                    "entity_count": entity_count,
                    "embedded_entity_count": embedded_entity_count,
                    "compared_pairs": 0,
                    "similarity_threshold": similarity_threshold,
                    "reason": "no_comparable_name_embeddings",
                },
            )

        ratio = duplicate_pairs / max(1, entity_count)
        health_score = max(0.0, 1.0 - min(ratio / 0.10, 1.0))
        duplicate_candidates.sort(key=lambda candidate: candidate["similarity"], reverse=True)
        return MetricResult(
            name="semantic_duplicate_ratio",
            raw_value=ratio,
            health_score=health_score,
            details={
                "duplicate_pairs": duplicate_pairs,
                "entity_count": entity_count,
                "embedded_entity_count": embedded_entity_count,
                "compared_pairs": compared_pairs,
                "similarity_threshold": similarity_threshold,
                "duplicate_candidates": duplicate_candidates[:10],
            },
        )

    async def _metric_duplicate_ratio(
        self,
        *,
        group_ids: list[str],
        graph: nx.DiGraph,
        similarity_threshold: float,
    ) -> MetricResult:
        gds_result = await self._metric_duplicate_ratio_gds(
            group_ids=group_ids,
            entity_count=graph.number_of_nodes(),
            similarity_threshold=similarity_threshold,
        )
        if gds_result is not None:
            return gds_result
        return self._metric_duplicate_ratio_name_fallback(
            graph=graph,
            similarity_threshold=similarity_threshold,
        )

    async def _metric_duplicate_ratio_gds(
        self,
        *,
        group_ids: list[str],
        entity_count: int,
        similarity_threshold: float,
    ) -> MetricResult | None:
        if entity_count <= 1:
            return MetricResult(
                name="duplicate_ratio",
                raw_value=0.0,
                health_score=1.0,
                details={
                    "duplicate_pairs": 0,
                    "entity_count": entity_count,
                    "similarity_threshold": similarity_threshold,
                    "reason": "graph too small",
                },
            )

        driver = self._get_async_driver()
        if driver is None:
            return None

        if not await self._supports_gds_duplicate_ratio():
            return None

        projection_name = f"kg_health_projection_{uuid.uuid4().hex[:12]}"
        node_projection = """
        MATCH (n:Entity)
        WHERE n.group_id IN $group_ids
        RETURN id(n) AS id
        """
        relationship_projection = """
        MATCH (a:Entity)-[r]->(b:Entity)
        WHERE a.group_id IN $group_ids
          AND b.group_id IN $group_ids
          AND r.group_id IN $group_ids
        RETURN id(a) AS source, id(b) AS target
        """

        duplicate_pairs = 0
        try:
            async with driver.session() as session:
                try:
                    drop_result = await session.run(
                        "CALL gds.graph.drop($name, false) YIELD graphName",
                        name=projection_name,
                    )
                    await drop_result.consume()
                except Exception:
                    pass

                project_result = await session.run(
                    """
                    CALL gds.graph.project.cypher(
                        $name,
                        $node_projection,
                        $relationship_projection,
                        {parameters: {group_ids: $group_ids}}
                    )
                    YIELD graphName
                    RETURN graphName
                    """,
                    name=projection_name,
                    node_projection=node_projection,
                    relationship_projection=relationship_projection,
                    group_ids=group_ids,
                )
                await project_result.consume()

                similarity_result = await session.run(
                    """
                    CALL gds.nodeSimilarity.stream($name)
                    YIELD node1, node2, similarity
                    WHERE similarity >= $threshold
                    RETURN count(*) AS duplicate_pairs
                    """,
                    name=projection_name,
                    threshold=similarity_threshold,
                )
                record = await similarity_result.single()
                duplicate_pairs = int(record["duplicate_pairs"]) if record else 0
        except Exception as exc:
            logger.warning(
                "Falling back to normalized-name duplicate_ratio because GDS node similarity failed: %s",
                exc,
            )
            return None
        finally:
            try:
                async with driver.session() as session:
                    drop_result = await session.run(
                        "CALL gds.graph.drop($name, false) YIELD graphName",
                        name=projection_name,
                    )
                    await drop_result.consume()
            except Exception:
                pass

        ratio = duplicate_pairs / max(1, entity_count)
        health_score = max(0.0, 1.0 - min(ratio / 0.10, 1.0))
        return MetricResult(
            name="duplicate_ratio",
            raw_value=ratio,
            health_score=health_score,
            details={
                "duplicate_pairs": duplicate_pairs,
                "entity_count": entity_count,
                "similarity_threshold": similarity_threshold,
            },
        )

    def _metric_duplicate_ratio_name_fallback(
        self,
        *,
        graph: nx.DiGraph,
        similarity_threshold: float,
    ) -> MetricResult:
        normalized_name_counts: Counter[str] = Counter()
        for _, attributes in graph.nodes(data=True):
            normalized_name = _normalize_entity_name(attributes.get("name"))
            if normalized_name:
                normalized_name_counts[normalized_name] += 1

        duplicate_name_groups = {
            normalized_name: count
            for normalized_name, count in normalized_name_counts.items()
            if count > 1
        }
        duplicate_pairs = sum(_n_choose_2(count) for count in duplicate_name_groups.values())
        entity_count = graph.number_of_nodes()
        ratio = duplicate_pairs / max(1, entity_count)
        health_score = max(0.0, 1.0 - min(ratio / 0.10, 1.0))

        return MetricResult(
            name="duplicate_ratio",
            raw_value=ratio,
            health_score=health_score,
            details={
                "duplicate_pairs": duplicate_pairs,
                "entity_count": entity_count,
                "similarity_threshold": similarity_threshold,
                "method": "normalized_name_exact",
                "threshold_applied": False,
                "duplicate_name_group_count": len(duplicate_name_groups),
                "duplicate_name_groups": [
                    {"normalized_name": normalized_name, "count": count}
                    for normalized_name, count in sorted(duplicate_name_groups.items())
                ],
            },
        )

    async def _load_entity_graph_as_networkx(
        self,
        group_ids: list[str],
        *,
        with_embeddings: bool = False,
    ) -> nx.DiGraph:
        nodes, edges = await self._load_group_entities(
            group_ids,
            with_embeddings=with_embeddings,
        )
        return self._build_entity_graph_as_networkx(nodes, edges)

    def _build_entity_graph_as_networkx(
        self,
        nodes: Sequence[dict[str, Any]],
        edges: Sequence[dict[str, Any]],
    ) -> nx.DiGraph:
        graph = nx.DiGraph()

        for node in nodes:
            graph.add_node(
                node["uuid"],
                name=node.get("name"),
                group_id=node.get("group_id"),
            )

        for edge in edges:
            source_node_uuid = edge["source_node_uuid"]
            target_node_uuid = edge["target_node_uuid"]
            graph.add_node(source_node_uuid)
            graph.add_node(target_node_uuid)
            graph.add_edge(
                source_node_uuid,
                target_node_uuid,
                uuid=edge.get("uuid"),
                group_id=edge.get("group_id"),
                name=edge.get("name"),
                fact=edge.get("fact"),
            )

        return graph

    async def _export_rdf_graph(self, group_ids: list[str]):
        try:
            from rdflib import Graph, Namespace, URIRef
        except ImportError as exc:
            raise RuntimeError("rdflib is required for RDF export") from exc

        _, edges = await self._load_group_entities(group_ids)
        graph_namespace = Namespace("http://example.org/kg/")
        graph = Graph()
        for edge in edges:
            source = URIRef(graph_namespace[str(edge["source_node_uuid"])])
            predicate = URIRef(graph_namespace[str(edge.get("name") or "related_to")])
            target = URIRef(graph_namespace[str(edge["target_node_uuid"])])
            graph.add((source, predicate, target))
        return graph

    async def _load_group_entities(
        self,
        group_ids: list[str],
        *,
        with_embeddings: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if hasattr(self.kg, "get_entity_nodes_by_group_ids") and hasattr(
            self.kg, "get_entity_edges_by_group_ids"
        ):
            if with_embeddings:
                try:
                    nodes = await self.kg.get_entity_nodes_by_group_ids(
                        group_ids,
                        with_embeddings=True,
                    )
                    edges = await self.kg.get_entity_edges_by_group_ids(
                        group_ids,
                        with_embeddings=True,
                    )
                    return list(nodes), list(edges)
                except TypeError:
                    try:
                        nodes = await self.kg.get_entity_nodes_by_group_ids(
                            group_ids,
                            include_embeddings=True,
                        )
                        edges = await self.kg.get_entity_edges_by_group_ids(
                            group_ids,
                            include_embeddings=True,
                        )
                        return list(nodes), list(edges)
                    except TypeError:
                        if self._get_async_driver() is not None:
                            return await self._query_group_entities(
                                group_ids,
                                with_embeddings=True,
                            )

            nodes = await self.kg.get_entity_nodes_by_group_ids(group_ids)
            edges = await self.kg.get_entity_edges_by_group_ids(group_ids)
            return list(nodes), list(edges)
        return await self._query_group_entities(
            group_ids,
            with_embeddings=with_embeddings,
        )

    async def _query_group_entities(
        self,
        group_ids: list[str],
        *,
        with_embeddings: bool = False,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        driver = self._get_async_driver()
        if driver is None:
            raise RuntimeError("Could not obtain an async graph driver for health evaluation.")

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        node_return_fields = [
            "n.uuid AS uuid",
            "n.group_id AS group_id",
            "n.name AS name",
        ]
        edge_return_fields = [
            "r.uuid AS uuid",
            "r.group_id AS group_id",
            "r.name AS name",
            "r.fact AS fact",
            "r.source_node_uuid AS source_node_uuid",
            "r.target_node_uuid AS target_node_uuid",
        ]
        if with_embeddings:
            node_return_fields.append("n.name_embedding AS name_embedding")
            edge_return_fields.append("r.fact_embedding AS fact_embedding")

        async with driver.session() as session:
            node_result = await session.run(
                f"""
                MATCH (n:Entity)
                WHERE n.group_id IN $group_ids
                RETURN
                    {", ".join(node_return_fields)}
                """,
                group_ids=group_ids,
            )
            nodes = [record async for record in node_result]

            edge_result = await session.run(
                f"""
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.group_id IN $group_ids
                  AND b.group_id IN $group_ids
                  AND r.group_id IN $group_ids
                RETURN
                    {", ".join(edge_return_fields)}
                """,
                group_ids=group_ids,
            )
            edges = [record async for record in edge_result]

        return nodes, edges

    def _get_async_driver(self) -> Any | None:
        candidates = [
            getattr(getattr(self.kg, "client", None), "driver", None),
            getattr(self.kg, "graph_driver", None),
            getattr(self.kg, "driver", None),
        ]
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "session"):
                return candidate
        return None

    async def _supports_gds_duplicate_ratio(self) -> bool:
        driver = self._get_async_driver()
        if driver is None:
            return False

        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    SHOW PROCEDURES YIELD name
                    WHERE name IN [
                        'gds.graph.project.cypher',
                        'gds.nodeSimilarity.stream'
                    ]
                    RETURN collect(name) AS names
                    """
                )
                record = await result.single()
        except Exception:
            return False

        available_names = set(record["names"] if record else [])
        return {
            "gds.graph.project.cypher",
            "gds.nodeSimilarity.stream",
        }.issubset(available_names)


def _normalize_group_ids(group_ids: Sequence[str]) -> list[str]:
    return [group_id for group_id in group_ids if str(group_id).strip()]


def _normalize_entity_name(name: Any) -> str:
    if name is None:
        return ""
    text = str(name).strip().lower()
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _n_choose_2(value: int) -> int:
    return (value * (value - 1)) // 2


def _normalize_embedding(value: Any) -> tuple[float, ...] | None:
    if value is None or isinstance(value, (str, bytes)):
        return None

    try:
        values = tuple(float(item) for item in value)
    except TypeError:
        return None
    except ValueError:
        return None

    if not values:
        return None

    magnitude = sqrt(sum(component * component for component in values))
    if magnitude <= 0.0:
        return None

    return tuple(component / magnitude for component in values)


def _cosine_similarity(
    source_embedding: Sequence[float],
    target_embedding: Sequence[float],
) -> float | None:
    if len(source_embedding) != len(target_embedding):
        return None
    return sum(
        source_component * target_component
        for source_component, target_component in zip(
            source_embedding, target_embedding, strict=True
        )
    )


async def evaluate_kg_health(
    kg: Any,
    *,
    group_ids: Sequence[str],
    shacl_shapes_graph: Any | None = None,
    duplicate_similarity_threshold: float = 0.95,
    semantic_duplicate_similarity_threshold: float = 0.98,
) -> dict[str, Any]:
    evaluator = KGHealthEvaluator(kg)
    report = await evaluator.evaluate(
        group_ids=group_ids,
        shacl_shapes_graph=shacl_shapes_graph,
        duplicate_similarity_threshold=duplicate_similarity_threshold,
        semantic_duplicate_similarity_threshold=semantic_duplicate_similarity_threshold,
    )
    return report.to_dict()
