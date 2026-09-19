import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodicNode

from gatekg.api.mcp_kg import MCP_KG
from gatekg.api.mcp_rlm import READ_WRITE_KG_TOOL_NAMES


@pytest.fixture
def kg():
    client = MCP_KG.__new__(MCP_KG)
    client.set_active_group("work")
    client.set_active_groups(["work", "base"])
    client.client = AsyncMock()
    return client


def test_group_override_and_empty_list_rejected(kg):
    with pytest.raises(ValueError, match="active graph group"):
        asyncio.run(kg.add_memory("n", "body", "base"))
    for groups in ([], ["base"], ["work", "base"]):
        with pytest.raises(ValueError, match="active graph group"):
            asyncio.run(kg.clear_graph(groups))
    assert not {"initialize", "build_communities", "build_indices_and_constraints"} & set(
        READ_WRITE_KG_TOOL_NAMES
    )
    kg.client.add_episode.assert_not_awaited()


def test_batch_checks_all_owners_before_any_delete(kg, monkeypatch):
    owned = SimpleNamespace(uuid="a", group_id="work", delete=AsyncMock())
    foreign = SimpleNamespace(uuid="b", group_id="base", delete=AsyncMock())
    for cls, method in [(EntityEdge, kg.delete_entity_edges), (EpisodicNode, kg.delete_episodes)]:
        monkeypatch.setattr(cls, "get_by_uuids", AsyncMock(return_value=[owned, foreign]))
        with pytest.raises(ValueError, match="active graph group"):
            asyncio.run(method(["a", "b"]))
    owned.delete.assert_not_awaited()
    kg.client.remove_episode.assert_not_awaited()


def test_triplet_checks_stored_ownership_and_reserved_attributes(kg, monkeypatch):
    node = EntityNode(name="memory", group_id="work")
    edge = EntityEdge(
        created_at=datetime.now(UTC),
        name="link",
        fact="fact",
        group_id="work",
        source_node_uuid=node.uuid,
        target_node_uuid=node.uuid,
    )
    monkeypatch.setattr(
        EntityNode, "get_by_uuid", AsyncMock(return_value=SimpleNamespace(group_id="base"))
    )
    with pytest.raises(ValueError, match="active graph group"):
        asyncio.run(kg.add_triplet(node, edge, node))
    node.attributes["group_id"] = "base"
    with pytest.raises(ValueError, match="override"):
        asyncio.run(kg.add_triplet(node, edge, node))
    kg.client.add_triplet.assert_not_awaited()


def test_local_triplet_fills_group_and_scopes_new_ids(kg, monkeypatch):
    from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError

    monkeypatch.setattr(EntityNode, "get_by_uuid", AsyncMock(side_effect=NodeNotFoundError("new")))
    monkeypatch.setattr(EntityEdge, "get_by_uuid", AsyncMock(side_effect=EdgeNotFoundError("new")))
    kg.client.add_triplet.return_value = SimpleNamespace(nodes=[], edges=[])
    assigned = []
    for group in ("work", "another-work"):
        kg.set_active_group(group)
        asyncio.run(
            kg.add_triplet(
                {"uuid": "client-id", "name": "a"}, {"name": "link", "fact": "text"}, {"name": "b"}
            )
        )
        call = kg.client.add_triplet.await_args.kwargs
        assert {item.group_id for item in call.values()} == {group}
        assert call["edge"].source_node_uuid == call["source_node"].uuid
        assert call["edge"].target_node_uuid == call["target_node"].uuid
        assigned.append(call["source_node"].uuid)
    assert assigned[0] != assigned[1]
