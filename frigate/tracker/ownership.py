"""Single camera ownership decision shared by process launch paths."""

from __future__ import annotations

from frigate.config.tracker import TrackerConfig


def should_start_local_camera(
    tracker: TrackerConfig, camera: str, *, edge_node_id: str | None = None
) -> bool:
    """Resolve one owner without changing the unassigned embedded path."""
    owner = tracker.owner_for(camera)
    if edge_node_id is None:
        return owner is None
    return owner == edge_node_id
