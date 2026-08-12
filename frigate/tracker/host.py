"""Fail-closed Frigate-side ordered tracker ingest adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .contracts import TrackerOperation, TrackerUpdate


class TrackerIngestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IngestAck:
    node_id: str
    node_epoch: str
    journal_sequence: int
    event_id: str
    duplicate: bool = False


class TrackerHostIngest:
    """Validate lineage before the canonical Event/SQLite transaction callback.

    The callback must atomically accept Event metadata and every media manifest. An
    ACK is returned only after it completes successfully.
    """

    def __init__(
        self,
        camera_owners: dict[str, str],
        commit: Callable[[TrackerUpdate], None],
    ) -> None:
        self._camera_owners = dict(camera_owners)
        self._commit = commit
        self._node_epochs: dict[str, str] = {}
        self._stream_epochs: dict[tuple[str, str], str] = {}
        self._last_sequences: dict[tuple[str, str], int] = {}
        self._accepted: dict[tuple[str, str, int], str] = {}
        self._active: dict[tuple[str, str, str, str], str] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def seed_sequence(self, node_id: str, node_epoch: str, sequence: int) -> None:
        """Restore the durable acceptance cursor before replay after main restart."""
        if sequence < 0:
            raise ValueError("sequence baseline must be non-negative")
        key = (node_id, node_epoch)
        self._last_sequences[key] = max(self._last_sequences.get(key, 0), sequence)
        self._node_epochs[node_id] = node_epoch

    def accept(self, update: TrackerUpdate) -> IngestAck:
        owner = self._camera_owners.get(update.camera_id)
        if owner != update.node_id:
            raise TrackerIngestError("camera_owner_mismatch")

        accepted_key = (
            update.node_id,
            update.node_epoch,
            update.journal_sequence,
        )
        accepted_event = self._accepted.get(accepted_key)
        if accepted_event is not None:
            if accepted_event != update.event_id:
                raise TrackerIngestError("sequence_event_conflict")
            return IngestAck(
                update.node_id,
                update.node_epoch,
                update.journal_sequence,
                update.event_id,
                True,
            )

        current_node_epoch = self._node_epochs.get(update.node_id)
        if current_node_epoch is not None and current_node_epoch != update.node_epoch:
            if any(key[0] == update.node_id for key in self._active):
                raise TrackerIngestError("node_epoch_changed_with_active_tracks")
            self._last_sequences.pop((update.node_id, current_node_epoch), None)
        self._node_epochs[update.node_id] = update.node_epoch

        stream_key = (update.node_id, update.camera_id)
        current_stream_epoch = self._stream_epochs.get(stream_key)
        if (
            current_stream_epoch is not None
            and current_stream_epoch != update.stream_epoch
            and any(
                key[0] == update.node_id and key[1] == update.camera_id
                for key in self._active
            )
        ):
            raise TrackerIngestError("stream_epoch_changed_with_active_tracks")
        self._stream_epochs[stream_key] = update.stream_epoch

        sequence_key = (update.node_id, update.node_epoch)
        expected = self._last_sequences.get(sequence_key, 0) + 1
        if update.journal_sequence != expected:
            raise TrackerIngestError(
                "journal_sequence_gap" if update.journal_sequence > expected else "stale_sequence"
            )

        track_key = (
            update.node_id,
            update.camera_id,
            update.stream_epoch,
            update.track_id,
        )
        active_event = self._active.get(track_key)
        if update.operation is TrackerOperation.START:
            if active_event is not None:
                raise TrackerIngestError("duplicate_track_start")
        elif active_event is None:
            raise TrackerIngestError("update_without_active_start")
        elif active_event != update.event_id:
            raise TrackerIngestError("producer_event_id_changed")

        for manifest in update.media:
            if manifest.event_id != update.event_id:
                raise TrackerIngestError("media_event_mismatch")

        self._commit(update)

        if update.operation is TrackerOperation.START:
            self._active[track_key] = update.event_id
        elif update.operation is TrackerOperation.END:
            self._active.pop(track_key, None)
        self._last_sequences[sequence_key] = update.journal_sequence
        self._accepted[accepted_key] = update.event_id
        return IngestAck(
            update.node_id,
            update.node_epoch,
            update.journal_sequence,
            update.event_id,
        )
