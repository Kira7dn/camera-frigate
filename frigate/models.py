from peewee import (
    BlobField,
    BooleanField,
    CharField,
    CompositeKey,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)
from playhouse.sqlite_ext import JSONField


class Event(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    label = CharField(index=True, max_length=20)
    sub_label = CharField(max_length=100, null=True)
    camera = CharField(index=True, max_length=20)
    start_time = DateTimeField()
    end_time = DateTimeField()
    top_score = (
        FloatField()
    )  # TODO remove when columns can be dropped without rebuilding table
    score = (
        FloatField()
    )  # TODO remove when columns can be dropped without rebuilding table
    false_positive = BooleanField()
    zones = JSONField()
    thumbnail = TextField()
    has_clip = BooleanField(default=True)
    has_snapshot = BooleanField(default=True)
    region = (
        JSONField()
    )  # TODO remove when columns can be dropped without rebuilding table
    box = (
        JSONField()
    )  # TODO remove when columns can be dropped without rebuilding table
    area = (
        IntegerField()
    )  # TODO remove when columns can be dropped without rebuilding table
    retain_indefinitely = BooleanField(default=False)
    ratio = FloatField(
        default=1.0
    )  # TODO remove when columns can be dropped without rebuilding table
    plus_id = CharField(max_length=30)
    model_hash = CharField(max_length=32)
    detector_type = CharField(max_length=32)
    model_type = CharField(max_length=32)
    data = JSONField()  # ex: tracked object box, region, etc.
    # Canonical event projection. EventAggregator is the sole writer of these
    # fields; legacy tracking fields above remain readable during rollout.
    state = CharField(max_length=20, default="TRACKING", index=True)
    revision = IntegerField(default=0)
    finalized_at = DateTimeField(null=True)
    canonical_plate = CharField(max_length=32, null=True)
    canonical_plate_score = FloatField(null=True)
    canonical_sub_label = CharField(max_length=100, null=True)
    display_label = CharField(max_length=100, null=True)
    canonical_evidence_id = CharField(max_length=64, null=True)
    canonical_artifact_id = CharField(max_length=64, null=True)


class EventObservation(Model):
    """Durable, idempotent input to the canonical event projection."""

    observation_id = CharField(primary_key=True, max_length=64)
    event_id = CharField(index=True, max_length=30)
    kind = CharField(index=True, max_length=32)
    observed_at = DateTimeField(index=True)
    frame_time = FloatField(null=True)
    evidence_id = CharField(max_length=64, null=True)
    payload = JSONField()
    expires_at = DateTimeField(index=True)

    class Meta:
        table_name = "event_observation"


class EventEvidence(Model):
    """One immutable full frame and every box measured on that frame."""

    id = CharField(primary_key=True, max_length=64)
    event_id = CharField(index=True, max_length=30)
    frame_ref = TextField()
    frame_time = FloatField()
    width = IntegerField()
    height = IntegerField()
    boxes = JSONField()
    technical = JSONField()
    created_at = DateTimeField(index=True)

    class Meta:
        table_name = "event_evidence"


class MediaArtifact(Model):
    """Immutable canonical media manifest; image bytes live outside SQLite."""

    id = CharField(primary_key=True, max_length=64)
    event_id = CharField(index=True, max_length=30)
    revision = IntegerField()
    evidence_id = CharField(index=True, max_length=64)
    profile = CharField(max_length=32)
    render_version = IntegerField()
    path = TextField(unique=True)
    sha256 = CharField(max_length=64)
    byte_size = IntegerField()
    manifest = JSONField()
    created_at = DateTimeField(index=True)
    expires_at = DateTimeField(index=True)
    pinned = BooleanField(default=False, index=True)

    class Meta:
        table_name = "media_artifact"
        indexes = (
            (("event_id", "revision", "evidence_id", "profile", "render_version"), True),
        )


class NotificationIntent(Model):
    """Immutable presentation contract, coalesced per event recipient/channel."""

    id = CharField(primary_key=True, max_length=64)
    event_id = CharField(index=True, max_length=30)
    revision = IntegerField()
    channel = CharField(max_length=20)
    recipient_id = CharField(max_length=128)
    facts = JSONField()
    caption = TextField()
    actions = JSONField()
    media_artifact_id = CharField(max_length=64, null=True, index=True)
    status = CharField(max_length=20, default="pending", index=True)
    created_at = DateTimeField(index=True)

    class Meta:
        table_name = "notification_intent"
        indexes = ((("event_id", "revision", "channel", "recipient_id"), True),)


class Timeline(Model):
    timestamp = DateTimeField()
    camera = CharField(index=True, max_length=20)
    source = CharField(index=True, max_length=20)  # ex: tracked object, audio, external
    source_id = CharField(index=True, max_length=30)
    class_type = CharField(max_length=50)  # ex: entered_zone, audio_heard
    data = JSONField()  # ex: tracked object id, region, box, etc.


class Regions(Model):
    camera = CharField(null=False, primary_key=True, max_length=20)
    grid = JSONField()  # json blob of grid
    last_update = DateTimeField()


class Recordings(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    camera = CharField(index=True, max_length=20)
    path = CharField(unique=True)
    start_time = DateTimeField()
    end_time = DateTimeField()
    duration = FloatField()
    motion = IntegerField(null=True)
    objects = IntegerField(null=True)
    dBFS = IntegerField(null=True)
    segment_size = FloatField(default=0)  # this should be stored as MB
    regions = IntegerField(null=True)
    motion_heatmap = JSONField(null=True)  # 16x16 grid, 256 values (0-255)


class ExportCase(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    name = CharField(index=True, max_length=100)
    description = TextField(null=True)
    created_at = DateTimeField()
    updated_at = DateTimeField()


class Export(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    camera = CharField(index=True, max_length=20)
    name = CharField(index=True, max_length=100)
    date = DateTimeField()
    video_path = CharField(unique=True)
    thumb_path = CharField(unique=True)
    in_progress = BooleanField()
    export_case = ForeignKeyField(
        ExportCase,
        null=True,
        backref="exports",
        column_name="export_case_id",
    )


class ReviewSegment(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    camera = CharField(index=True, max_length=20)
    start_time = DateTimeField()
    end_time = DateTimeField()
    severity = CharField(max_length=30)  # alert, detection
    thumb_path = CharField(unique=True)
    data = JSONField()  # additional data about detection like list of labels, zone, areas of significant motion


class UserReviewStatus(Model):
    user_id = CharField(max_length=30)
    review_segment = ForeignKeyField(ReviewSegment, backref="user_reviews")
    has_been_reviewed = BooleanField(default=False)

    class Meta:
        indexes = ((("user_id", "review_segment"), True),)


class Previews(Model):
    id = CharField(null=False, primary_key=True, max_length=30)
    camera = CharField(index=True, max_length=20)
    path = CharField(unique=True)
    start_time = DateTimeField()
    end_time = DateTimeField()
    duration = FloatField()


# Used for temporary table in record/cleanup.py
class RecordingsToDelete(Model):
    id = CharField(null=False, primary_key=False, max_length=30)

    class Meta:
        temporary = True


class User(Model):
    username = CharField(null=False, primary_key=True, max_length=30)
    role = CharField(
        max_length=20,
        default="admin",
    )
    password_hash = CharField(null=False, max_length=120)
    password_changed_at = DateTimeField(null=True)
    notification_tokens = JSONField()

    @classmethod
    def get_allowed_cameras(
        cls, role: str, roles_dict: dict[str, list[str]], all_camera_names: set[str]
    ) -> list[str]:
        if role not in roles_dict:
            return []  # Invalid role grants no access
        allowed = roles_dict[role]
        if not allowed:  # Empty list means all cameras
            return list(all_camera_names)

        return [cam for cam in allowed if cam in all_camera_names]


class Trigger(Model):
    camera = CharField(max_length=20)
    name = CharField()
    type = CharField(max_length=10)
    data = TextField()
    threshold = FloatField()
    model = CharField(max_length=30)
    embedding = BlobField()
    triggering_event_id = CharField(max_length=30)
    last_triggered = DateTimeField()

    class Meta:
        primary_key = CompositeKey("camera", "name")


class NotificationDelivery(Model):
    """Durable social notification outbox entry."""

    id = CharField(null=False, primary_key=True, max_length=36)
    provider = CharField(null=False, max_length=20, index=True)
    recipient_id = CharField(null=False, max_length=64)
    rule_id = CharField(null=False, max_length=64, default="legacy")
    source_type = CharField(null=False, max_length=30)
    source_id = CharField(null=False, max_length=64)
    payload = JSONField()
    status = CharField(null=False, max_length=20, default="pending", index=True)
    attempts = IntegerField(null=False, default=0)
    next_attempt = DateTimeField(null=False, index=True)
    created_at = DateTimeField(null=False)
    updated_at = DateTimeField(null=False)
    completed_at = DateTimeField(null=True)
    last_error = TextField(null=True)
    intent_id = CharField(max_length=64, null=True, index=True)
    media_artifact_id = CharField(max_length=64, null=True, index=True)

    class Meta:
        table_name = "notification_delivery"
        indexes = (
            (("provider", "recipient_id", "rule_id", "source_type", "source_id"), True),
        )


class NotificationRuleState(Model):
    """Durable per-destination cooldown and latest source receipt."""

    rule_id = CharField(null=False, max_length=64)
    camera = CharField(null=False, max_length=64)
    channel = CharField(null=False, max_length=20)
    recipient_id = CharField(null=False, max_length=128)
    last_source_type = CharField(null=True, max_length=30)
    last_source_id = CharField(null=True, max_length=128)
    last_sent = DateTimeField(null=False)

    class Meta:
        table_name = "notification_rule_state"
        primary_key = CompositeKey("rule_id", "camera", "channel", "recipient_id")
