"""Create the canonical event observation, evidence, artifact and intent stores."""


def _columns(database, table, fake):
    if fake:
        return set()
    return {row[1] for row in database.execute_sql(f"PRAGMA table_info({table})")}


def migrate(migrator, database, fake=False, **kwargs):
    event_columns = _columns(database, "event", fake)
    additions = {
        "state": "VARCHAR(20) NOT NULL DEFAULT 'TRACKING'",
        "revision": "INTEGER NOT NULL DEFAULT 0",
        "finalized_at": "DATETIME",
        "canonical_plate": "VARCHAR(32)",
        "canonical_plate_score": "REAL",
        "canonical_sub_label": "VARCHAR(100)",
        "display_label": "VARCHAR(100)",
        "canonical_evidence_id": "VARCHAR(64)",
        "canonical_artifact_id": "VARCHAR(64)",
    }
    for name, sql_type in additions.items():
        if name not in event_columns:
            migrator.sql(f"ALTER TABLE event ADD COLUMN {name} {sql_type}")

    migrator.sql("CREATE INDEX IF NOT EXISTS event_state ON event(state)")
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS event_observation (
            observation_id VARCHAR(64) PRIMARY KEY, event_id VARCHAR(30) NOT NULL,
            kind VARCHAR(32) NOT NULL, observed_at DATETIME NOT NULL,
            frame_time REAL, evidence_id VARCHAR(64), payload TEXT NOT NULL,
            expires_at DATETIME NOT NULL)
    """)
    migrator.sql("CREATE INDEX IF NOT EXISTS event_observation_event ON event_observation(event_id)")
    migrator.sql("CREATE INDEX IF NOT EXISTS event_observation_expiry ON event_observation(expires_at)")
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS event_evidence (
            id VARCHAR(64) PRIMARY KEY, event_id VARCHAR(30) NOT NULL,
            frame_ref TEXT NOT NULL, frame_time REAL NOT NULL, width INTEGER NOT NULL,
            height INTEGER NOT NULL, boxes TEXT NOT NULL, technical TEXT NOT NULL,
            created_at DATETIME NOT NULL)
    """)
    migrator.sql("CREATE INDEX IF NOT EXISTS event_evidence_event ON event_evidence(event_id)")
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS media_artifact (
            id VARCHAR(64) PRIMARY KEY, event_id VARCHAR(30) NOT NULL,
            revision INTEGER NOT NULL, evidence_id VARCHAR(64) NOT NULL,
            profile VARCHAR(32) NOT NULL, render_version INTEGER NOT NULL,
            path TEXT NOT NULL UNIQUE, sha256 VARCHAR(64) NOT NULL,
            byte_size INTEGER NOT NULL, manifest TEXT NOT NULL,
            created_at DATETIME NOT NULL, expires_at DATETIME NOT NULL,
            pinned INTEGER NOT NULL DEFAULT 0)
    """)
    migrator.sql("CREATE UNIQUE INDEX IF NOT EXISTS media_artifact_spec ON media_artifact(event_id, revision, evidence_id, profile, render_version)")
    migrator.sql("CREATE INDEX IF NOT EXISTS media_artifact_expiry ON media_artifact(expires_at)")
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS notification_intent (
            id VARCHAR(64) PRIMARY KEY, event_id VARCHAR(30) NOT NULL,
            revision INTEGER NOT NULL, channel VARCHAR(20) NOT NULL,
            recipient_id VARCHAR(128) NOT NULL, facts TEXT NOT NULL,
            caption TEXT NOT NULL, actions TEXT NOT NULL,
            media_artifact_id VARCHAR(64), status VARCHAR(20) NOT NULL DEFAULT 'pending',
            created_at DATETIME NOT NULL)
    """)
    migrator.sql("CREATE UNIQUE INDEX IF NOT EXISTS notification_intent_dedupe ON notification_intent(event_id, revision, channel, recipient_id)")

    delivery_columns = _columns(database, "notification_delivery", fake)
    if "intent_id" not in delivery_columns:
        migrator.sql("ALTER TABLE notification_delivery ADD COLUMN intent_id VARCHAR(64)")
    if "media_artifact_id" not in delivery_columns:
        migrator.sql("ALTER TABLE notification_delivery ADD COLUMN media_artifact_id VARCHAR(64)")


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP TABLE IF EXISTS notification_intent")
    migrator.sql("DROP TABLE IF EXISTS media_artifact")
    migrator.sql("DROP TABLE IF EXISTS event_evidence")
    migrator.sql("DROP TABLE IF EXISTS event_observation")
