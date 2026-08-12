"""Create durable tracker journal acceptance and private edge media manifests."""


def migrate(migrator, database, fake=False, **kwargs):
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS tracker_journal_entry (
            node_id VARCHAR(64) NOT NULL,
            node_epoch VARCHAR(64) NOT NULL,
            journal_sequence INTEGER NOT NULL,
            camera_id VARCHAR(64) NOT NULL,
            stream_epoch VARCHAR(64) NOT NULL,
            event_id VARCHAR(30) NOT NULL,
            operation VARCHAR(16) NOT NULL,
            payload TEXT NOT NULL,
            accepted_at DATETIME NOT NULL,
            PRIMARY KEY (node_id, node_epoch, journal_sequence))
    """)
    migrator.sql(
        "CREATE INDEX IF NOT EXISTS tracker_journal_event "
        "ON tracker_journal_entry(event_id)"
    )
    migrator.sql("""
        CREATE TABLE IF NOT EXISTS edge_media_manifest (
            media_id VARCHAR(128) PRIMARY KEY,
            node_id VARCHAR(64) NOT NULL,
            camera_id VARCHAR(64) NOT NULL,
            event_id VARCHAR(30) NOT NULL,
            media_type VARCHAR(24) NOT NULL,
            codec VARCHAR(32) NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            byte_size INTEGER NOT NULL,
            sha256 VARCHAR(64) NOT NULL,
            expires_at DATETIME NOT NULL,
            retained INTEGER NOT NULL DEFAULT 0)
    """)
    migrator.sql(
        "CREATE INDEX IF NOT EXISTS edge_media_manifest_event "
        "ON edge_media_manifest(event_id)"
    )
    migrator.sql(
        "CREATE INDEX IF NOT EXISTS edge_media_manifest_expiry "
        "ON edge_media_manifest(expires_at)"
    )


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP TABLE IF EXISTS edge_media_manifest")
    migrator.sql("DROP TABLE IF EXISTS tracker_journal_entry")
