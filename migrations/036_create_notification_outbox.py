"""Create the durable social notification outbox."""


def migrate(migrator, database, fake=False, **kwargs):
    migrator.sql(
        """
        CREATE TABLE IF NOT EXISTS notification_delivery (
            id VARCHAR(36) NOT NULL PRIMARY KEY,
            provider VARCHAR(20) NOT NULL,
            recipient_id VARCHAR(64) NOT NULL,
            source_type VARCHAR(30) NOT NULL,
            source_id VARCHAR(64) NOT NULL,
            payload TEXT NOT NULL,
            status VARCHAR(20) NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt DATETIME NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            completed_at DATETIME,
            last_error TEXT
        )
        """
    )
    migrator.sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS notification_delivery_dedupe "
        "ON notification_delivery(provider, recipient_id, source_type, source_id)"
    )
    migrator.sql(
        "CREATE INDEX IF NOT EXISTS notification_delivery_status "
        "ON notification_delivery(status)"
    )
    migrator.sql(
        "CREATE INDEX IF NOT EXISTS notification_delivery_next_attempt "
        "ON notification_delivery(next_attempt)"
    )


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP TABLE IF EXISTS notification_delivery")
