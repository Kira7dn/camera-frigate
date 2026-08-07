"""Persist notification cooldown and WebPush deduplication state."""


def migrate(migrator, database, fake=False, **kwargs):
    migrator.sql(
        """
        CREATE TABLE IF NOT EXISTS notification_rule_state (
            rule_id VARCHAR(64) NOT NULL,
            camera VARCHAR(64) NOT NULL,
            channel VARCHAR(20) NOT NULL,
            recipient_id VARCHAR(128) NOT NULL,
            last_source_type VARCHAR(30),
            last_source_id VARCHAR(128),
            last_sent DATETIME NOT NULL,
            PRIMARY KEY (rule_id, camera, channel, recipient_id)
        )
        """
    )


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP TABLE IF EXISTS notification_rule_state")
