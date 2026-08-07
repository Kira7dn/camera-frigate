"""Add the notification rule to the durable delivery identity."""


def migrate(migrator, database, fake=False, **kwargs):
    columns = {
        row[1]
        for row in database.execute_sql("PRAGMA table_info(notification_delivery)")
    }
    if "rule_id" not in columns:
        migrator.sql(
            "ALTER TABLE notification_delivery "
            "ADD COLUMN rule_id VARCHAR(64) NOT NULL DEFAULT 'legacy'"
        )
    migrator.sql("DROP INDEX IF EXISTS notification_delivery_dedupe")
    migrator.sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS notification_delivery_rule_dedupe "
        "ON notification_delivery(provider, recipient_id, rule_id, source_type, source_id)"
    )


def rollback(migrator, database, fake=False, **kwargs):
    migrator.sql("DROP INDEX IF EXISTS notification_delivery_rule_dedupe")
    migrator.sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS notification_delivery_dedupe "
        "ON notification_delivery(provider, recipient_id, source_type, source_id)"
    )
