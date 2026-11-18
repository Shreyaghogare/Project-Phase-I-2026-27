from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "instance" / "churn.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def add_column_if_missing(connection: sqlite3.Connection, table_name: str, column_sql: str) -> None:
    column_name = column_sql.split()[0]
    if not column_exists(connection, table_name, column_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
        print(f"Added {table_name}.{column_name}")


def backup_database() -> None:
    if DB_PATH.exists():
        backup_path = DB_PATH.with_name("churn.sqlite.pre-v3.bak")
        shutil.copy2(DB_PATH, backup_path)
        print(f"Backed up database to {backup_path}")


def ensure_email_otp_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS email_otp (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            code_hash VARCHAR(255) NOT NULL,
            purpose VARCHAR(50) NOT NULL DEFAULT 'verify_email',
            expires_at DATETIME NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            used_at DATETIME,
            FOREIGN KEY(user_id) REFERENCES user(id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ix_email_otp_user_id ON email_otp (user_id)"
    )


def migrate() -> None:
    backup_database()
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    # Add email verification columns to user table
    if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'").fetchone() is not None:
        add_column_if_missing(connection, "user", "email_verified BOOLEAN NOT NULL DEFAULT 0")
        add_column_if_missing(connection, "user", "email_verified_at TEXT")

        # Set email_verified=True for all existing users so they can still log in
        updated = connection.execute(
            "UPDATE user SET email_verified = 1, email_verified_at = CURRENT_TIMESTAMP WHERE email_verified = 0 OR email_verified IS NULL"
        ).rowcount
        print(f"Marked {updated} existing user(s) as verified.")

    # Create email_otp table
    ensure_email_otp_table(connection)

    connection.commit()
    connection.close()
    print(f"Migration v3 complete for {DB_PATH}")


if __name__ == "__main__":
    migrate()
