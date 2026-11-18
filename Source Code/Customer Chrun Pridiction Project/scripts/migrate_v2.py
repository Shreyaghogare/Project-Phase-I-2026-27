from __future__ import annotations

from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "instance" / "churn.sqlite"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DB_PATH)


def column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row[1] == column_name for row in rows)


def add_column_if_missing(connection: sqlite3.Connection, table_name: str, column_sql: str) -> None:
    column_name = column_sql.split()[0]
    if not column_exists(connection, table_name, column_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql}")
        print(f"Added {table_name}.{column_name}")


def make_prediction_inputs_nullable(connection: sqlite3.Connection) -> None:
    rows = connection.execute("PRAGMA table_info(prediction)").fetchall()
    nullable_input_columns = {
        "age", "gender", "tenure", "monthly_charges", "total_charges",
        "contract_type", "internet_service", "tech_support",
    }
    if not any(row[1] in nullable_input_columns and row[3] for row in rows):
        return

    index_and_trigger_sql = connection.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE tbl_name='prediction' AND type IN ('index', 'trigger') AND sql IS NOT NULL"
    ).fetchall()
    columns = [row[1] for row in rows]
    connection.execute("PRAGMA foreign_keys=OFF")
    try:
        connection.execute("BEGIN")
        connection.execute(
            """
            CREATE TABLE prediction_rebuild (
                id INTEGER NOT NULL PRIMARY KEY,
                user_id INTEGER NOT NULL,
                company_id INTEGER,
                model_id INTEGER,
                age INTEGER,
                gender VARCHAR(30),
                tenure INTEGER,
                monthly_charges FLOAT,
                total_charges FLOAT,
                contract_type VARCHAR(50),
                internet_service VARCHAR(50),
                tech_support VARCHAR(30),
                prediction VARCHAR(3) NOT NULL,
                probability FLOAT NOT NULL,
                risk_level VARCHAR(20) NOT NULL,
                risk_factors JSON NOT NULL,
                suggested_actions JSON NOT NULL,
                input_json JSON,
                created_at DATETIME NOT NULL,
                FOREIGN KEY(user_id) REFERENCES user(id),
                FOREIGN KEY(company_id) REFERENCES company(id),
                FOREIGN KEY(model_id) REFERENCES trained_model(id)
            )
            """
        )
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        connection.execute(
            f"INSERT INTO prediction_rebuild ({quoted_columns}) "
            f"SELECT {quoted_columns} FROM prediction"
        )
        connection.execute("DROP TABLE prediction")
        connection.execute("ALTER TABLE prediction_rebuild RENAME TO prediction")
        for (statement,) in index_and_trigger_sql:
            connection.execute(statement)
        connection.commit()
        print("Rebuilt prediction table to allow company-specific input schemas")
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys=ON")


def ensure_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS company (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            slug TEXT NOT NULL UNIQUE,
            industry TEXT NOT NULL,
            description TEXT,
            logo_color TEXT DEFAULT '#4f46e5',
            is_active BOOLEAN NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_by INTEGER,
            FOREIGN KEY(created_by) REFERENCES user(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS company_dataset (
            id INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            row_count INTEGER NOT NULL DEFAULT 0,
            uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            uploaded_by INTEGER,
            is_active BOOLEAN NOT NULL DEFAULT 1,
            schema_json TEXT,
            data_path TEXT,
            FOREIGN KEY(company_id) REFERENCES company(id),
            FOREIGN KEY(uploaded_by) REFERENCES user(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS trained_model (
            id INTEGER PRIMARY KEY,
            company_id INTEGER NOT NULL,
            dataset_id INTEGER NOT NULL,
            algorithm TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            trained_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            metrics_json TEXT,
            feature_importance_json TEXT,
            model_path TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT 0,
            threshold_medium REAL NOT NULL DEFAULT 0.3,
            threshold_high REAL NOT NULL DEFAULT 0.6,
            FOREIGN KEY(company_id) REFERENCES company(id),
            FOREIGN KEY(dataset_id) REFERENCES company_dataset(id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS user_company_access (
            user_id INTEGER NOT NULL,
            company_id INTEGER NOT NULL,
            granted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, company_id),
            FOREIGN KEY(user_id) REFERENCES user(id),
            FOREIGN KEY(company_id) REFERENCES company(id)
        )
        """
    )


def migrate() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    ensure_tables(connection)

    if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prediction'").fetchone() is not None:
        add_column_if_missing(connection, "prediction", "company_id INTEGER")
        add_column_if_missing(connection, "prediction", "model_id INTEGER")
        add_column_if_missing(connection, "prediction", "input_json TEXT")
        make_prediction_inputs_nullable(connection)

    for table_name, column_sql_list in {
        "user": ["created_at TEXT DEFAULT CURRENT_TIMESTAMP"],
        "company": [
            "created_by INTEGER",
            "created_at TEXT DEFAULT CURRENT_TIMESTAMP",
            "logo_color TEXT DEFAULT '#4f46e5'",
            "is_active BOOLEAN DEFAULT 1",
        ],
        "company_dataset": [
            "uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP",
            "uploaded_by INTEGER",
            "is_active BOOLEAN DEFAULT 1",
            "schema_json TEXT",
            "data_path TEXT",
        ],
        "trained_model": [
            "trained_at TEXT DEFAULT CURRENT_TIMESTAMP",
            "metrics_json TEXT",
            "feature_importance_json TEXT",
            "is_active BOOLEAN DEFAULT 0",
            "threshold_medium REAL DEFAULT 0.3",
            "threshold_high REAL DEFAULT 0.6",
        ],
    }.items():
        if connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)).fetchone() is not None:
            for column_sql in column_sql_list:
                add_column_if_missing(connection, table_name, column_sql)

    connection.commit()
    connection.close()
    print(f"Migration complete for {DB_PATH}")


if __name__ == "__main__":
    migrate()
