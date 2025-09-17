import logging
import psycopg2
from psycopg2 import extras
from sqlalchemy import create_engine
from config import DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT
import os
import re

logger = logging.getLogger(__name__)

# Memoization cache for database engines
_engine_cache = {}

def get_db_connection(dbname=DB_NAME):
    """
    Establishes and retrieves a database engine, caching the engine for reuse.
    """
    if dbname in _engine_cache:
        return _engine_cache[dbname]

    try:
        engine = create_engine(
            f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{dbname}',
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20
        )
        _engine_cache[dbname] = engine
        logger.info(f"Database connection to {dbname} established successfully.")
        return engine
    except Exception as e:
        logger.error(f"Error connecting to database {dbname}: {e}")
        return None

def apply_migrations(migration_dir="database/migrations"):
    from sqlalchemy import text

    engine_postgres = None
    engine = None
    try:
        # Connect to the default 'postgres' database to create the new database if it doesn't exist
        engine_postgres = get_db_connection(dbname="postgres")
        if engine_postgres:
            with engine_postgres.connect() as conn_postgres:
                conn_postgres.execute(text("COMMIT;"))
                result = conn_postgres.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"))
                exists = result.fetchone()
                if not exists:
                    conn_postgres.execute(text(f"CREATE DATABASE {DB_NAME};"))
                    logger.info(f"Database '{DB_NAME}' created successfully.")
        else:
            logger.error("Could not connect to 'postgres' database to check/create defiyieldopt. Exiting migration.")
            return

        # Now connect to the actual application database
        engine = get_db_connection()
        if not engine:
            logger.error("Could not establish database connection to defiyieldopt. Exiting migration.")
            return

        with engine.begin() as conn:
            # Create a table to track applied migrations
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS applied_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))

            migrations = sorted(
                [f for f in os.listdir(migration_dir) if f.endswith(".sql")],
                key=lambda f: int(re.match(r"V(\d+)__.*\.sql", f).group(1))
            )

            for migration_file in migrations:
                version_match = re.match(r"V(\d+)__.*\.sql", migration_file)
                if not version_match:
                    logger.warning(f"Skipping invalid migration file name: {migration_file}")
                    continue
                version = version_match.group(1)

                result = conn.execute(text("SELECT 1 FROM applied_migrations WHERE version = :version"), {"version": version})
                applied = result.fetchone()
                if applied:
                    logger.info(f"Migration {migration_file} already applied. Skipping.")
                    continue

                filepath = os.path.join(migration_dir, migration_file)
                with open(filepath, 'r') as f:
                    sql_script = f.read()
                    # Remove the CREATE DATABASE and \c commands from the migration script
                    # as the database is created and connected to separately
                    sql_script = re.sub(r"CREATE DATABASE\s+\w+;", "", sql_script, flags=re.IGNORECASE)
                    sql_script = re.sub(r"\\c\s+\w+;", "", sql_script, flags=re.IGNORECASE)
                    
                    conn.execute(text(sql_script))
                    conn.execute(text("INSERT INTO applied_migrations (version) VALUES (:version)"), {"version": version})
                    logger.info(f"Successfully applied migration: {migration_file}")

        logger.info("All migrations applied successfully.")
    except Exception as e:
        logger.error(f"Error applying migrations: {e}")
    finally:
        if engine_postgres:
            engine_postgres.dispose()
        if engine:
            engine.dispose()

if __name__ == "__main__":
    apply_migrations()