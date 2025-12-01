import logging
import psycopg2
from psycopg2 import extras
from sqlalchemy import create_engine, text
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
        logger.debug(f"Using cached connection for database: {dbname}")
        return _engine_cache[dbname]

    # Validate required parameters
    if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
        logger.error("❌ Missing required database connection parameters:")
        logger.error(f"   DB_USER: {'✅' if DB_USER else '❌ MISSING'}")
        logger.error(f"   DB_PASSWORD: {'✅' if DB_PASSWORD else '❌ MISSING'}")
        logger.error(f"   DB_HOST: {'✅' if DB_HOST else '❌ MISSING'}")
        logger.error(f"   DB_PORT: {'✅' if DB_PORT else '❌ MISSING'}")
        logger.error(f"   DB_NAME: {'✅' if dbname else '❌ MISSING'}")
        return None

    connection_string = f'postgresql+psycopg2://{DB_USER}:***@{DB_HOST}:{DB_PORT}/{dbname}'
    logger.info(f"🔄 Establishing new database connection to {dbname} at {DB_HOST}:{DB_PORT}")
    
    try:
        engine = create_engine(
            f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{dbname}',
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
            # Add connection timeout and retry logic
            connect_args={
                "connect_timeout": 30,
                "application_name": "defi_pipeline",
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5
            }
        )
        
        # Test the connection with retries
        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT 1"))
                    if result.fetchone()[0] != 1:
                        raise Exception("Connection test failed")
                break # Connection successful
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Connection attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"❌ All connection attempts failed.")
                    raise e
        
        _engine_cache[dbname] = engine
        logger.info(f"✅ Database connection to {dbname} established successfully.")
        logger.info(f"   Connection string: {connection_string}")
        return engine
        
    except Exception as e:
        logger.error(f"❌ Error connecting to database {dbname}: {e}")
        logger.error(f"   Connection string: {connection_string}")
        logger.error(f"   This might be due to:")
        logger.error(f"   - Database server not running")
        logger.error(f"   - Incorrect host/port ({DB_HOST}:{DB_PORT})")
        logger.error(f"   - Invalid credentials")
        logger.error(f"   - Network connectivity issues")
        logger.error(f"   - Database '{dbname}' not existing")
        return None

def apply_migrations(migration_dir="database/migrations"):
    from sqlalchemy import text

    logger.info("=== STARTING MIGRATION PROCESS ===")
    logger.info(f"Database: {DB_NAME}, Host: {DB_HOST}, Port: {DB_PORT}, User: {DB_USER}")
    
    engine = None
    migrations_applied = []
    migrations_skipped = []
    
    try:
        # For Cloud Run environments, the database should already exist
        # Skip trying to connect to postgres database as it may not be accessible
        logger.info("Checking if target database exists...")
        logger.info("Note: In Cloud Run, the database should already exist and be accessible")
        
        # Try to connect directly to the target database first
        logger.info(f"Attempting to connect directly to database '{DB_NAME}'...")
        try:
            engine = get_db_connection()
            if not engine:
                logger.error(f"❌ Could not connect to target database '{DB_NAME}'.")
                # Only try postgres database in development/local environments
                if os.getenv("ENVIRONMENT", "development") == "development":
                    logger.info("Attempting to connect to 'postgres' database (local development only)...")
                    engine_postgres = get_db_connection(dbname="postgres")
                    if engine_postgres:
                        with engine_postgres.connect() as conn_postgres:
                            conn_postgres.execute(text("COMMIT;"))
                            result = conn_postgres.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'"))
                            exists = result.fetchone()
                            if not exists:
                                logger.info(f"Database '{DB_NAME}' does not exist. Creating...")
                                conn_postgres.execute(text(f"CREATE DATABASE {DB_NAME};"))
                                logger.info(f"✅ Database '{DB_NAME}' created successfully.")
                            else:
                                logger.info(f"✅ Database '{DB_NAME}' already exists.")
                        engine_postgres.dispose()
                        # Now try to connect to the created database
                        engine = get_db_connection()
                        if not engine:
                            logger.error(f"❌ Still could not connect to database '{DB_NAME}' after creation.")
                            raise Exception("Failed to connect to database after creation")
                    else:
                        logger.error("❌ Could not connect to 'postgres' database in development mode.")
                        raise Exception("Failed to connect to postgres database in development")
                else:
                    logger.error("❌ In production mode, database should already exist and be accessible.")
                    logger.error(f"❌ Connection details - Host: {DB_HOST}, Port: {DB_PORT}, User: {DB_USER}, Database: {DB_NAME}")
                    raise Exception("Database connection failed in production - database should already exist")
            else:
                logger.info(f"✅ Successfully connected to database '{DB_NAME}'")
        except Exception as conn_error:
            logger.error(f"❌ Database connection error: {str(conn_error)}")
            logger.error(f"❌ Connection details - Host: {DB_HOST}, Port: {DB_PORT}, User: {DB_USER}, Database: {DB_NAME}")
            raise Exception("Database connection failed in production - database should already exist")

        # Now connect to the actual application database
        logger.info(f"Connecting to application database '{DB_NAME}'...")
        engine = get_db_connection()
        if not engine:
            logger.error(f"❌ Could not establish database connection to {DB_NAME}. Exiting migration.")
            raise Exception("Failed to connect to application database")

        logger.info("✅ Successfully connected to application database")

        with engine.begin() as conn:
            # Create a table to track applied migrations
            logger.info("Ensuring applied_migrations table exists...")
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS applied_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(255) UNIQUE NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
            logger.info("✅ applied_migrations table ready")

            # Get list of migration files
            logger.info(f"Scanning migration directory: {migration_dir}")
            if not os.path.exists(migration_dir):
                logger.error(f"❌ Migration directory does not exist: {migration_dir}")
                raise Exception(f"Migration directory not found: {migration_dir}")
                
            migration_files = [f for f in os.listdir(migration_dir) if f.endswith(".sql")]
            logger.info(f"Found {len(migration_files)} migration files: {migration_files}")
            
            migrations = sorted(
                migration_files,
                key=lambda f: int(re.match(r"V(\d+)__.*\.sql", f).group(1))
            )

            # Check current migration state
            result = conn.execute(text("SELECT version, applied_at FROM applied_migrations ORDER BY version::int"))
            applied_migrations = {row[0]: row[1] for row in result.fetchall()}
            logger.info(f"Currently applied migrations: {sorted(applied_migrations.keys(), key=int)}")

            for migration_file in migrations:
                version_match = re.match(r"V(\d+)__.*\.sql", migration_file)
                if not version_match:
                    logger.warning(f"⚠️ Skipping invalid migration file name: {migration_file}")
                    continue
                version = version_match.group(1)

                if version in applied_migrations:
                    logger.info(f"⏭️ Migration {migration_file} already applied at {applied_migrations[version]}. Skipping.")
                    migrations_skipped.append(migration_file)
                    continue

                logger.info(f"🔄 Applying migration: {migration_file}")
                filepath = os.path.join(migration_dir, migration_file)
                
                if not os.path.exists(filepath):
                    logger.error(f"❌ Migration file not found: {filepath}")
                    raise Exception(f"Migration file not found: {filepath}")
                
                with open(filepath, 'r') as f:
                    sql_script = f.read()
                    
                    # Log the SQL script for debugging (first 200 chars)
                    logger.debug(f"Migration SQL preview: {sql_script[:200]}...")
                    
                    # Remove the CREATE DATABASE and \c commands from the migration script
                    # as the database is created and connected to separately
                    sql_script = re.sub(r"CREATE DATABASE\s+\w+;", "", sql_script, flags=re.IGNORECASE)
                    sql_script = re.sub(r"\\c\s+\w+;", "", sql_script, flags=re.IGNORECASE)
                    
                    if not sql_script.strip():
                        logger.warning(f"⚠️ Migration {migration_file} appears to be empty after cleaning")
                        continue
                    
                    # Execute the migration
                    try:
                        conn.execute(text(sql_script))
                        conn.execute(text("INSERT INTO applied_migrations (version) VALUES (:version)"), {"version": version})
                        logger.info(f"✅ Successfully applied migration: {migration_file}")
                        migrations_applied.append(migration_file)
                    except Exception as migration_error:
                        logger.error(f"❌ Failed to apply migration {migration_file}: {migration_error}")
                        logger.error(f"SQL that failed: {sql_script[:500]}...")
                        raise migration_error

            # Final status
            logger.info("=== MIGRATION SUMMARY ===")
            logger.info(f"✅ Applied migrations: {len(migrations_applied)} - {migrations_applied}")
            logger.info(f"⏭️ Skipped migrations: {len(migrations_skipped)} - {migrations_skipped}")
            logger.info(f"🎉 Migration process completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Error applying migrations: {e}")
        logger.error(f"❌ Migration process failed!")
        import traceback
        logger.error(f"Full error traceback: {traceback.format_exc()}")
        raise e
    finally:
        if engine:
            engine.dispose()

if __name__ == "__main__":
    apply_migrations()