import logging
from database.db_utils import get_db_connection
from sqlalchemy import text

logger = logging.getLogger(__name__)

def filter_pools_pre_debug():
    """
    Debug version of filter_pools_pre that adds diagnostic logging to validate database schema.
    """
    logger.info("Starting DEBUG pre-pool filtering...")
    engine = get_db_connection()
    if not engine:
        logger.error("Could not establish database connection. Exiting.")
        return

    with engine.connect() as conn:
        try:
            # DEBUG: Check database connection details
            result = conn.execute(text("SELECT current_database(), current_user, version();"))
            db_info = result.fetchone()
            logger.info(f"=== DATABASE CONNECTION INFO ===")
            logger.info(f"Database: {db_info[0]}")
            logger.info(f"User: {db_info[1]}")
            logger.info(f"PostgreSQL Version: {db_info[2]}")
            logger.info(f"================================")
            
            # DEBUG: Check if pools table exists
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'pools'
                );
            """))
            pools_exists = result.fetchone()[0]
            logger.info(f"Pools table exists: {pools_exists}")
            
            if pools_exists:
                # DEBUG: Check pools table structure
                result = conn.execute(text("""
                    SELECT column_name, data_type, column_default
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' 
                    AND table_name = 'pools'
                    ORDER BY ordinal_position;
                """))
                columns = result.fetchall()
                logger.info(f"=== POOLS TABLE SCHEMA ===")
                for col in columns:
                    logger.info(f"Column: {col[0]}, Type: {col[1]}, Default: {col[2]}")
                logger.info(f"===========================")
                
                # DEBUG: Check if currently_filtered_out column exists
                has_filtered_column = any(col[0] == 'currently_filtered_out' for col in columns)
                logger.info(f"currently_filtered_out column exists: {has_filtered_column}")
                
                if not has_filtered_column:
                    logger.error("MISSING COLUMN: currently_filtered_out not found in pools table!")
                    
                    # DEBUG: Check applied migrations
                    result = conn.execute(text("""
                        SELECT version, applied_at 
                        FROM applied_migrations 
                        ORDER BY applied_at DESC;
                    """))
                    migrations = result.fetchall()
                    logger.info(f"=== APPLIED MIGRATIONS ===")
                    for migration in migrations:
                        logger.info(f"Migration: {migration[0]}, Applied: {migration[1]}")
                    logger.info(f"===========================")
                    
                    # DEBUG: Check if V16 migration was applied
                    v16_applied = any(migration[0] == '16' for migration in migrations)
                    logger.info(f"V16 migration (add_currently_filtered_out_to_pools) applied: {v16_applied}")
                    
                    if not v16_applied:
                        logger.error("V16 migration was not applied! This is the root cause.")
                    else:
                        logger.error("V16 migration was applied but column still missing! Possible migration failure.")
                else:
                    logger.info("currently_filtered_out column exists - proceeding with filtering")
                    
                    # DEBUG: Test a simple query on the column
                    try:
                        result = conn.execute(text("SELECT COUNT(*) FROM pools WHERE currently_filtered_out = FALSE;"))
                        count = result.fetchone()[0]
                        logger.info(f"Test query successful: {count} pools with currently_filtered_out = FALSE")
                    except Exception as e:
                        logger.error(f"Test query failed: {e}")
            else:
                logger.error("Pools table does not exist!")
                
        except Exception as e:
            logger.error(f"Error during diagnostic checks: {e}")
            import traceback
            traceback.print_exc()
            
    engine.dispose()

if __name__ == "__main__":
    filter_pools_pre_debug()