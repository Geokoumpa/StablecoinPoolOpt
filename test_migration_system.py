#!/usr/bin/env python3
"""
Test script for the enhanced migration system
This will help us verify that the migration system works correctly before deploying to production.
"""

import logging
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_migration_system():
    """Test the enhanced migration system."""
    logger.info("=== TESTING ENHANCED MIGRATION SYSTEM ===")
    
    try:
        # Import after path setup
        from database.db_utils import apply_migrations, get_db_connection
        from sqlalchemy import text
        
        # Test database connection first
        logger.info("Testing database connection...")
        engine = get_db_connection()
        if not engine:
            logger.error("❌ Database connection failed. Please check your .env file.")
            return False
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT current_database(), current_user()"))
            db_info = result.fetchone()
            logger.info(f"✅ Connected to database: {db_info[0]} as user: {db_info[1]}")
        
        engine.dispose()
        
        # Test migration system
        logger.info("Testing migration system...")
        apply_migrations()
        
        logger.info("✅ Migration system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration system test failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_migration_system()
    if success:
        logger.info("🎉 All tests passed! The migration system is ready for production.")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed! Please fix the issues before deploying to production.")
        sys.exit(1)