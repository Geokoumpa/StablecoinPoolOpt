
import logging
from contextlib import contextmanager
from typing import Any, List, Optional, Tuple, Dict, TypeVar, Generic, Type

from sqlalchemy import text, Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import psycopg2
import psycopg2.extras

from database.db_utils import get_db_connection
from database.repositories.exceptions import (
    RepositoryError,
    DatabaseConnectionError,
    DuplicateEntityError,
    DuplicateEntityError,
)

# Set up logger
logger = logging.getLogger(__name__)

# Type variable for ORM models
T = TypeVar("T")

class BaseRepository(Generic[T]):
    """
    Base repository class providing common database operations and connection management.
    This class handles connection pooling, transaction management, and common CRUD operations.
    """

    def __init__(self, model_class: Type[T] = None):
        """
        Initialize the repository.
        
        Args:
            model_class: The SQLAlchemy model class this repository manages (optional)
        """
        self._engine: Optional[Engine] = None
        self.model_class = model_class
        self._session_factory = None

    @property
    def engine(self) -> Engine:
        """Lazy load the database engine."""
        if self._engine is None:
            self._engine = get_db_connection()
            if self._engine is None:
                raise DatabaseConnectionError("Failed to obtain database connection")
        return self._engine

    @property
    def session_factory(self):
        """Lazy load the session factory."""
        if self._session_factory is None:
            self._session_factory = sessionmaker(bind=self.engine)
        return self._session_factory

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions using SQLAlchemy Connection.
        Useful for raw SQL execution.
        """
        conn = self.engine.connect()
        trans = conn.begin()
        try:
            yield conn
            trans.commit()
        except IntegrityError as e:
            trans.rollback()
            logger.error(f"Integrity Error in transaction: {e}")
            if "unique constraint" in str(e).lower():
                raise DuplicateEntityError(f"Duplicate entity: {e}")
            raise RepositoryError(f"Database integrity error: {e}")
        except SQLAlchemyError as e:
            trans.rollback()
            logger.error(f"Database Error in transaction: {e}")
            raise RepositoryError(f"Database error: {e}")
        except Exception as e:
            trans.rollback()
            logger.error(f"Unexpected error in transaction: {e}")
            raise e
        finally:
            conn.close()

    @contextmanager
    def session(self) -> Session:
        """
        Context manager for ORM sessions.
        Handles commit/rollback automatically.
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.error(f"Integrity Error in session: {e}")
            if "unique constraint" in str(e).lower():
                raise DuplicateEntityError(f"Duplicate entity: {e}")
            raise RepositoryError(f"Database integrity error: {e}")
        except SQLAlchemyError as e:
            session.rollback()
            logger.error(f"Database Error in session: {e}")
            raise RepositoryError(f"Database error: {e}")
        except Exception as e:
            session.rollback()
            logger.error(f"Unexpected error in session: {e}")
            raise e
        finally:
            session.close()

    def execute_bulk_values(self, sql: str, values: List[Tuple], page_size: int = 1000, template: Optional[str] = None) -> None:
        """
        Executes a bulk insert/update using psycopg2.extras.execute_values for performance.
        
        Args:
            sql: The SQL statement (e.g. "INSERT INTO table (a, b) VALUES %s")
            values: List of tuples representing rows
            page_size: Batch size for execution
            template: Optional template for values
        """
        if not values:
            return

        # Use raw connection for psycopg2 performance
        conn = self.engine.raw_connection()
        try:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(
                    cur, sql, values, template=template, page_size=page_size
                )
            conn.commit()
        except psycopg2.Error as e:
            conn.rollback()
            logger.error(f"Bulk execution failed: {e}")
            if hasattr(e, 'pgcode') and e.pgcode == '23505': # Unique violation
                raise DuplicateEntityError(f"Bulk operation failed due to duplicate keys: {e}")
            raise RepositoryError(f"Bulk operation failed: {e}")
        finally:
            conn.close()

    # Common CRUD Patterns
    
    def create(self, entity: T) -> T:
        """Create a new entity."""
        with self.session() as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            return entity





    def update(self, entity: T) -> T:
        """Update an existing entity."""
        with self.session() as session:
            merged = session.merge(entity)
            return merged

    def delete(self, entity: T) -> None:
        """Delete an entity."""
        with self.session() as session:
            session.delete(entity)

