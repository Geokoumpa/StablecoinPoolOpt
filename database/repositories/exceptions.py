class RepositoryError(Exception):
    """Base exception for all repository errors."""
    pass

class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found in the database by ID or other criteria."""
    pass

class DuplicateEntityError(RepositoryError):
    """Raised when an attempt is made to create an entity that violates a unique constraint."""
    pass

class DatabaseConnectionError(RepositoryError):
    """Raised when the repository cannot connect to the database."""
    pass

class DataValidationError(RepositoryError):
    """Raised when data provided to the repository is invalid."""
    pass
