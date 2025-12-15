class RepositoryError(Exception):
    """Base exception for all repository errors."""
    pass



class DuplicateEntityError(RepositoryError):
    """Raised when an attempt is made to create an entity that violates a unique constraint."""
    pass

class DatabaseConnectionError(RepositoryError):
    """Raised when the repository cannot connect to the database."""
    pass


