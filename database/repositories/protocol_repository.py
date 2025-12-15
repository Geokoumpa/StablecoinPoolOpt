
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, text
from database.models.protocol import ApprovedProtocol
from database.repositories.base_repository import BaseRepository

class ProtocolRepository(BaseRepository[ApprovedProtocol]):
    """
    Repository for Protocol management (Approved Protocols).
    """
    def __init__(self):
        super().__init__(model_class=ApprovedProtocol)

    def get_approved_protocols(self) -> List[ApprovedProtocol]:
        """Get all approved protocols."""
        with self.session() as session:
            results = session.execute(select(ApprovedProtocol).where(ApprovedProtocol.removed_timestamp.is_(None))).scalars().all()
            session.expunge_all()
            return results





