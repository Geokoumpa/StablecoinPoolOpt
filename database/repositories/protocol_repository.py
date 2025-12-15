
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




    def sync_approved_protocols(self, protocols_names: List[str]) -> None:
        """
        Bulk upsert approved protocols.
        """
        if not protocols_names:
            return

        from datetime import datetime
        now_val = datetime.now()

        sql = """
            INSERT INTO approved_protocols (protocol_name, added_timestamp)
            VALUES %s
            ON CONFLICT (protocol_name) DO UPDATE SET
                removed_timestamp = NULL
        """
        
        values = [(p, now_val) for p in protocols_names]
        self.execute_bulk_values(sql, values)
