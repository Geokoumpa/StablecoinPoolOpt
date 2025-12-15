
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy import select, func, text
from database.models.token import ApprovedToken, BlacklistedToken, IceboxToken
from database.repositories.base_repository import BaseRepository

class TokenRepository(BaseRepository[ApprovedToken]):
    """
    Repository for Token management (Approved, Blacklisted, Icebox).
    """
    def __init__(self):
        # We default to ApprovedToken, but methods handle others explicitly
        super().__init__(model_class=ApprovedToken)

    def get_approved_tokens(self) -> List[ApprovedToken]:
        """Get all approved tokens."""
        with self.session() as session:
            results = session.execute(select(ApprovedToken).where(ApprovedToken.removed_timestamp.is_(None))).scalars().all()
            session.expunge_all()
            return results
            
    def get_blacklisted_tokens(self) -> List[BlacklistedToken]:
        """Get all blacklisted tokens."""
        with self.session() as session:
            results = session.execute(select(BlacklistedToken).where(BlacklistedToken.removed_timestamp.is_(None))).scalars().all()
            session.expunge_all()
            return results
            
    def get_icebox_tokens(self) -> List[IceboxToken]:
        """Get all icebox tokens."""
        with self.session() as session:
            results = session.execute(select(IceboxToken).where(IceboxToken.removed_timestamp.is_(None))).scalars().all()
            session.expunge_all()
            return results







    def sync_icebox_tokens(self, tokens_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert icebox tokens.
        """
        if not tokens_data:
            return

        from datetime import datetime
        now_val = datetime.now()

        sql = """
            INSERT INTO icebox_tokens (token_symbol, reason, added_timestamp)
            VALUES %s
            ON CONFLICT (token_symbol) DO UPDATE SET
                reason = EXCLUDED.reason,
                removed_timestamp = NULL
        """
        
        values = [(t['token_symbol'], t.get('reason'), now_val) for t in tokens_data]
        self.execute_bulk_values(sql, values)

    def remove_from_icebox(self, token_symbol: str) -> None:
        """Remove a token from the icebox (soft delete)."""
        from datetime import datetime
        with self.session() as session:
            stmt = text("UPDATE icebox_tokens SET removed_timestamp = :now WHERE token_symbol = :symbol AND removed_timestamp IS NULL")
            session.execute(stmt, {'now': datetime.now(), 'symbol': token_symbol})

