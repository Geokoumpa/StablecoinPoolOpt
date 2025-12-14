
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
            return session.execute(select(ApprovedToken).where(ApprovedToken.removed_timestamp.is_(None))).scalars().all()
            
    def get_blacklisted_tokens(self) -> List[BlacklistedToken]:
        """Get all blacklisted tokens."""
        with self.session() as session:
            return session.execute(select(BlacklistedToken).where(BlacklistedToken.removed_timestamp.is_(None))).scalars().all()
            
    def get_icebox_tokens(self) -> List[IceboxToken]:
        """Get all icebox tokens."""
        with self.session() as session:
            return session.execute(select(IceboxToken).where(IceboxToken.removed_timestamp.is_(None))).scalars().all()

    def sync_approved_tokens(self, tokens_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert approved tokens.
        """
        if not tokens_data:
            return

        sql = """
            INSERT INTO approved_tokens (token_symbol, token_address, added_timestamp)
            VALUES %s
            ON CONFLICT (token_symbol) DO UPDATE SET
                token_address = EXCLUDED.token_address,
                removed_timestamp = NULL
        """
        
        values = [
            (t['token_symbol'], t.get('token_address'), datetime.now())
            for t in tokens_data
        ]
        
        # Need datetime for now() in python to pass as value, or let postgres handle it if we don't pass it?
        # execute_values expects values for placeholders.
        # Construct values list carefully.
        
        # Actually easier to let postgres default added_timestamp if missing, but we are providing it in VALUES.
        # If we want to use server time in Python variable:
        from datetime import datetime
        now_val = datetime.now()
        
        values = [
            (t['token_symbol'], t.get('token_address'), now_val)
            for t in tokens_data
        ]
        
        self.execute_bulk_values(sql, values)

    def sync_blacklisted_tokens(self, tokens_symbols: List[str]) -> None:
        """
        Bulk upsert blacklisted tokens.
        """
        if not tokens_symbols:
            return

        from datetime import datetime
        now_val = datetime.now()

        sql = """
            INSERT INTO blacklisted_tokens (token_symbol, added_timestamp)
            VALUES %s
            ON CONFLICT (token_symbol) DO UPDATE SET
                removed_timestamp = NULL
        """
        
        values = [(s, now_val) for s in tokens_symbols]
        self.execute_bulk_values(sql, values)

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
