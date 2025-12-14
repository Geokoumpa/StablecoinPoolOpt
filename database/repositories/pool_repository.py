
from typing import List, Optional, Dict, Any
from sqlalchemy import select, update, and_, or_
from database.models.pool import Pool
from database.repositories.base_repository import BaseRepository
from database.repositories.exceptions import RepositoryError

class PoolRepository(BaseRepository[Pool]):
    """
    Repository for Pool entity operations.
    """
    def __init__(self):
        super().__init__(model_class=Pool)

    def bulk_upsert_pools(self, pools_data: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert pools. Update if exists, insert if not.
        """
        if not pools_data:
            return

        # Note: We use "poolMeta" with quotes to handle potential case sensitivity 
        # or special characters if inherited from strict schema
        sql = """
            INSERT INTO pools (
                pool_id, name, chain, protocol, symbol, tvl, apy, 
                last_updated, pool_address, underlying_tokens, 
                underlying_token_addresses, "poolMeta", is_active, currently_filtered_out
            ) VALUES %s
            ON CONFLICT (pool_id) DO UPDATE SET
                name = EXCLUDED.name,
                chain = EXCLUDED.chain,
                protocol = EXCLUDED.protocol,
                symbol = EXCLUDED.symbol,
                tvl = EXCLUDED.tvl,
                apy = EXCLUDED.apy,
                last_updated = EXCLUDED.last_updated,
                pool_address = EXCLUDED.pool_address,
                underlying_tokens = EXCLUDED.underlying_tokens,
                underlying_token_addresses = EXCLUDED.underlying_token_addresses,
                "poolMeta" = EXCLUDED."poolMeta",
                is_active = EXCLUDED.is_active,
                currently_filtered_out = EXCLUDED.currently_filtered_out
        """
        
        values = [
            (
                p['pool_id'], p['name'], p['chain'], p['protocol'], p['symbol'], 
                p.get('tvl'), p.get('apy'), p.get('last_updated'), p.get('pool_address'), 
                p.get('underlying_tokens'), p.get('underlying_token_addresses'), 
                p.get('poolMeta'), p.get('is_active', True), p.get('currently_filtered_out', False)
            )
            for p in pools_data
        ]
        
        self.execute_bulk_values(sql, values)

    def get_active_pools(self) -> List[Pool]:
        """Get all active pools."""
        with self.session() as session:
            stmt = select(Pool).where(Pool.is_active == True)
            return session.execute(stmt).scalars().all()

    def get_pool_by_id(self, pool_id: str) -> Optional[Pool]:
        """Get a pool by its ID."""
        return self.get_by_id(pool_id)
        
    def mark_pools_inactive(self, pool_ids: List[str]) -> None:
        """Mark specified pools as inactive."""
        if not pool_ids:
            return
        
        with self.session() as session:
             stmt = update(Pool).where(Pool.pool_id.in_(pool_ids)).values(is_active=False)
             session.execute(stmt)

    def get_filtered_pools(self, 
                          chain_filter: List[str] = None, 
                          min_tvl: float = None,
                          is_active: bool = True) -> List[Pool]:
        """Get pools matching filter criteria."""
        with self.session() as session:
            stmt = select(Pool)
            conditions = []
            
            if is_active is not None:
                conditions.append(Pool.is_active == is_active)
            
            if chain_filter:
                conditions.append(Pool.chain.in_(chain_filter))
            
            if min_tvl is not None:
                conditions.append(Pool.tvl >= min_tvl)
            
            if conditions:
                stmt = stmt.where(and_(*conditions))
                
            return session.execute(stmt).scalars().all()
