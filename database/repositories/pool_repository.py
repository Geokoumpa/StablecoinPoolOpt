
from typing import List, Optional, Dict, Any, Tuple
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
                underlying_token_addresses, poolMeta, is_active, currently_filtered_out
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
                poolMeta = EXCLUDED.poolMeta,
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
            results = session.execute(stmt).scalars().all()
            session.expunge_all()
            return results

    def get_all_pool_ids(self) -> List[str]:
        """Get all pool IDs."""
        with self.session() as session:
            stmt = select(Pool.pool_id)
            return session.execute(stmt).scalars().all()



        
    def mark_pools_inactive(self, pool_ids: List[str]) -> None:
        """Mark specified pools as inactive."""
        if not pool_ids:
            return
        
        with self.session() as session:
             stmt = update(Pool).where(Pool.pool_id.in_(pool_ids)).values(is_active=False)
             session.execute(stmt)



    def reset_all_currently_filtered_out(self) -> None:
        """Reset currently_filtered_out flag for all pools."""
        with self.session() as session:
            stmt = update(Pool).values(currently_filtered_out=False)
            session.execute(stmt)

    def bulk_update_currently_filtered_out(self, pool_ids: List[str]) -> None:
        """Set currently_filtered_out to True for specified pools."""
        if not pool_ids:
            return
        with self.session() as session:
             stmt = update(Pool).where(Pool.pool_id.in_(pool_ids)).values(currently_filtered_out=True)
             session.execute(stmt)

    def bulk_update_underlying_tokens(self, updates: List[Tuple[str, List[str]]]) -> None:
        """
        Bulk update underlying tokens.
        updates: List of (pool_id, tokens)
        """
        if not updates:
            return
        
        # Use template to ensure array type casting if needed, 
        # though psycopg2 lists often map array correctly. 
        # Explicit casting is safer.
        sql = """
            UPDATE pools
            SET underlying_tokens = data.tokens 
            FROM (VALUES %s) AS data (pool_id, tokens) 
            WHERE pools.pool_id = data.pool_id
        """
        self.execute_bulk_values(sql, updates, template="(%s, %s::text[])")

    def bulk_update_underlying_token_addresses(self, updates: List[Tuple[str, List[str]]]) -> None:
        """
        Bulk update underlying token addresses.
        updates: List of (pool_id, addresses)
        """
        if not updates:
            return
        
        sql = """
            UPDATE pools
            SET underlying_token_addresses = data.addresses 
            FROM (VALUES %s) AS data (pool_id, addresses) 
            WHERE pools.pool_id = data.pool_id
        """
        self.execute_bulk_values(sql, updates, template="(%s, %s::text[])")

    def get_approved_protocols(self) -> List[str]:
        """Get list of approved protocol names."""
        from database.models.protocol import ApprovedProtocol
        with self.session() as session:
             stmt = select(ApprovedProtocol.protocol_name).where(ApprovedProtocol.removed_timestamp.is_(None))
             return session.execute(stmt).scalars().all()

    def bulk_update_addresses(self, updates: Dict[str, str]) -> int:
        """
        Bulk update pool addresses.
        updates: Dict of pool_id -> address.
        """
        if not updates:
             return 0
             
        # Filter for valid addresses
        update_list = [(pid, addr) for pid, addr in updates.items() if addr]
        
        if not update_list:
            return 0

        sql = """
            UPDATE pools
            SET pool_address = data.address
            FROM (VALUES %s) AS data (pool_id, address)
            WHERE pools.pool_id = data.pool_id
        """
        self.execute_bulk_values(sql, update_list)
        return len(update_list)


