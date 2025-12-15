import logging
import json

from database.repositories.token_repository import TokenRepository
from database.repositories.parameter_repository import ParameterRepository
from database.repositories.raw_data_repository import RawDataRepository

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

def process_icebox_logic():
    """
    Applies Icebox rules to OHLCV data for approved stablecoins.
    Identifies tokens that breach thresholds and adds them to icebox_tokens.
    Identifies tokens that meet recovery criteria and removes them from icebox_tokens.
    """
    logger.info("Starting Icebox logic processing...")
    
    # Initialize repositories
    token_repo = TokenRepository()
    param_repo = ParameterRepository()
    raw_repo = RawDataRepository()

    try:
        # Fetch latest allocation parameters for Icebox thresholds
        latest_params = param_repo.get_latest_parameters()

        if not latest_params:
            logger.warning("No allocation parameters found. Skipping Icebox logic.")
            return

        other_dynamic_limits = latest_params.other_dynamic_limits
        approved_tokens_snapshot = latest_params.approved_tokens_snapshot

        icebox_ohlc_l_threshold_pct = other_dynamic_limits.get('icebox_ohlc_l_threshold_pct', 0.02) if other_dynamic_limits else 0.02
        icebox_ohlc_l_days_threshold = other_dynamic_limits.get('icebox_ohlc_l_days_threshold', 2) if other_dynamic_limits else 2
        icebox_ohlc_c_threshold_pct = other_dynamic_limits.get('icebox_ohlc_c_threshold_pct', 0.01) if other_dynamic_limits else 0.01
        icebox_ohlc_c_days_threshold = other_dynamic_limits.get('icebox_ohlc_c_days_threshold', 1) if other_dynamic_limits else 1
        icebox_recovery_l_days_threshold = other_dynamic_limits.get('icebox_recovery_l_days_threshold', 2) if other_dynamic_limits else 2
        icebox_recovery_c_days_threshold = other_dynamic_limits.get('icebox_recovery_c_days_threshold', 3) if other_dynamic_limits else 3

        if approved_tokens_snapshot:
            logger.debug(f"approved_tokens_snapshot raw: {approved_tokens_snapshot}")
            if isinstance(approved_tokens_snapshot, str):
                try:
                    approved_tokens_snapshot = json.loads(approved_tokens_snapshot)
                except Exception:
                    logger.warning(f"Could not decode approved_tokens_snapshot: {approved_tokens_snapshot}")
                    approved_tokens_snapshot = []
            if isinstance(approved_tokens_snapshot, list):
                approved_tokens = {token['token_symbol'] for token in approved_tokens_snapshot if isinstance(token, dict) and 'token_symbol' in token}
            else:
                approved_tokens = set()
        else:
            approved_tokens = set()

        # 1. Identify tokens to add to icebox_tokens
        # Fetch OHLCV data for approved tokens
        if approved_tokens:
            ohlcv_data = raw_repo.get_latest_token_metrics(list(approved_tokens))

            for row in ohlcv_data:
                token_symbol = row[0]
                val = row[1] # current_price
                try:
                    current_price = float(val) if val is not None else None
                except Exception:
                    logger.warning(f"Could not convert current_price for {token_symbol}: {val}")
                    current_price = None
                
                if token_symbol not in approved_tokens:
                    continue

                # Check for OHLCV Low Threshold (L)
                min_price_l = raw_repo.get_token_min_price(token_symbol, icebox_ohlc_l_days_threshold)

                if current_price and min_price_l and (current_price - min_price_l) / min_price_l < -icebox_ohlc_l_threshold_pct:
                    reason = f"OHLCV Low Threshold (L) breached: price dropped by more than {icebox_ohlc_l_threshold_pct*100}% in {icebox_ohlc_l_days_threshold} days."
                    
                    token_repo.sync_icebox_tokens([{
                        'token_symbol': token_symbol,
                        'reason': reason
                    }])
                    
                    logger.info(f"Token {token_symbol} added to Icebox. Reason: {reason}")
                    continue # Move to next token if already added for this reason

                # Check for OHLCV Close Threshold (C)
                old_price_c = raw_repo.get_token_price_at_interval(token_symbol, icebox_ohlc_c_days_threshold)

                if current_price and old_price_c and (current_price - old_price_c) / old_price_c < -icebox_ohlc_c_threshold_pct:
                    reason = f"OHLCV Close Threshold (C) breached: price dropped by more than {icebox_ohlc_c_threshold_pct*100}% in {icebox_ohlc_c_days_threshold} days."
                    
                    token_repo.sync_icebox_tokens([{
                        'token_symbol': token_symbol,
                        'reason': reason
                    }])
                    
                    logger.info(f"Token {token_symbol} added to Icebox. Reason: {reason}")
                    continue

        # 2. Identify tokens to remove from icebox_tokens (recovery criteria)
        icebox_tokens_active = token_repo.get_icebox_tokens()

        for icebox_token in icebox_tokens_active:
            token_symbol = icebox_token.token_symbol
            
            # Get current price
            ohlcv = raw_repo.get_latest_token_metrics([token_symbol])
            current_price = None
            if ohlcv:
                try:
                    current_price = float(ohlcv[0][1]) if ohlcv[0][1] else None
                except Exception:
                    pass

            # Check for Recovery Low Threshold (L)
            min_price_recovery_l = raw_repo.get_token_min_price(token_symbol, icebox_recovery_l_days_threshold)

            # Check for Recovery Close Threshold (C)
            old_price_recovery_c = raw_repo.get_token_price_at_interval(token_symbol, icebox_recovery_c_days_threshold)

            # If both recovery conditions are met, remove from icebox
            if (current_price and min_price_recovery_l and (current_price - min_price_recovery_l) / min_price_recovery_l >= 0) and \
               (current_price and old_price_recovery_c and (current_price - old_price_recovery_c) / old_price_recovery_c >= 0):
                
                token_repo.remove_from_icebox(token_symbol)
                logger.info(f"Token {token_symbol} removed from Icebox (recovery).")

        logger.info("Icebox logic processing completed successfully.")

    except Exception as e:
        logger.error(f"Error during Icebox logic processing: {e}")
        raise

if __name__ == "__main__":
    process_icebox_logic()