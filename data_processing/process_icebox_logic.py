import logging
from datetime import datetime, timezone
import json
from database.db_utils import get_db_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_icebox_logic():
    """
    Applies Icebox rules to OHLCV data for approved stablecoins.
    Identifies tokens that breach thresholds and adds them to icebox_tokens.
    Identifies tokens that meet recovery criteria and removes them from icebox_tokens.
    """
    logging.info("Starting Icebox logic processing...")
    conn = None
    try:
        conn = get_db_connection()
        from sqlalchemy import text
        with conn.connect() as connection:
            # Fetch latest allocation parameters for Icebox thresholds
            result = connection.execute(text("SELECT other_dynamic_limits, approved_tokens_snapshot FROM allocation_parameters ORDER BY timestamp DESC LIMIT 1;"))
            latest_params = result.fetchone()

            if not latest_params:
                logging.warning("No allocation parameters found. Skipping Icebox logic.")
                return

            other_dynamic_limits = latest_params[0]
            approved_tokens_snapshot = latest_params[1]

            icebox_ohlc_l_threshold_pct = other_dynamic_limits.get('icebox_ohlc_l_threshold_pct', 0.02) if other_dynamic_limits else 0.02
            icebox_ohlc_l_days_threshold = other_dynamic_limits.get('icebox_ohlc_l_days_threshold', 2) if other_dynamic_limits else 2
            icebox_ohlc_c_threshold_pct = other_dynamic_limits.get('icebox_ohlc_c_threshold_pct', 0.01) if other_dynamic_limits else 0.01
            icebox_ohlc_c_days_threshold = other_dynamic_limits.get('icebox_ohlc_c_days_threshold', 1) if other_dynamic_limits else 1
            icebox_recovery_l_days_threshold = other_dynamic_limits.get('icebox_recovery_l_days_threshold', 2) if other_dynamic_limits else 2
            icebox_recovery_c_days_threshold = other_dynamic_limits.get('icebox_recovery_c_days_threshold', 3) if other_dynamic_limits else 3

            if approved_tokens_snapshot:
                logging.debug(f"approved_tokens_snapshot raw: {approved_tokens_snapshot}")
                if isinstance(approved_tokens_snapshot, str):
                    try:
                        approved_tokens_snapshot = json.loads(approved_tokens_snapshot)
                    except Exception:
                        logging.warning(f"Could not decode approved_tokens_snapshot: {approved_tokens_snapshot}")
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
                result = connection.execute(
                    text("""
                        SELECT DISTINCT ON ((raw_json_data->>'symbol'))
                                raw_json_data->>'symbol' AS token_symbol,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'price' AS current_price,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'market_cap' AS market_cap,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'volume_24h' AS volume_24h,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_1h' AS percent_change_1h,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_24h' AS percent_change_24h,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_7d' AS percent_change_7d,
                                (raw_json_data->>'quote')::jsonb->'USD'->>'percent_change_30d' AS percent_change_30d,
                                data_timestamp
                        FROM raw_coinmarketcap_ohlcv
                        WHERE raw_json_data->>'symbol' = ANY(:tokens)
                        ORDER BY (raw_json_data->>'symbol'), data_timestamp DESC;
                    """),
                    {"tokens": list(approved_tokens)}
                )
                ohlcv_data = result.fetchall()

                for row in ohlcv_data:
                    token_symbol = row[0]
                    # Defensive extraction for current_price (always string from SQL)
                    val = row[1]
                    try:
                        current_price = float(val) if val is not None else None
                    except Exception:
                        logging.warning(f"Could not convert current_price for {token_symbol}: {val}")
                        current_price = None
                    timestamp = row[8] # This is data_timestamp from raw_coinmarketcap_ohlcv

                    if token_symbol not in approved_tokens:
                        continue

                    # Check for OHLCV Low Threshold (L)
                    result = connection.execute(
                        text(f"""
                            SELECT MIN((raw_json_data->>'quote')::jsonb->'USD'->>'price')
                            FROM raw_coinmarketcap_ohlcv
                            WHERE raw_json_data->>'symbol' = :token_symbol
                            AND data_timestamp >= NOW() - INTERVAL '{icebox_ohlc_l_days_threshold} days';
                        """),
                        {"token_symbol": token_symbol}
                    )
                    min_price_l_result = result.fetchone()
                    min_price_l = float(min_price_l_result[0]) if min_price_l_result and min_price_l_result[0] else None

                    if current_price and min_price_l and (current_price - min_price_l) / min_price_l < -icebox_ohlc_l_threshold_pct:
                        reason = f"OHLCV Low Threshold (L) breached: price dropped by more than {icebox_ohlc_l_threshold_pct*100}% in {icebox_ohlc_l_days_threshold} days."
                        connection.execute(
                            text("""
                                INSERT INTO icebox_tokens (token_symbol, reason)
                                VALUES (:token_symbol, :reason)
                                ON CONFLICT (token_symbol) DO UPDATE SET
                                    reason = EXCLUDED.reason,
                                    removed_timestamp = NULL;
                            """),
                            {"token_symbol": token_symbol, "reason": reason}
                        )
                        logging.info(f"Token {token_symbol} added to Icebox. Reason: {reason}")
                        continue # Move to next token if already added for this reason

                    # Check for OHLCV Close Threshold (C)
                    result = connection.execute(
                        text(f"""
                            SELECT (raw_json_data->>'quote')::jsonb->'USD'->>'price'
                            FROM raw_coinmarketcap_ohlcv
                            WHERE raw_json_data->>'symbol' = :token_symbol
                            AND data_timestamp >= NOW() - INTERVAL '{icebox_ohlc_c_days_threshold} days'
                            ORDER BY data_timestamp ASC LIMIT 1;
                        """),
                        {"token_symbol": token_symbol}
                    )
                    old_price_c_result = result.fetchone()
                    old_price_c = float(old_price_c_result[0]) if old_price_c_result and old_price_c_result[0] else None

                    if current_price and old_price_c and (current_price - old_price_c) / old_price_c < -icebox_ohlc_c_threshold_pct:
                        reason = f"OHLCV Close Threshold (C) breached: price dropped by more than {icebox_ohlc_c_threshold_pct*100}% in {icebox_ohlc_c_days_threshold} days."
                        connection.execute(
                            text("""
                                INSERT INTO icebox_tokens (token_symbol, reason)
                                VALUES (:token_symbol, :reason)
                                ON CONFLICT (token_symbol) DO UPDATE SET
                                    reason = EXCLUDED.reason,
                                    removed_timestamp = NULL;
                            """),
                            {"token_symbol": token_symbol, "reason": reason}
                        )
                        logging.info(f"Token {token_symbol} added to Icebox. Reason: {reason}")
                        continue

            # 2. Identify tokens to remove from icebox_tokens (recovery criteria)
            result = connection.execute(text("SELECT token_symbol, added_timestamp FROM icebox_tokens WHERE removed_timestamp IS NULL;"))
            icebox_tokens_active = result.fetchall()

            for token_symbol, added_timestamp in icebox_tokens_active:
                # Get current price for this token
                current_price = None
                result = connection.execute(
                    text("""
                        SELECT (raw_json_data->>'quote')::jsonb->'USD'->>'price' AS current_price
                        FROM raw_coinmarketcap_ohlcv
                        WHERE raw_json_data->>'symbol' = :token_symbol
                        ORDER BY data_timestamp DESC LIMIT 1;
                    """),
                    {"token_symbol": token_symbol}
                )
                current_price_result = result.fetchone()
                if current_price_result and current_price_result[0]:
                    try:
                        current_price = float(current_price_result[0])
                    except Exception:
                        logging.warning(f"Could not convert current_price for recovery check {token_symbol}: {current_price_result[0]}")
                        current_price = None

                # Check for Recovery Low Threshold (L)
                result = connection.execute(
                    text(f"""
                        SELECT MIN((raw_json_data->>'quote')::jsonb->'USD'->>'price')
                        FROM raw_coinmarketcap_ohlcv
                        WHERE raw_json_data->>'symbol' = :token_symbol
                        AND data_timestamp >= NOW() - INTERVAL '{icebox_recovery_l_days_threshold} days';
                    """),
                    {"token_symbol": token_symbol}
                )
                min_price_recovery_l_result = result.fetchone()
                min_price_recovery_l = float(min_price_recovery_l_result[0]) if min_price_recovery_l_result and min_price_recovery_l_result[0] else None

                # Check for Recovery Close Threshold (C)
                result = connection.execute(
                    text(f"""
                        SELECT (raw_json_data->>'quote')::jsonb->'USD'->>'price'
                        FROM raw_coinmarketcap_ohlcv
                        WHERE raw_json_data->>'symbol' = :token_symbol
                        AND data_timestamp >= NOW() - INTERVAL '{icebox_recovery_c_days_threshold} days'
                        ORDER BY data_timestamp ASC LIMIT 1;
                    """),
                    {"token_symbol": token_symbol}
                )
                old_price_recovery_c_result = result.fetchone()
                old_price_recovery_c = float(old_price_recovery_c_result[0]) if old_price_recovery_c_result and old_price_recovery_c_result[0] else None

                # If both recovery conditions are met, remove from icebox
                if (current_price and min_price_recovery_l and (current_price - min_price_recovery_l) / min_price_recovery_l >= 0) and \
                   (current_price and old_price_recovery_c and (current_price - old_price_recovery_c) / old_price_recovery_c >= 0):
                    connection.execute(
                        text("""
                            UPDATE icebox_tokens
                            SET removed_timestamp = :removed_timestamp
                            WHERE token_symbol = :token_symbol AND removed_timestamp IS NULL;
                        """),
                        {"removed_timestamp": datetime.now(timezone.utc), "token_symbol": token_symbol}
                    )
                    logging.info(f"Token {token_symbol} removed from Icebox (recovery).")

            connection.commit()
            logging.info("Icebox logic processing completed successfully.")

    except Exception as e:
        logging.error(f"Error during Icebox logic processing: {e}")
    finally:
        if conn:
            conn.dispose()

if __name__ == "__main__":
    process_icebox_logic()