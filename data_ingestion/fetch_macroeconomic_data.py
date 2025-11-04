import logging
from datetime import datetime, date, timedelta
from database.db_utils import get_db_connection
from api_clients.fred_client import get_all_macro_data
from sqlalchemy import text

logger = logging.getLogger(__name__)

def fetch_macroeconomic_data():
    """
    Fetches macroeconomic data from FRED API and stores in database.
    """
    logger.info("Starting macroeconomic data fetch...")
    
    # Get all macroeconomic data
    logger.info("Fetching macroeconomic data from FRED API...")
    macro_data = get_all_macro_data()
    logger.info(f"Retrieved data for {len(macro_data)} series from FRED API")
    
    engine = get_db_connection()
    if not engine:
        logger.error("Failed to establish database connection")
        raise Exception("Database connection failed")
    
    total_records_processed = 0
    total_records_inserted = 0
    total_records_skipped = 0
    
    try:
        with engine.connect() as conn:
            for series_id, series_info in macro_data.items():
                logger.info(f"Processing series: {series_id} - {series_info['description']}")
                
                frequency = 'daily' if series_id in [
                    'BAMLH0A0HYM2EY', 'DGS1', 'SOFR', 'SOFR30DAYAVG', 
                    'SOFR90DAYAVG', 'SOFR180DAYAVG', 'SOFRINDEX', 'RRPONTSYAWARD',
                    'NASDAQQGLDI', 'SP500', 'NASDAQ100', 'DTWEXBGS', 'FEDFUNDS',
                    'DGS1MO', 'DGS3MO', 'DGS6MO', 'DGS2', 'DGS10', 'DGS30',
                    'T10Y2Y', 'T10Y3MM'
                ] else 'monthly'
                
                logger.info(f"Series {series_id} frequency: {frequency}")
                
                series_records_processed = 0
                series_records_inserted = 0
                series_records_skipped = 0
                
                for observation in series_info['data']:
                    date_str = observation.get('date')
                    value = observation.get('value')
                    unit = observation.get('units', '')
                    
                    series_records_processed += 1
                    total_records_processed += 1
                    
                    # Skip records with missing or invalid values
                    if not date_str or not value or value == '.' or value == '':
                        if not date_str:
                            logger.debug(f"Skipping record for {series_id}: missing date")
                        elif not value:
                            logger.debug(f"Skipping record for {series_id} on {date_str}: missing value")
                        elif value == '.':
                            logger.debug(f"Skipping record for {series_id} on {date_str}: placeholder value '.'")
                        else:
                            logger.debug(f"Skipping record for {series_id} on {date_str}: empty value")
                        
                        series_records_skipped += 1
                        total_records_skipped += 1
                        continue
                    
                    try:
                        # Convert value to float to validate it's a proper number
                        numeric_value = float(value)
                        
                        # Upsert to handle duplicates
                        conn.execute(text("""
                            INSERT INTO macroeconomic_data 
                                (series_id, series_name, frequency, date, value, unit, description)
                                VALUES (:series_id, :series_name, :frequency, :date, :value, :unit, :description)
                                ON CONFLICT (series_id, date) 
                                DO UPDATE SET 
                                    value = EXCLUDED.value,
                                    insertion_timestamp = CURRENT_TIMESTAMP
                            """), {
                            'series_id': series_id,
                            'series_name': series_info['description'],
                            'frequency': frequency,
                            'date': date_str,
                            'value': numeric_value,
                            'unit': unit,
                            'description': series_info['description']
                        })
                        
                        series_records_inserted += 1
                        total_records_inserted += 1
                        
                        if series_records_inserted % 100 == 0:
                            logger.info(f"Inserted {series_records_inserted} records for series {series_id}...")
                            
                    except ValueError as ve:
                        logger.warning(f"Skipping record for {series_id} on {date_str}: invalid numeric value '{value}' - {ve}")
                        series_records_skipped += 1
                        total_records_skipped += 1
                        continue
                    except Exception as e:
                        logger.error(f"Error inserting record for {series_id} on {date_str} with value '{value}': {e}")
                        continue
                
                logger.info(f"Completed series {series_id}: processed {series_records_processed}, inserted {series_records_inserted}, skipped {series_records_skipped}")
            
            # Commit the transaction
            conn.commit()
            logger.info("Transaction committed successfully")
            
        logger.info(f"=== MACROECONOMIC DATA FETCH SUMMARY ===")
        logger.info(f"Total series processed: {len(macro_data)}")
        logger.info(f"Total records processed: {total_records_processed}")
        logger.info(f"Total records inserted: {total_records_inserted}")
        logger.info(f"Total records skipped: {total_records_skipped}")
        logger.info(f"Macroeconomic data fetch completed successfully!")
        
    except Exception as e:
        logger.error(f"Error storing macroeconomic data: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    fetch_macroeconomic_data()