#!/usr/bin/env python3
"""
One-time script to import historical gas price data from AvgGasPrice.csv into gas_fees_daily table.
This script uses a similar mechanism to database migrations to ensure it only runs once.
"""

import csv
import os
import sys
from datetime import datetime
from decimal import Decimal, InvalidOperation

# Add parent directory to path to import database utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_utils import get_db_connection
from sqlalchemy import text

def wei_to_gwei(wei_value):
    """
    Convert wei to gwei (1 gwei = 10^9 wei)
    
    Args:
        wei_value (str): Gas price in wei as string
        
    Returns:
        Decimal: Gas price in gwei, rounded to 2 decimal places
    """
    try:
        if not wei_value or wei_value == "0":
            return Decimal("0.00")
        
        wei_decimal = Decimal(wei_value)
        gwei_decimal = wei_decimal / Decimal("1000000000")  # 10^9
        return gwei_decimal.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as e:
        print(f"Error converting wei to gwei for value '{wei_value}': {e}")
        return Decimal("0.00")

def parse_date(date_str):
    """
    Parse date from MM/DD/YYYY format to YYYY-MM-DD format
    
    Args:
        date_str (str): Date in MM/DD/YYYY format
        
    Returns:
        str: Date in YYYY-MM-DD format, or None if parsing fails
    """
    try:
        parsed_date = datetime.strptime(date_str, "%m/%d/%Y")
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

def is_import_already_completed(engine):
    """
    Check if the historical gas data import has already been completed.
    
    Args:
        engine: SQLAlchemy engine
        
    Returns:
        bool: True if import was already completed, False otherwise
    """
    try:
        with engine.connect() as conn:
            # Create import tracking table if it doesn't exist
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS data_imports (
                    id SERIAL PRIMARY KEY,
                    import_name VARCHAR(255) UNIQUE NOT NULL,
                    completed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    record_count INTEGER DEFAULT 0
                );
            """))
            conn.commit()
            
            # Check if this import has already been completed
            result = conn.execute(
                text("SELECT record_count FROM data_imports WHERE import_name = :import_name"),
                {"import_name": "historical_gas_data_csv"}
            )
            row = result.fetchone()
            if row:
                print(f"Historical gas data import already completed with {row[0]} records.")
                return True
            return False
    except Exception as e:
        print(f"Error checking import status: {e}")
        return False

def mark_import_completed(engine, record_count):
    """
    Mark the import as completed in the tracking table.
    
    Args:
        engine: SQLAlchemy engine
        record_count (int): Number of records imported
    """
    try:
        with engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO data_imports (import_name, record_count) 
                    VALUES (:import_name, :record_count)
                """),
                {"import_name": "historical_gas_data_csv", "record_count": record_count}
            )
            conn.commit()
            print(f"Import marked as completed with {record_count} records.")
    except Exception as e:
        print(f"Error marking import as completed: {e}")

def import_gas_data_from_csv(csv_filepath):
    """
    Import historical gas data from CSV file into gas_fees_daily table.
    
    Args:
        csv_filepath (str): Path to the AvgGasPrice.csv file
        
    Returns:
        bool: True if import was successful, False otherwise
    """
    if not os.path.exists(csv_filepath):
        print(f"CSV file not found: {csv_filepath}")
        return False
    
    engine = get_db_connection()
    if not engine:
        print("Could not establish database connection.")
        return False
    
    try:
        # Check if import has already been completed
        if is_import_already_completed(engine):
            return True
        
        print(f"Starting import from {csv_filepath}")
        
        imported_count = 0
        skipped_count = 0
        error_count = 0
        
        with engine.begin() as conn:
            # Open and read the CSV file
            with open(csv_filepath, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 because row 1 is header
                    try:
                        # Extract data from CSV row
                        date_str = row.get('Date(UTC)', '').strip()
                        wei_value = row.get('Value (Wei)', '').strip()
                        
                        if not date_str or not wei_value:
                            print(f"Row {row_num}: Missing date or value, skipping")
                            skipped_count += 1
                            continue
                        
                        # Parse and convert data
                        formatted_date = parse_date(date_str)
                        if not formatted_date:
                            print(f"Row {row_num}: Could not parse date '{date_str}', skipping")
                            skipped_count += 1
                            continue
                        
                        gwei_value = wei_to_gwei(wei_value)
                        
                        # Insert into database (using ON CONFLICT to handle duplicates)
                        conn.execute(
                            text("""
                                INSERT INTO gas_fees_daily (date, actual_avg_gas_gwei) 
                                VALUES (:date, :actual_avg_gas_gwei)
                                ON CONFLICT (date) DO UPDATE SET 
                                    actual_avg_gas_gwei = EXCLUDED.actual_avg_gas_gwei
                            """),
                            {
                                "date": formatted_date,
                                "actual_avg_gas_gwei": gwei_value
                            }
                        )
                        
                        imported_count += 1
                        
                        # Progress indicator
                        if imported_count % 100 == 0:
                            print(f"Imported {imported_count} records...")
                            
                    except Exception as e:
                        print(f"Row {row_num}: Error processing row - {e}")
                        error_count += 1
                        continue
        
        # Mark import as completed
        mark_import_completed(engine, imported_count)
        
        print(f"\nImport completed successfully!")
        print(f"Records imported: {imported_count}")
        print(f"Records skipped: {skipped_count}")
        print(f"Errors encountered: {error_count}")
        
        return True
        
    except Exception as e:
        print(f"Error during import: {e}")
        return False
    finally:
        engine.dispose()

def main():
    """
    Main function to run the historical gas data import.
    """
    print("=" * 60)
    print("Historical Gas Data Import Script")
    print("=" * 60)
    
    # Path to the CSV file (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_filepath = os.path.join(project_root, "AvgGasPrice.csv")
    
    print(f"Looking for CSV file at: {csv_filepath}")
    
    success = import_gas_data_from_csv(csv_filepath)
    
    if success:
        print("\n✅ Historical gas data import completed successfully!")
    else:
        print("\n❌ Historical gas data import failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()