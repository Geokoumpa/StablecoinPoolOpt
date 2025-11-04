import logging
import requests
import json
from datetime import datetime, date, timedelta
from config import FRED_API_KEY

logger = logging.getLogger(__name__)

def get_one_year_ago_date() -> str:
    """
    Calculate the date from one year ago in YYYY-MM-DD format.
    """
    one_year_ago = datetime.now() - timedelta(days=365)
    return one_year_ago.strftime('%Y-%m-%d')

def get_series_data(series_id: str, observation_start: str = None, observation_end: str = None) -> list:
    """
    Fetches data for a specific FRED series.
    """
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': observation_start,
        'observation_end': observation_end
    }
    
    # Remove None values from params
    params = {k: v for k, v in params.items() if v is not None}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get('observations', [])
    except Exception as e:
        logger.error(f"Error fetching FRED data for {series_id}: {e}")
        return []

def get_all_macro_data() -> dict:
    """
    Fetches all configured macroeconomic data from FRED for the last year.
    """
    # Calculate date range for the last year
    observation_start = get_one_year_ago_date()
    observation_end = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Fetching macroeconomic data from {observation_start} to {observation_end}")
    
    all_series = {
        # Daily indicators
        'BAMLH0A0HYM2EY': 'ICE BofA US High Yield Index Effective Yield',
        'DGS1': '1-Year Treasury Bills Yield',
        'SOFR': 'Secured Overnight Financing Rate',
        'SOFR30DAYAVG': 'SOFR 30-Day Average',
        'SOFR90DAYAVG': 'SOFR 90-Day Average',
        'SOFR180DAYAVG': 'SOFR 180-Day Average',
        'SOFRINDEX': 'SOFR Index',
        'RRPONTSYAWARD': 'Reverse Repo Yield (Overnight Reverse Repurchase Award Rate)',
        'NASDAQQGLDI': 'Credit Suisse NASDAQ Gold FLOWS103 Price Index',
        'SP500': 'S&P 500 Index (Daily Close)',
        'NASDAQ100': 'Nasdaq 100 Index',
        'DTWEXBGS': 'Nominal Broad U.S. Dollar Index',
        'FEDFUNDS': 'Federal Funds Effective Rate',
        'DGS1MO': '1-Month Treasury Yield',
        'DGS3MO': '3-Month Treasury Yield',
        'DGS6MO': '6-Month Treasury Yield',
        'DGS2': '2-Year Treasury Yield',
        'DGS10': '10-Year Treasury Yield',
        'DGS30': '30-Year Treasury Yield',
        'T10Y2Y': '10-Year Treasury Minus 2-Year Treasury Spread',
        'T10Y3MM': '10-Year Treasury Minus 3-Month Treasury Spread',
        
        # Monthly indicators
        'M2SL': 'M2 Seasonally Adjusted',
        'WM2NS': 'M2 Not Seasonally Adjusted'
    }
    
    results = {}
    for series_id, description in all_series.items():
        logger.info(f"Fetching data for series {series_id}: {description}")
        data = get_series_data(series_id, observation_start=observation_start, observation_end=observation_end)
        results[series_id] = {
            'description': description,
            'data': data
        }
        logger.info(f"Retrieved {len(data)} observations for {series_id}")
    
    logger.info(f"Completed fetching macroeconomic data for {len(all_series)} series")
    return results

if __name__ == "__main__":
    if not FRED_API_KEY:
        logger.error("FRED_API_KEY environment variable not set in config.py.")
    else:
        # Example of fetching data for a specific series with date filtering
        observation_start = get_one_year_ago_date()
        observation_end = datetime.now().strftime('%Y-%m-%d')
        logger.info(f"Fetching sample data from {observation_start} to {observation_end}")
        
        test_data = get_series_data('DGS1', observation_start=observation_start, observation_end=observation_end)
        if test_data:
            logger.info(f"Fetched {len(test_data)} observations for DGS1")
            for observation in test_data[:3]:  # Show first 3 observations
                logger.info(f"Date: {observation.get('date')}, Value: {observation.get('value')}")