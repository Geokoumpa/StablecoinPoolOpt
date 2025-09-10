import logging
import requests
import json
from config import ETHGASTRACKER_API_KEY

logger = logging.getLogger(__name__)

def get_hourly_gas_averages_past_week():
    """
    Fetch hourly average gas prices for the past week from ethgastracker /average endpoint.
    """
    url = "https://www.ethgastracker.com/api/gas/average"
    params = {'apiKey': ETHGASTRACKER_API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    try:
        return response.json()
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from EthGasTracker: {e}")
        return {}

if __name__ == "__main__":
    if not ETHGASTRACKER_API_KEY:
        logger.error("ETHGASTRACKER_API_KEY environment variable not set in config.py.")
    else:
        hourly_averages = get_hourly_gas_averages_past_week()
        if hourly_averages:
            logger.info(f"\nHourly average gas data (first 5 data points):")
            # Assuming the response structure has a 'data' key with a list of hourly averages
            # Adjust this based on the actual API response structure
            for entry in hourly_averages.get('data', [])[:5]:
                logger.debug(entry)