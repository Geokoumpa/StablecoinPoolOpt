import logging
import requests
import json
from datetime import datetime, timezone
from config import COINMARKETCAP_API_KEY

logger = logging.getLogger(__name__)

def get_latest_eth_price() -> float:
    """
    Fetches the latest ETH price from the CoinMarketCap API.
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }
    params = {
        'symbol': 'ETH',
        'convert': 'USD'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        eth_data = data['data']['ETH']
        latest_price = eth_data['quote']['USD']['price']
        return latest_price
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching latest ETH price from CoinMarketCap: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from CoinMarketCap: {e}")
        return None
    except KeyError as e:
        logger.error(f"Could not find expected data in CoinMarketCap response: {e}")
        return None

def get_latest_btc_price() -> float:
    """
    Fetches the latest BTC price from the CoinMarketCap API.
    """
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }
    params = {
        'symbol': 'BTC',
        'convert': 'USD'
    }

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        btc_data = data['data']['BTC']
        latest_price = btc_data['quote']['USD']['price']
        return latest_price
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching latest BTC price from CoinMarketCap: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response from CoinMarketCap: {e}")
        return None
    except KeyError as e:
        logger.error(f"Could not find expected data in CoinMarketCap response: {e}")
        return None

def get_historical_ohlcv_data(symbol: str, count: int = 30) -> list:
    """
    Fetches historical OHLCV data for a given symbol from the CoinMarketCap API.
    Implements exponential backoff on rate limiting (HTTP 429).
    """
    import time
    from config import COINMARKETCAP_API_KEY
    
    if not COINMARKETCAP_API_KEY:
        logging.getLogger(__name__).error("COINMARKETCAP_API_KEY not available from config.")
        return []

    url = "https://pro-api.coinmarketcap.com/v2/cryptocurrency/ohlcv/historical"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_API_KEY,
    }
    params = {
        'symbol': symbol,
        'time_period': 'daily',
        'convert': 'USD',
        'count': count,
        'skip_invalid': True
    }

    max_retries = 5
    backoff = 2

    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 429:
                logger.warning(f"Rate limited by CoinMarketCap for {symbol}. Attempt {attempt+1}/{max_retries}. Retrying in {backoff} seconds.")
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            raw_data = response.json()
            data = raw_data.get('data', {})
            # Case-insensitive search for the symbol in the response data
            for key in data:
                if key.lower() == symbol.lower():
                    quotes_data = data[key]
                    if quotes_data and isinstance(quotes_data, list) and 'quotes' in quotes_data[0]:
                        return quotes_data[0]['quotes']
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching historical OHLCV for {symbol}: {e}")
            time.sleep(backoff)
            backoff *= 2
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON response for {symbol}: {e}")
            return []
        except KeyError as e:
            logger.error(f"Could not find expected data in CoinMarketCap historical response: {e}")
            return []
    logger.error(f"Failed to fetch historical OHLCV for {symbol} after {max_retries} attempts.")
    return []

if __name__ == "__main__":
    if not COINMARKETCAP_API_KEY:
        logger.error("COINMARKETCAP_API_KEY environment variable not set in config.py.")
    else:
        latest_eth_price = get_latest_eth_price()
        if latest_eth_price is not None:
            logger.info(f"Latest ETH Price: {latest_eth_price} USD")
        
        # Example of fetching historical data
        eth_historical_data = get_historical_ohlcv_data(symbol='ETH', count=5)
        if eth_historical_data:
            logger.info(f"\nLast 5 days of ETH historical data:")
            for entry in eth_historical_data:
                timestamp = entry.get('quote', {}).get('USD', {}).get('timestamp')
                close_price = entry.get('quote', {}).get('USD', {}).get('close')
                logger.info(f"Date: {timestamp}, Close: {close_price}")