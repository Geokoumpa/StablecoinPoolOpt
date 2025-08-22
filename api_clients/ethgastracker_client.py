import requests
import json
from datetime import datetime, timezone
from config import ETHGASTRACKER_API_KEY

def get_current_average_gas_price() -> float:
    """
    Fetches the latest average gas price in Gwei from the EthGasTracker API.
    """
    url = "https://www.ethgastracker.com/api/gas/average"
    params = {'apiKey': ETHGASTRACKER_API_KEY}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        raw_response_data = response.json()
        
        latest_data_point = raw_response_data.get('data', {}).get('data', [])
        if latest_data_point:
            latest_gas_price_gwei = latest_data_point[0].get('baseFee')
            return float(latest_gas_price_gwei)
        else:
            print("No data found in EthGasTracker response.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching latest gas price from EthGasTracker: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from EthGasTracker: {e}")
        return None
    except KeyError as e:
        print(f"Could not find expected data in EthGasTracker response: {e}")
        return None
    except IndexError:
        print("EthGasTracker response 'data' array is empty.")
        return None

def get_historical_gas_data_raw_response() -> dict:
    """
    Fetches the full raw historical average gas price data response from the EthGasTracker API.
    Returns the entire JSON response as a dictionary.
    """
    url = "https://www.ethgastracker.com/api/gas/average"
    params = {'apiKey': ETHGASTRACKER_API_KEY}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical raw gas data from EthGasTracker: {e}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response from EthGasTracker: {e}")
        return {}

if __name__ == "__main__":
    if not ETHGASTRACKER_API_KEY:
        print("ETHGASTRACKER_API_KEY environment variable not set in config.py.")
    else:
        current_gas_price = get_current_average_gas_price()
        if current_gas_price is not None:
            print(f"Current Average Gas Price: {current_gas_price} Gwei")
        
        historical_raw_data = get_historical_gas_data_raw_response()
        if historical_raw_data:
            print(f"\nHistorical raw gas data (first 5 data points from 'data' array):")
            historical_points = historical_raw_data.get('data', {}).get('data', [])
            for entry in historical_points[:5]:
                period = entry.get('period')
                base_fee = entry.get('baseFee')
                print(f"Period: {period}, Base Fee: {base_fee} Gwei")