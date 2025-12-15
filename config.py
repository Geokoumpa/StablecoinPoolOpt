import os
from dotenv import load_dotenv

# Load .env file only in development environment
if os.getenv("ENVIRONMENT", "development") == "development":
    load_dotenv()  # Load environment variables from .env file

# Database Configuration
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")

# API Keys
COINMARKETCAP_API_KEY = os.getenv("COINMARKETCAP_API_KEY")
ETHPLORER_API_KEY = os.getenv("ETHPLORER_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
ETHGASTRACKER_API_KEY = os.getenv("ETHGASTRACKER_API_KEY")

# Other Configurations
MAIN_ASSET_HOLDING_ADDRESS = os.getenv("MAIN_ASSET_HOLDING_ADDRESS")
COLD_WALLET_ADDRESS = os.getenv("COLD_WALLET_ADDRESS")


SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")