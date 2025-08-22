# DefiYieldOpt

DefiYieldOpt is a comprehensive pipeline for optimizing yield-generating opportunities in decentralized finance (DeFi). It automates data ingestion, processing, forecasting, and asset allocation to maximize returns from stablecoin pools.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Pipeline](#running-the-pipeline)
- [Project Structure](#project-structure)
- [Database Migrations](#database-migrations)

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- An Infura account and API key

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/DefiYieldOpt.git
    cd DefiYieldOpt
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Create a `.env` file:**

    Duplicate the `.env.example` file and rename it to `.env`.

    ```bash
    cp .env.example .env
    ```

2.  **Update the `.env` file:**

    Update the `.env` file with your actual credentials and settings.

    ```
    # Database Configuration
    DB_USER=your_db_user
    DB_PASSWORD=your_db_password
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=your_db_name

    # API Keys
    COINMARKETCAP_API_KEY=your_coinmarketcap_api_key
    ETHERSCAN_API_KEY=your_etherscan_api_key
    INFURA_API_KEY=your_infura_api_key

    # Wallet Address
    MAIN_ASSET_HOLDING_ADDRESS=your_wallet_address

    # Slack Notifications
    SLACK_WEBHOOK_URL=your_slack_webhook_url
    ```

## Running the Pipeline

1.  **Start the services:**

    Use Docker Compose to start the required services.

    ```bash
    docker compose up -d
    ```

2.  **Run the main pipeline:**

    Execute the `main_pipeline.py` script to run the entire pipeline.

    ```bash
    python main_pipeline.py
    ```

    You can also run specific phases of the pipeline using the `--phases` argument. For example, to run only Phase 1 and Phase 3:

    ```bash
    python main_pipeline.py --phases phase1 phase3
    ```

## Project Structure

The project is organized into the following directories:

-   `asset_allocation/`: Scripts for optimizing asset allocations.
-   `data_ingestion/`: Scripts for fetching data from various sources.
-   `data_processing/`: Scripts for cleaning and transforming raw data.
-   `database/`: Database schema, migrations, and utility scripts.
-   `forecasting/`: Scripts for forecasting pool performance and gas fees.
-   `reporting_notification/`: Scripts for generating reports and sending notifications.

## Database Migrations

Database migrations are managed using a simple migration script. To apply the latest migrations, the pipeline calls the `apply_migrations` function from `database/db_utils.py` at the start of its run.
