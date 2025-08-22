# Phase 9: Admin Web Application (Lowest Priority)

This phase focuses on developing the web-based administrative application for managing configuration parameters and visualizing ledger information.

## Detailed Tasks:

### 9.1 Set up Next.js Frontend Project
- Initialize a new Next.js project.
- Set up basic project structure, routing, and styling (e.g., Tailwind CSS or Material-UI).

### 9.2 Integrate Clerk for Authentication
- Integrate Clerk for secure user authentication and authorization.
- Implement user sign-up, sign-in, and session management.
- Define roles and permissions for different types of users (e.g., admin, viewer).

### 9.3 Develop Configuration Management UI
- Create an interface to view and modify all configurable parameters stored in the `allocation_parameters` table.
- Implement forms and input fields for updating `tvl_limit_percentage`, `max_alloc_percentage`, `conversion_rate`, `min_pools`, `profit_optimization`, etc.
- Develop interfaces for adding/removing approved tokens, blacklisted tokens, and approved protocols.
- Implement management of Icebox token thresholds and manual override capabilities.
- Ensure changes are securely written back to the database.

### 9.4 Develop Ledger Visualization Dashboards
- Create interactive dashboards and charts to display daily ledger information from the `daily_ledger` table.
- Visualize tracking of token balances, daily NAV, realized yield (daily and YTD).
- Implement historical views and trend analysis of financial metrics.
- Consider using charting libraries like Chart.js, Recharts, or Nivo.

### 9.5 Implement Account Performance & Audit Trail Features
- Display actual account balances and transaction history fetched from Etherscan (potentially through a backend API).
- Provide tools for auditing past transactions and verifying investment performance against ledger records.
- Implement a view to display historical optimization runs and the specific parameters (snapshots) used for each run from the `allocation_parameters` table.

### 9.6 Deploy Admin Web Application to GCP Cloud Run
- Containerize the Next.js application using Docker.
- Configure GCP Cloud Run for web applications.
- Define environment variables for database connection and Clerk API keys.
- Implement secure deployment practices.