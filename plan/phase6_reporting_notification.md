# Phase 6: Reporting & Notification Layer

This phase focuses on managing the daily ledger and sending out notifications based on the allocation results.

## Detailed Tasks:

### 6.1 Implement `manage_ledger.py`
- Create a Python script to:
    - Retain a daily ledger with the number of tokens per stablecoin at the beginning of the day (before optimization run for today) and at the end of the day (before optimization run for tomorrow).
    - Calculate daily NAV = Balance of token N * C price from OHLC every day before running optimization for today.
    - Calculate realized Yield for yesterday ((Total Assets before optimization run for tomorrow) - (Total Assets before optimization run for today))/(Total Assets before optimization run for today).
    - Calculate realized Yield YTD ((((Total Assets before optimization run for tomorrow) - (Total Assets before optimization DAY 0))/(Total Assets before optimization DAY 0)).
    - Record daily token balances and NAV in the `daily_ledger` table.

### 6.2 Implement `post_slack_notification.py`
- Create a Python script to:
    - Format daily allocation recommendations and status updates.
    - Integrate with Slack API to send messages to a designated channel.
    - Include key metrics from the optimization run and ledger updates in the notification.
    - Implement secure handling of Slack API tokens.