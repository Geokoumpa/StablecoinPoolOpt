import os
import json
import requests
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

class SlackNotifier:
    def __init__(self):
        self.webhook_url = config.SLACK_WEBHOOK_URL

    def send_notification(self, message, title="Daily Allocation Report"):
        """
        Sends a formatted message to Slack via webhook.
        """
        if not self.webhook_url:
            logger.warning("Slack webhook URL not configured. Skipping notification.")
            return

        payload = {
            "attachments": [
                {
                    "fallback": title,
                    "color": "#36a64f",
                    "pretext": title,
                    "text": message,
                    "ts": os.path.getmtime(__file__) # Timestamp of the file modification
                }
            ]
        }

        try:
            response = requests.post(
                self.webhook_url, data=json.dumps(payload),
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            logger.info("Slack notification sent successfully!")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending Slack notification: {e}")

if __name__ == "__main__":
    notifier = SlackNotifier()
    # Example usage:
    # You would typically fetch these details from your ledger and optimization results
    daily_metrics = {
        "date": "2025-08-13",
        "total_assets_before_optimization": "$22,500.00",
        "realized_yield_yesterday": "0.5%",
        "realized_yield_ytd": "12.3%",
        "allocation_recommendations": {
            "USDC": "40%",
            "USDT": "30%",
            "DAI": "30%"
        },
        "status": "Optimization run completed successfully."
    }

    message_text = f"""
    *Date:* {daily_metrics['date']}
    *Total Assets (Before Optimization):* {daily_metrics['total_assets_before_optimization']}
    *Realized Yield (Yesterday):* {daily_metrics['realized_yield_yesterday']}
    *Realized Yield (YTD):* {daily_metrics['realized_yield_ytd']}

    *Allocation Recommendations:*
    """
    for token, percentage in daily_metrics['allocation_recommendations'].items():
        message_text += f"    - {token}: {percentage}\n"

    message_text += f"\n*Status:* {daily_metrics['status']}"

    notifier.send_notification(message_text)