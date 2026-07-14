# daily_report.py
# Generates a daily WhatsApp summary of misinformation trends
# Schedule to run at 8 AM every day via cron or Railway scheduler

import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import os
from twilio.rest import Client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USAGE_LOG = os.path.join(BASE_DIR, 'usage_log.csv')
ANALYSIS_LOG = os.path.join(BASE_DIR, 'analysis_log.csv')

DIVIDER = "━━━━━━━━━━━━━━━━━━━━"

def generate_daily_report() -> str:
    """
    Reads usage_log.csv and analysis_log.csv,
    generates a concise daily intelligence summary.
    """
    today     = datetime.now().date()
    yesterday = today - timedelta(days=1)

    # Load logs
    try:
        usage = pd.read_csv(USAGE_LOG)
        usage['date'] = pd.to_datetime(usage['timestamp']).dt.date
        today_data = usage[usage['date'] == yesterday]
    except:
        today_data = pd.DataFrame()

    total_msgs   = len(today_data)
    unique_users = today_data['user_hash'].nunique() if not today_data.empty else 0

    # Load detailed analysis log
    try:
        analyses = pd.read_csv(ANALYSIS_LOG)
        analyses['date'] = pd.to_datetime(analyses['timestamp']).dt.date
        today_analyses = analyses[analyses['date'] == yesterday]
        fake_count = (today_analyses['verdict'] == 'FAKE').sum()
        real_count = (today_analyses['verdict'] == 'REAL').sum()
        hindi_count = (today_analyses['language'] == 'hindi').sum()
        telugu_count = (today_analyses['language'] == 'telugu').sum()
        avg_conf = today_analyses['confidence'].mean()
        if pd.isna(avg_conf):
            avg_conf = 0.0
    except:
        fake_count = real_count = hindi_count = telugu_count = 0
        avg_conf = 0

    date_str = yesterday.strftime('%d %b %Y')
    total_checks = fake_count + real_count

    report = f"""🇮🇳 *VerifyAI Daily Report — {date_str}*
{DIVIDER}
📊 *Yesterday's activity:*
  Total checks   : {total_checks}
  Unique users   : {unique_users}
  Fake detected  : {fake_count} ({fake_count/max(total_checks,1)*100:.0f}%)
  Real verified  : {real_count} ({real_count/max(total_checks,1)*100:.0f}%)
  Avg confidence : {avg_conf:.1f}%

🌐 *By language:*
  English : {total_checks - hindi_count - telugu_count}
  Hindi   : {hindi_count}
  Telugu  : {telugu_count}

⚠️ *Misinformation rate: {fake_count/max(total_checks,1)*100:.0f}%*
{DIVIDER}
_VerifyAI National Misinformation Intelligence_
_Report generated automatically at 8:00 AM_"""

    return report


def send_daily_report(to_numbers: list):
    """Send daily report to a list of WhatsApp numbers."""
    client = Client(
        os.getenv('TWILIO_ACCOUNT_SID'),
        os.getenv('TWILIO_AUTH_TOKEN')
    )
    report = generate_daily_report()

    for number in to_numbers:
        client.messages.create(
            body = report,
            from_= os.getenv('TWILIO_WHATSAPP_FROM'),
            to   = f"whatsapp:{number}"
        )
        print(f"Report sent to {number}")


# Run this manually or schedule it:
# python daily_report.py

if __name__ == '__main__':
    report = generate_daily_report()
    print(report)
