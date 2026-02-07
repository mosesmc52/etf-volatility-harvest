import os
from datetime import datetime

from alpaca_adapter import AlpacaAPI
from dotenv import find_dotenv, load_dotenv
from helpers import (
    getenv_float,
    getenv_int,
    print_orders_table,
    run_single_iteration,
    str2bool,
)
from log import log
from SES import AmazonSES

load_dotenv(find_dotenv())


EQUITY_FRACTION = getenv_float("EQUITY_FRACTION", 1)

LIQUIDATION_SYMBOLS_TO_IGNORE = None


FORCED_REBALANCE = str2bool(os.getenv("FORCED_REBALANCE", False))
LIVE_TRADE = str2bool(os.getenv("LIVE_TRADE", False))
log(f"Running in {'LIVE' if LIVE_TRADE else 'TEST'} mode", "info")

alpaca_key = os.getenv("ALPACA_KEY_ID")
alpaca_secret = os.getenv("ALPACA_SECRET_KEY")
base_url = (os.getenv("ALPACA_BASE_URL") or "").lower()

# Simple heuristic: treat "paper" URLs as paper trading


TICKERS = ("JEPQ", "SGOV", "SGOL")
HIGH_VOL_ANCHOR = "SGOL"
BASELINE_WHEN_INACTIVE = (0.0, 1.0, 0.0)


is_paper = ("paper" in base_url) or str2bool(os.getenv("ALPACA_PAPER", True))
DRIFT_ONLY_WHEN_ACTIVE = str2bool(os.getenv("DRIFT_ONLY_WHEN_ACTIVE", True))

FORCED_REBALANCED = str2bool(os.getenv("FORCED_REBALANCED", False))
IS_REBALANCE_DATE = os.getenv("IS_REBALANCE_DATE", "W-FRI")

EQUITY_FRACTION = getenv_int(os.getenv("EQUITY_FRACTION"), 1)
VOL_LOOKBACK = getenv_float(os.getenv("VOL_LOOKBACK"), 40)
TARGET_VOL = getenv_float(os.getenv("TARGET_VOL"), 0.08)
LEVERAGE_CAP = getenv_float(os.getenv("LEVERAGE_CAP"), 2.0)

HIGH_VOL_WINDOW = getenv_float(os.getenv("HIGH_VOL_WINDOW"), 63)
HIGH_VOL_THESHOLD = getenv_float(os.getenv("HIGH_VOL_THESHOLD"), 0.22)

DRIFT_THRESHOLD = getenv_float(os.getenv("DRIFT_THRESHOLD"), 0.08)


api = AlpacaAPI.from_env(
    api_key=alpaca_key,
    secret_key=alpaca_secret,
    paper=is_paper,
)

account = api.get_account()
portfolio_value = round(float(account.equity), 3)

portfolio = run_single_iteration(
    api,
    tickers=TICKERS,
    equity_fraction=EQUITY_FRACTION,
    is_live_trade=LIVE_TRADE,
    vol_lookback=VOL_LOOKBACK,
    target_vol=TARGET_VOL,
    leverage_cap=LEVERAGE_CAP,
    high_vol_anchor=HIGH_VOL_ANCHOR,
    high_vol_window=HIGH_VOL_WINDOW,
    high_vol_threshold=HIGH_VOL_THESHOLD,
    baseline_when_inactive=BASELINE_WHEN_INACTIVE,
    drift_threshold=DRIFT_THRESHOLD,
    drift_only_when_active=DRIFT_ONLY_WHEN_ACTIVE,
    forced_rebalance=FORCED_REBALANCED,
    is_rebalance_date=IS_REBALANCE_DATE,
)

# # Email Positions
EMAIL_POSITIONS = str2bool(os.getenv("EMAIL_POSITIONS", False))


message_body_html = f"Portfolio Value: {portfolio_value}<br>"
message_body_plain = f"Portfolio Value: {portfolio_value}\n"

out = print_orders_table(portfolio)

message_body_html += "<pre>" + out.replace("\n", "<br>") + "</pre>"
message_body_plain += out
print("---------------------------------------------------\n")
print(message_body_plain)

if EMAIL_POSITIONS:
    TO_ADDRESSES = [
        a.strip() for a in os.getenv("TO_ADDRESSES", "").split(",") if a.strip()
    ]
    FROM_ADDRESS = os.getenv("FROM_ADDRESS", "")

    ses = AmazonSES(
        region=os.environ.get("AWS_SES_REGION_NAME"),
        access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
        secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
        from_address=FROM_ADDRESS,
    )

    status = "Live" if LIVE_TRADE else "Test"
    today = datetime.utcnow().strftime("%Y-%m-%d")

    subject = f"Monthly Trend Algo Report - {status} - {today}"

    for to_address in TO_ADDRESSES:
        ses.send_html_email(
            to_address=to_address, subject=subject, content=message_body_html
        )

print("---------------------------------------------------\n")
print(message_body_plain)
