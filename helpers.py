# === single_iteration_trader.py ===
from __future__ import annotations

import io
import json
import math
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, TextIO

import boto3
import numpy as np
import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.data.timeframe import TimeFrame

# ---------- Config (match your backtest) ----------


PERIODS_PER_YEAR = 252
BUY, SELL = "buy", "sell"


def str2bool(value):
    valid = {
        "true": True,
        "t": True,
        "1": True,
        "on": True,
        "false": False,
        "f": False,
        "0": False,
    }

    if isinstance(value, bool):
        return value

    lower_value = value.lower()
    if lower_value in valid:
        return valid[lower_value]
    else:
        raise ValueError('invalid literal for boolean: "%s"' % value)


def getenv_float(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def getenv_int(name: str, default: float) -> float:
    """
    Read an environment variable as a float.

    - Returns `default` if the variable is missing
    - Returns `default` if conversion fails
    """
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_current_positions(api) -> Dict[str, int]:
    """
    Returns {symbol: signed_qty} for all open positions.
    Alpaca qty is typically positive for long positions.
    """
    out = {}
    try:
        positions = api.list_positions()
        for p in positions:
            sym = getattr(p, "symbol", None)
            qty = int(float(getattr(p, "qty", 0)))
            if sym and qty != 0:
                out[sym] = qty
    except Exception as e:
        print(f"[WARN] Unable to list positions: {e}")
    return out


def process_position(api, security, qty, is_live_trade=False):
    try:
        position = api.get_position(security)
        current_qty = int(float(getattr(position, "qty", 0)))
    except APIError:
        current_qty = 0

    diff = qty - current_qty

    if is_live_trade:
        if diff > 0:
            api.submit_order(
                symbol=security,
                time_in_force="day",
                side=BUY,
                type="market",
                qty=abs(diff),
            )
        elif diff < 0:
            api.submit_order(
                symbol=security,
                time_in_force="day",
                side=SELL,
                type="market",
                qty=abs(diff),
            )

    action = "NOOP" if diff == 0 else (BUY if diff > 0 else SELL)
    return action, current_qty, qty, diff


def price_history(
    api,
    ticker: str,
    start_date,
    end_date,
    feed: str = "iex",
):
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        return api.get_bars(
            ticker,
            TimeFrame.Day,
            start_str,
            end_str,
            adjustment="all",
            feed=feed,
        )
    except APIError as e:
        msg = str(e).lower()
        # Preserve your SIP subscription fallback
        if "sip" in msg and "subscription" in msg:
            return api.get_bars(
                ticker,
                TimeFrame.Day,
                start_str,
                end_str,
                adjustment="all",
                feed="iex",
            )
        raise


def _bars_to_series_close(bars, symbol: str | None = None) -> pd.Series:
    """
    Convert Alpaca bars to a daily close Series.

    alpaca-py BarSet.df is typically MultiIndex: (symbol, timestamp).
    """
    if bars is None:
        return pd.Series(dtype=float)

    if hasattr(bars, "df"):
        df = bars.df.copy()
    else:
        try:
            df = pd.DataFrame(bars)
        except Exception:
            return pd.Series(dtype=float)

    if df.empty:
        return pd.Series(dtype=float)

    # If MultiIndex (symbol, timestamp), slice to one symbol
    if isinstance(df.index, pd.MultiIndex):
        if symbol is None:
            # if caller didn't pass symbol, try to infer the first
            try:
                symbol = df.index.get_level_values(0)[0]
            except Exception:
                return pd.Series(dtype=float)
        try:
            df = df.xs(symbol, level=0)
        except Exception:
            return pd.Series(dtype=float)

    if "close" not in df.columns:
        return pd.Series(dtype=float)

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        elif "t" in df.columns:
            df.index = pd.to_datetime(df["t"], utc=True, errors="coerce")
        else:
            return pd.Series(dtype=float)

    dates = df.index.tz_convert(None).normalize()
    ser = pd.Series(df["close"].astype(float).values, index=dates)
    return ser.groupby(level=0).last().dropna().sort_index()


def _compute_target_weights_and_leverage(
    px: pd.DataFrame,
    *,
    tickers: tuple[str, str, str],
    vol_lookback: int = 60,
    target_vol: float = 0.10,
    leverage_cap: float = 2.0,
    high_vol_anchor: str = "SPY",
    high_vol_window: int = 63,
    high_vol_threshold: float = 0.20,
    baseline_when_inactive: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> dict:
    """
    Computes *latest* target weights + leverage using 1-day lag logic.
    """
    cols = list(tickers)
    px = px[cols].dropna()

    if len(px) < max(vol_lookback, high_vol_window) + 10:
        raise ValueError("Not enough history for requested windows.")

    r = px.pct_change().dropna()
    idx = r.index

    # -------------------------------------------------
    # High-vol regime signal (anchor realized vol)
    # -------------------------------------------------
    rv = r[high_vol_anchor].rolling(high_vol_window).std(ddof=1) * np.sqrt(
        PERIODS_PER_YEAR
    )
    active = (rv >= high_vol_threshold).astype(float).reindex(idx).fillna(0.0)

    # -------------------------------------------------
    # Risk parity weights (inverse vol, 1-day lag)
    # -------------------------------------------------
    vol = r.rolling(vol_lookback).std(ddof=1).shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    w_rp = inv.div(inv.sum(axis=1), axis=0)

    # Warm-up → equal weight
    w_rp = w_rp.fillna(pd.Series(1 / len(cols), index=cols)).ffill()

    # -------------------------------------------------
    # Baseline weights (inactive)
    # -------------------------------------------------
    w_base = pd.DataFrame(
        np.tile(np.array(baseline_when_inactive, float), (len(idx), 1)),
        index=idx,
        columns=cols,
    )
    w_base = w_base.div(w_base.sum(axis=1), axis=0)

    # -------------------------------------------------
    # Select weights by regime
    # -------------------------------------------------
    cond = (active.values > 0.5).reshape(-1, 1)
    w_target = pd.DataFrame(
        np.where(cond, w_rp.values, w_base.values),
        index=idx,
        columns=cols,
    )

    # -------------------------------------------------
    # Portfolio vol targeting (1-day lag)
    # -------------------------------------------------
    port_ret = (r[cols] * w_target[cols]).sum(axis=1)
    realized = port_ret.rolling(vol_lookback).std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)
    leverage = (target_vol / realized).clip(0.0, leverage_cap).shift(1).fillna(1.0)

    dt = idx[-1]

    return {
        "asof": dt,
        "active": float(active.loc[dt]),
        "anchor_realized_vol": float(rv.loc[dt]),
        "weights": w_target.loc[dt],
        "risk_parity_weights": w_rp.loc[dt],
        "baseline_weights": w_base.loc[dt],
        "leverage": float(leverage.loc[dt]),
        "last_close": px.loc[dt],
    }


def _should_trade_today(
    px_index: pd.DatetimeIndex, asof: pd.Timestamp, rule: str
) -> bool:
    """
    Decide if `asof` is a rebalance date according to `rule`,
    using the *actual* trading dates available in px_index.

    rule examples: "W-FRI", "W-MON", "M"
    """
    s = pd.Series(1.0, index=px_index).sort_index()
    rb_dates = s.resample(rule).last().index
    # only consider rebalance dates that are actual trading dates
    rb_dates = rb_dates.intersection(px_index)
    return asof in set(rb_dates)


def _current_portfolio_weights(
    api, tickers: tuple[str, str, str], last_close: pd.Series
) -> pd.Series:
    """
    Compute current portfolio weights across `tickers` using Alpaca positions * last_close.
    Missing positions treated as 0.
    """
    pos_map: dict[str, float] = {}
    for p in api.list_positions():
        pos_map[str(p.symbol).upper()] = float(p.qty)

    vals: dict[str, float] = {}
    total = 0.0
    for t in tickers:
        qty = float(pos_map.get(t.upper(), 0.0))
        px = float(last_close.get(t, np.nan))
        v = 0.0 if (qty == 0.0 or (not np.isfinite(px)) or px <= 0.0) else qty * px
        vals[t] = v
        total += v

    if total <= 0.0:
        return pd.Series({t: 0.0 for t in tickers}, dtype=float)

    return pd.Series({t: vals[t] / total for t in tickers}, dtype=float)


def run_single_iteration(
    api,
    *,
    tickers: tuple[str, str, str] = ("SPY", "TLT", "GLD"),
    equity_fraction: float = 1.0,
    is_live_trade: bool = False,
    feed: str = "iex",
    # Trade gating
    is_rebalance_date: str = "W-FRI",  # calendar rule, e.g. "W-FRI" or "M"
    forced_rebalance: bool = False,  # NEW: force rebalance today (ignores calendar+drift)
    # Strategy params
    vol_lookback: int = 60,
    target_vol: float = 0.10,
    leverage_cap: float = 2.0,
    high_vol_anchor: str = "SPY",
    high_vol_window: int = 63,
    high_vol_threshold: float = 0.20,
    baseline_when_inactive: tuple[float, float, float] = (0.0, 1.0, 0.0),
    # NEW: drift-based trigger
    drift_threshold: float | None = 0.08,  # L1 drift in weights; None disables
    drift_only_when_active: bool = True,  # recommended: only drift-trade when active==1
    # History
    min_days: int = 420,
) -> dict:
    """
    Single-iteration execution for Alpaca.

    - Computes target weights + leverage each run (using only past data, no lookahead).
    - Trades only when:
        (calendar rebalance date) OR (drift >= drift_threshold) OR (forced_rebalance=True).
      *Optionally* drift-trigger only when active==1.
    - When not trading, returns diagnostics.

    Notes:
    - This function sizes target quantities based on account equity * equity_fraction * leverage.
    - It uses last close prices for sizing (not intraday).
    """
    # --------------------------------------------
    # Download history
    # --------------------------------------------
    end_date = datetime.utcnow().date() + timedelta(days=1)
    start_date = end_date - timedelta(days=min_days)

    closes: dict[str, pd.Series] = {}
    for t in tickers:
        bars = price_history(api, t, start_date, end_date, feed=feed)
        ser = _bars_to_series_close(bars)
        if not ser.empty:
            closes[t] = ser

    if not closes:
        raise RuntimeError("No price history available for tickers.")

    px = pd.concat(closes, axis=1).dropna(how="all")
    px.columns = [c.upper() for c in px.columns]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    px = px.sort_index()

    # ensure all tickers present
    missing = [t for t in tickers if t.upper() not in px.columns]
    if missing:
        raise RuntimeError(f"Missing price history for: {missing}")

    px = px[[t.upper() for t in tickers]].dropna()
    if len(px) < max(vol_lookback, high_vol_window) + 10:
        raise RuntimeError("Not enough history to compute signals.")

    # --------------------------------------------
    # Signals (as-of latest bar)
    # --------------------------------------------
    sig = _compute_target_weights_and_leverage(
        px,
        tickers=tuple([t.upper() for t in tickers]),
        vol_lookback=vol_lookback,
        target_vol=target_vol,
        leverage_cap=leverage_cap,
        high_vol_anchor=high_vol_anchor.upper(),
        high_vol_window=high_vol_window,
        high_vol_threshold=high_vol_threshold,
        baseline_when_inactive=baseline_when_inactive,
    )

    asof: pd.Timestamp = sig["asof"]
    w_target: pd.Series = (
        sig["weights"].reindex([t.upper() for t in tickers]).astype(float)
    )
    last_close: pd.Series = (
        sig["last_close"].reindex([t.upper() for t in tickers]).astype(float)
    )
    active: float = float(sig["active"])
    lev: float = float(sig["leverage"])

    # --------------------------------------------
    # Trade gating: calendar OR drift OR forced
    # --------------------------------------------
    should_trade_calendar = _should_trade_today(px.index, asof, is_rebalance_date)

    drift: Optional[float] = None
    should_trade_drift = False
    if drift_threshold is not None:
        w_now = _current_portfolio_weights(
            api, tuple([t.upper() for t in tickers]), last_close
        )
        drift = float((w_now - w_target).abs().sum())
        if drift_only_when_active:
            should_trade_drift = (active > 0.5) and (drift >= float(drift_threshold))
        else:
            should_trade_drift = drift >= float(drift_threshold)

    should_trade = bool(forced_rebalance or should_trade_calendar or should_trade_drift)

    # --------------------------------------------
    # Diagnostics payload (always)
    # --------------------------------------------
    base_out = {
        "asof": asof,
        "forced_rebalance": forced_rebalance,
        "should_trade": should_trade,
        "should_trade_calendar": bool(should_trade_calendar),
        "should_trade_drift": bool(should_trade_drift),
        "rebalance_rule": is_rebalance_date,
        "drift": drift,
        "drift_threshold": drift_threshold,
        "drift_only_when_active": drift_only_when_active,
        "active": active,
        "anchor_realized_vol": float(sig.get("anchor_realized_vol", np.nan)),
        "weights": w_target.to_dict(),
        "leverage": lev,
        "last_close": last_close.to_dict(),
        "equity_fraction": float(equity_fraction),
        "Params": {
            "tickers": tuple([t.upper() for t in tickers]),
            "vol_lookback": vol_lookback,
            "target_vol": target_vol,
            "leverage_cap": leverage_cap,
            "high_vol_anchor": high_vol_anchor.upper(),
            "high_vol_window": high_vol_window,
            "high_vol_threshold": high_vol_threshold,
            "baseline_when_inactive": baseline_when_inactive,
            "periods_per_year": PERIODS_PER_YEAR,
            "feed": feed,
            "min_days": min_days,
            "is_rebalance_date": is_rebalance_date,
            "drift_threshold": drift_threshold,
            "drift_only_when_active": drift_only_when_active,
            "forced_rebalance": forced_rebalance,
        },
    }

    if not should_trade:
        base_out.update(
            {
                "account_equity": None,
                "alloc_equity": None,
                "target_qty": {t.upper(): None for t in tickers},
                "execution": {
                    t.upper(): {
                        "action": "NOOP",
                        "target_qty": None,
                        "diff": 0,
                        "submitted": [],
                    }
                    for t in tickers
                },
                "note": (
                    f"Skipped trading: forced={forced_rebalance}, "
                    f"calendar={should_trade_calendar} (rule={is_rebalance_date}), "
                    f"drift={drift} (thr={drift_threshold}, only_when_active={drift_only_when_active}), "
                    f"active={active}."
                ),
            }
        )
        return base_out

    # --------------------------------------------
    # Position sizing (only when trading)
    # --------------------------------------------
    account_equity = float(api.get_account().equity)
    alloc_equity = account_equity * float(equity_fraction)

    target_qty: dict[str, int] = {}
    for sym in [t.upper() for t in tickers]:
        w = float(w_target.get(sym, 0.0))
        dollars = alloc_equity * lev * w
        price = float(last_close.get(sym, np.nan))
        target_qty[sym] = (
            int(dollars // price) if (np.isfinite(price) and price > 0) else 0
        )

    # --------------------------------------------
    # Execute
    # --------------------------------------------
    executions: dict[str, dict] = {}
    for sym, qty in target_qty.items():
        # process_position returns dict in your shared format
        action, current_qty, qty, diff = process_position(
            api, sym, qty, is_live_trade=is_live_trade
        )

        executions[sym] = {
            "action": action,
            "current_qty": current_qty,
            "target_qty": qty,
            "diff": diff,
            "submitted": is_live_trade,
        }

    base_out.update(
        {
            "account_equity": account_equity,
            "alloc_equity": alloc_equity,
            "target_qty": target_qty,
            "execution": executions,
            "note": (
                f"Traded: forced={forced_rebalance}, calendar={should_trade_calendar}, "
                f"drift_trigger={should_trade_drift}, drift={drift}."
            ),
        }
    )
    return base_out


def print_orders_table(result: dict, stream: Optional[TextIO] = None) -> str:
    """
    Pretty-print execution orders from run_single_iteration() output.

    Returns the formatted string.
    If stream is provided, also writes to that stream.

    Safe for both trade and NOOP days.
    """

    buf = io.StringIO()

    def w(line=""):
        buf.write(line + "\n")

    asof = result.get("asof")
    should_trade = result.get("should_trade", False)
    rebalance_rule = result.get("rebalance_rule", "N/A")
    active = result.get("active")
    leverage = result.get("leverage")
    execution = result.get("execution", {})

    w("=" * 72)
    w(f"Orders as of: {asof.date() if asof is not None else 'N/A'}")
    w(f"Rebalance rule: {rebalance_rule}")
    w(f"High-vol regime active: {bool(active)}")
    w(
        f"Leverage: {leverage:.2f}"
        if isinstance(leverage, (int, float))
        else "Leverage: N/A"
    )
    w(f"Should trade today: {should_trade}")
    w("-" * 72)

    if not should_trade or not execution:
        w("NO ORDERS (not a rebalance date)")
        w("=" * 72)
    else:
        header = f"{'Symbol':<8} {'Action':<6} {'TargetQty':>10} {'Diff':>8}"
        w(header)
        w("-" * len(header))

        for sym, info in execution.items():
            action = info.get("action", "NOOP")
            tgt = info.get("target_qty")
            diff = info.get("diff", 0)

            tgt_str = f"{int(tgt):d}" if isinstance(tgt, (int, float)) else "—"
            diff_str = f"{int(diff):+d}" if isinstance(diff, (int, float)) else "—"

            w(f"{sym:<8} {action:<6} {tgt_str:>10} {diff_str:>8}")

        w("=" * 72)

    out = buf.getvalue()

    if stream:
        print(out, file=stream)

    return out


def _infer_holding_period_days(rule: str | None) -> int | None:
    if not rule:
        return None

    rule = str(rule).upper()
    if rule.startswith("W-"):
        return 7
    if rule in {"D", "B"}:
        return 1
    if rule.startswith("M"):
        return 31
    if rule.startswith("Q"):
        return 92
    return None


def _to_float(value, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def export_strategy_json(
    result: dict,
    output_path: str = "strategy_export.json",
    *,
    strategy_name: str = "etf-volatility-harvest",
    equity_fraction: float = 1.0,
) -> dict:
    """
    Export a run_single_iteration() result as a compact strategy JSON payload.

    Mapping:
    - `positions[].target_weight` is leverage-adjusted exposure per symbol.
    - `capital_requested` is the unlevered account allocation fraction.
    - `gross_exposure` / `net_exposure` are derived from exported positions.
    """

    weights = result.get("weights") or {}
    leverage = _to_float(result.get("leverage"), default=0.0)

    capital_requested = _to_float(equity_fraction, default=0.0)

    trade_today = capital_requested > 0.0
    liquidate_when_inactive = capital_requested <= 0.0

    positions = []
    if trade_today:
        for symbol, weight in weights.items():
            target_weight = _to_float(weight) * leverage
            if abs(target_weight) <= 1e-12:
                continue
            positions.append(
                {
                    "symbol": str(symbol).upper(),
                    "target_weight": round(target_weight, 10),
                }
            )

        positions.sort(key=lambda item: item["symbol"])

    gross_exposure = round(sum(abs(p["target_weight"]) for p in positions), 10)
    net_exposure = round(sum(p["target_weight"] for p in positions), 10)

    payload = {
        "strategy": strategy_name,
        "trade_today": trade_today,
        "liquidate_when_inactive": liquidate_when_inactive,
        "capital_requested": round(capital_requested, 10),
        "positions": positions,
        "gross_exposure": gross_exposure,
        "net_exposure": net_exposure,
        "holding_period_days": _infer_holding_period_days(result.get("rebalance_rule")),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")

    return payload


def upload_file_to_digitalocean_spaces(
    file_path: str,
    *,
    bucket_name: str,
    region: str,
    object_key: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    content_type: str = "application/json",
    acl: str | None = None,
) -> dict:
    """Upload a local file to DigitalOcean Spaces via the S3-compatible API."""

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not os.path.isfile(file_path):
        raise IsADirectoryError(
            f"Expected a file path, got a directory or non-file path: {file_path}"
        )
    if not bucket_name:
        raise ValueError("bucket_name is required")
    if not region:
        raise ValueError("region is required")

    spaces_key = access_key or os.getenv("DO_SPACES_KEY")
    spaces_secret = secret_key or os.getenv("DO_SPACES_SECRET")
    if not spaces_key or not spaces_secret:
        raise ValueError(
            "DigitalOcean Spaces credentials are required via arguments or "
            "DO_SPACES_KEY / DO_SPACES_SECRET environment variables."
        )

    key = object_key or os.path.basename(file_path)
    endpoint_url = f"https://{region}.digitaloceanspaces.com"

    client = boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
    )

    extra_args = {"ContentType": content_type}
    if acl:
        extra_args["ACL"] = acl

    client.upload_file(file_path, bucket_name, key, ExtraArgs=extra_args)

    return {
        "bucket": bucket_name,
        "region": region,
        "object_key": key,
        "endpoint_url": endpoint_url,
        "object_url": f"https://{bucket_name}.{region}.digitaloceanspaces.com/{key}",
    }
