#!/usr/bin/env python3
"""
5-year daily market data updater for the Global Market Briefing dashboard.

Fetches daily closes via yfinance for the 6 series the dashboard charts
(S&P 500, KOSPI, Nikkei 225, USD/KRW, Gold, US 10Y Treasury yield) and
writes data.json in the schema index.html's Chart.js code expects.

Run manually:
    pip install yfinance pandas
    python update_data.py

Run automatically:
    see .github/workflows/update.yml (daily schedule + manual dispatch)

To add another series to the chart later: add an entry to SERIES_CONFIG
below, add a matching key to the SERIES object in index.html, and make
that item's card/row pass a "tap" key equal to the same dict key.
"""
import json
import sys
from datetime import datetime, timezone

import yfinance as yf

# key -> {ticker: Yahoo Finance symbol, label, unit, decimals, ...}
# "scale" (optional): multiply the raw Yahoo close by this before storing.
#   ^TNX's Close on Yahoo already IS the actual 10-year Treasury yield in
#   percent (e.g. 5.00 means 5.00%) - verified against Treasury.gov's daily
#   par yield curve for several dates. (An earlier version of this script
#   assumed ^TNX was quoted as 10x the yield and divided by 10, which was
#   wrong - it produced ~0.50 instead of ~5.00. No scaling is applied now.)
SERIES_CONFIG = {
    "sp500":  {"ticker": "^GSPC", "label": "S&P 500",         "unit": "pt",   "decimals": 2},
    "nasdaq": {"ticker": "^IXIC", "label": "나스닥종합",        "unit": "pt",   "decimals": 2},
    "kospi":  {"ticker": "^KS11", "label": "코스피",           "unit": "pt",   "decimals": 2},
    "nikkei": {"ticker": "^N225", "label": "니케이225",         "unit": "pt",   "decimals": 2},
    "krw":    {"ticker": "KRW=X", "label": "원/달러",          "unit": "원",   "decimals": 2},
    "gold":   {"ticker": "GC=F",  "label": "금",               "unit": "$/oz", "decimals": 2},
    "wti":    {"ticker": "CL=F",  "label": "WTI",              "unit": "$/bbl","decimals": 2},
    "us10y":  {"ticker": "^TNX",  "label": "미국 10년물 금리",  "unit": "%",    "decimals": 2,
               "deltaUnit": "bp"},
}

PERIOD = "5y"
INTERVAL = "1d"
OUTPUT_PATH = "data.json"


def fetch_series(key, cfg):
    ticker = cfg["ticker"]
    print(f"  fetching {key} ({ticker}) ...", file=sys.stderr)
    hist = yf.Ticker(ticker).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        raise RuntimeError(f"all rows for {ticker} had no Close value")

    scale = cfg.get("scale", 1.0)
    decimals = cfg["decimals"]
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    values = [round(float(v) * scale, decimals) for v in hist["Close"]]

    out = {
        "yahoo_ticker": ticker,
        "label": cfg["label"],
        "unit": cfg["unit"],
        "decimals": decimals,
        "dates": dates,
        "values": values,
    }
    if "deltaUnit" in cfg:
        out["deltaUnit"] = cfg["deltaUnit"]
    return out


def main():
    series = {}
    failures = []

    for key, cfg in SERIES_CONFIG.items():
        try:
            series[key] = fetch_series(key, cfg)
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't kill the whole run
            print(f"  WARNING: failed to fetch '{key}' ({cfg['ticker']}): {e}", file=sys.stderr)
            failures.append(key)

    if not series:
        print("ERROR: every series failed to fetch — leaving the existing data.json untouched.", file=sys.stderr)
        sys.exit(1)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "yfinance",
        "period": PERIOD,
        "interval": INTERVAL,
        "series": series,
    }
    if failures:
        # index.html doesn't read this field; it's just left as a breadcrumb
        # for whoever is debugging a partial update.
        payload["failed_series"] = failures

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    ok = ", ".join(sorted(series.keys()))
    msg = f"wrote {OUTPUT_PATH} with {len(series)} series ({ok})"
    if failures:
        msg += f" — FAILED: {', '.join(failures)}"
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
