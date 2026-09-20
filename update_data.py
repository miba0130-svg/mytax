#!/usr/bin/env python3
"""
5-year daily market data updater for the Global Market Briefing dashboard.

Fetches daily closes via yfinance for the series the dashboard charts
(S&P 500, KOSPI, Nikkei 225, USD/KRW, Gold, WTI, US 10Y Treasury yield,
plus derived KRW/JPY and KRW/BRL cross rates), and separately pulls
Brazil's 10-year government yield from ANBIMA's ETTJ curve via pyettj
(Yahoo Finance has no reliable direct ticker for this one). Writes
data.json in the schema index.html's Chart.js code expects.

Run manually:
    pip install yfinance pandas pyettj numpy
    python update_data.py

Run automatically:
    see .github/workflows/update.yml (weekday schedule + manual dispatch)

To add another plain yfinance series later: add an entry to SERIES_CONFIG
below, add a matching key to the SERIES object in index.html, and make
that item's card/row pass a "tap" key equal to the same dict key.

To add another derived FX cross (KRW vs. some third currency) later: add
an entry to DERIVED_FX_CONFIG the same way — see krwjpy/krwbrl below for
the pattern (base "krw" series divided by a USD-quoted cross ticker).

Brazil 10Y (and any other series ANBIMA/pyettj-style sources are needed
for) is handled separately by fetch_brazil10y() below: unlike the
yfinance series, it can't do one bulk 5-year pull — pyettj issues one
HTTP request per calendar day, so a full 5-year backfill is ~1,300
requests. To keep each run fast and avoid hammering ANBIMA, this script
only fetches BRAZIL10Y_MAX_NEW_DAYS new days per run, and relies on
data.json (committed back to the repo every run) as its own incremental
cache: the first runs slowly backfill history a chunk at a time, and once
caught up each run just appends the latest day. No separate cache file
or manual backfill step is needed — running the existing scheduled
workflow repeatedly is what performs the backfill.
"""
import json
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
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

# Derived FX crosses: value = (base series' USD/KRW close) / (cross_ticker's
# USD/<currency> close) * multiplier. Yahoo Finance doesn't reliably carry
# direct KRWJPY=X / KRWBRL=X-style cross tickers, but it does reliably carry
# USD-quoted pairs for almost every currency, so we derive the cross from
# two of those instead — same trick markets use internally.
# "base" must be a key already fetched into `series` via SERIES_CONFIG.
DERIVED_FX_CONFIG = {
    "krwjpy": {"base": "krw", "cross_ticker": "JPY=X", "multiplier": 100,
               "label": "원/엔", "unit": "원 (100엔)", "decimals": 2},
    "krwbrl": {"base": "krw", "cross_ticker": "BRL=X", "multiplier": 1,
               "label": "원/헤알", "unit": "원", "decimals": 2},
}

# Brazil 10Y government yield, from ANBIMA's ETTJ "PRE" (pre-fixed nominal)
# curve via pyettj, interpolated at the 10-year (3650 calendar-day) vertex.
BRAZIL10Y_KEY = "brazil10y"
BRAZIL10Y_LABEL = "브라질 10년물 금리"
BRAZIL10Y_TARGET_DIAS_CORRIDOS = 3650  # 10 years, calendar days
BRAZIL10Y_DECIMALS = 2
BRAZIL10Y_MAX_NEW_DAYS = 60  # cap per run — keep runs fast, be polite to ANBIMA
BRAZIL10Y_BACKFILL_YEARS = 5

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


def fetch_derived_fx(key, cfg, series):
    """Build a KRW cross rate from an already-fetched base series (KRW=X)
    and a freshly-fetched USD-quoted cross ticker, aligned by date.
    """
    base = series.get(cfg["base"])
    if base is None:
        raise RuntimeError(f"base series '{cfg['base']}' unavailable for '{key}'")

    cross_ticker = cfg["cross_ticker"]
    print(f"  fetching {key} (derived: {cfg['base']} / {cross_ticker}) ...", file=sys.stderr)
    hist = yf.Ticker(cross_ticker).history(period=PERIOD, interval=INTERVAL, auto_adjust=False)
    if hist.empty:
        raise RuntimeError(f"no data returned for {cross_ticker}")
    hist = hist.dropna(subset=["Close"])
    if hist.empty:
        raise RuntimeError(f"all rows for {cross_ticker} had no Close value")

    cross_by_date = {d.strftime("%Y-%m-%d"): float(v) for d, v in zip(hist.index, hist["Close"])}
    base_by_date = dict(zip(base["dates"], base["values"]))

    decimals = cfg["decimals"]
    multiplier = cfg["multiplier"]
    dates_out, values_out = [], []
    for d in sorted(set(base_by_date) & set(cross_by_date)):
        cross_v = cross_by_date[d]
        if not cross_v:
            continue
        dates_out.append(d)
        values_out.append(round(base_by_date[d] / cross_v * multiplier, decimals))

    if not dates_out:
        raise RuntimeError(f"no overlapping dates between '{cfg['base']}' and {cross_ticker}")

    return {
        "yahoo_ticker": f"{cfg['base']}/{cross_ticker}",
        "label": cfg["label"],
        "unit": cfg["unit"],
        "decimals": decimals,
        "dates": dates_out,
        "values": values_out,
    }


def _load_previous_output():
    """Read the data.json this script last committed, if any. Used so
    fetch_brazil10y() knows where its incremental backfill left off.
    """
    try:
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _interpolate_10y(day_df):
    """Given one day's ETTJ 'PRE' curve rows, linearly interpolate the
    taxa (rate) at the 10-year (3650 calendar-day) vertex.
    """
    day_df = day_df.sort_values("dias_corridos")
    xs = day_df["dias_corridos"].to_numpy(dtype=float)
    ys = day_df["taxa"].to_numpy(dtype=float)
    if len(xs) < 2:
        return None
    # np.interp clips to the nearest edge value outside [xs.min(), xs.max()]
    # rather than extrapolating — conservative, and the PRE curve's longest
    # vertex is normally well past 10 years anyway.
    return float(np.interp(BRAZIL10Y_TARGET_DIAS_CORRIDOS, xs, ys))


def fetch_brazil10y(previous_output):
    """Incrementally backfill/update Brazil's 10-year yield from ANBIMA's
    ETTJ PRE curve. See the module docstring for why this can't be a single
    bulk pull like the yfinance series.
    """
    import pyettj

    prev_series = (previous_output or {}).get("series", {}).get(BRAZIL10Y_KEY)
    existing_dates = list(prev_series["dates"]) if prev_series else []
    existing_values = list(prev_series["values"]) if prev_series else []

    today = date.today()
    if existing_dates:
        last = datetime.strptime(existing_dates[-1], "%Y-%m-%d").date()
        start = last + timedelta(days=1)
    else:
        start = today - timedelta(days=365 * BRAZIL10Y_BACKFILL_YEARS)

    if start > today:
        print(f"  {BRAZIL10Y_KEY}: already up to date ({existing_dates[-1] if existing_dates else 'n/a'})",
              file=sys.stderr)
        return {
            "source": "ANBIMA ETTJ (pyettj, curva PRE, interpolado em 3650 dias corridos)",
            "label": BRAZIL10Y_LABEL,
            "unit": "%",
            "decimals": BRAZIL10Y_DECIMALS,
            "deltaUnit": "bp",
            "dates": existing_dates,
            "values": existing_values,
        }

    end = min(start + timedelta(days=BRAZIL10Y_MAX_NEW_DAYS), today)
    print(f"  fetching {BRAZIL10Y_KEY} (ANBIMA PRE curve, {start} .. {end}) ...", file=sys.stderr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pyettj.get_ettj_historico(
            start.strftime("%d/%m/%Y"), end.strftime("%d/%m/%Y"), curva="PRE",
        )
    if df.empty:
        raise RuntimeError(f"pyettj returned no PRE curve data for {start}..{end}")

    new_dates, new_values = [], []
    for refdate, day_df in df.groupby("refdate"):
        v = _interpolate_10y(day_df)
        if v is None:
            continue
        new_dates.append(pd.Timestamp(refdate).strftime("%Y-%m-%d"))
        new_values.append(round(v, BRAZIL10Y_DECIMALS))

    combined = dict(zip(existing_dates, existing_values))
    combined.update(dict(zip(new_dates, new_values)))
    all_dates = sorted(combined)

    print(f"  {BRAZIL10Y_KEY}: +{len(new_dates)} day(s), {len(all_dates)} total "
          f"(caught up to {all_dates[-1] if all_dates else 'n/a'}, today is {today})",
          file=sys.stderr)

    return {
        "source": "ANBIMA ETTJ (pyettj, curva PRE, interpolado em 3650 dias corridos)",
        "label": BRAZIL10Y_LABEL,
        "unit": "%",
        "decimals": BRAZIL10Y_DECIMALS,
        "deltaUnit": "bp",
        "dates": all_dates,
        "values": [combined[d] for d in all_dates],
    }


def main():
    previous_output = _load_previous_output()
    series = {}
    failures = []

    for key, cfg in SERIES_CONFIG.items():
        try:
            series[key] = fetch_series(key, cfg)
        except Exception as e:  # noqa: BLE001 - one bad ticker shouldn't kill the whole run
            print(f"  WARNING: failed to fetch '{key}' ({cfg['ticker']}): {e}", file=sys.stderr)
            failures.append(key)

    for key, cfg in DERIVED_FX_CONFIG.items():
        try:
            series[key] = fetch_derived_fx(key, cfg, series)
        except Exception as e:  # noqa: BLE001
            print(f"  WARNING: failed to fetch '{key}' (derived): {e}", file=sys.stderr)
            failures.append(key)

    try:
        series[BRAZIL10Y_KEY] = fetch_brazil10y(previous_output)
    except Exception as e:  # noqa: BLE001
        print(f"  WARNING: failed to fetch '{BRAZIL10Y_KEY}': {e}", file=sys.stderr)
        failures.append(BRAZIL10Y_KEY)
        # don't lose yesterday's brazil10y data just because today's
        # incremental fetch failed
        if previous_output and BRAZIL10Y_KEY in previous_output.get("series", {}):
            series[BRAZIL10Y_KEY] = previous_output["series"][BRAZIL10Y_KEY]

    if not series:
        print("ERROR: every series failed to fetch — leaving the existing data.json untouched.", file=sys.stderr)
        sys.exit(1)

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "yfinance + ANBIMA ETTJ (pyettj)",
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
