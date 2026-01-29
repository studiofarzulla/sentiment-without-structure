"""
Data Fetcher Module
====================

Fetches OHLCV and funding rate data from Binance APIs.
Handles rate limiting, caching, and data validation.
"""

import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from . import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """Fetches historical data from Binance APIs."""

    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize fetcher with optional cache directory."""
        self.cache_dir = cache_dir or config.DATA_DIR / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        time.sleep(config.BINANCE_RATE_LIMIT)

    def _get_cache_path(self, symbol: str, data_type: str, start: str, end: str) -> Path:
        """Get cache file path for a data request."""
        return self.cache_dir / f"{symbol}_{data_type}_{start}_{end}.parquet"

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV data from Binance spot API.

        Args:
            symbol: Asset symbol (e.g., 'BTC')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: open, high, low, close, volume, returns
        """
        # Map to Binance symbol
        binance_symbol = config.BINANCE_SPOT_SYMBOLS.get(
            symbol.upper(),
            f"{symbol.upper()}USDT"
        )

        # Check cache
        cache_path = self._get_cache_path(symbol, 'ohlcv', start_date, end_date)
        if use_cache and cache_path.exists():
            logger.info(f"Loading {symbol} OHLCV from cache")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching {symbol} ({binance_symbol}) OHLCV: {start_date} to {end_date}")

        # Convert dates to timestamps
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            try:
                params = {
                    'symbol': binance_symbol,
                    'interval': '1d',
                    'startTime': current_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }

                resp = requests.get(
                    f"{self.SPOT_BASE_URL}/api/v3/klines",
                    params=params
                )

                if resp.status_code != 200:
                    logger.warning(f"API error: {resp.status_code} - {resp.text[:100]}")
                    break

                data = resp.json()
                if not data:
                    break

                all_data.extend(data)
                current_ts = data[-1][0] + 1

                self._rate_limit()

            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                break

        if not all_data:
            logger.warning(f"No OHLCV data for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Add returns
        df['returns'] = df['close'].pct_change()

        # Select columns
        df = df[['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'returns']]
        df['symbol'] = symbol.upper()

        # Cache
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} OHLCV records for {symbol}")

        return df

    def fetch_funding_rates(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch funding rate data from Binance Futures API.

        Funding rates are settled every 8 hours (00:00, 08:00, 16:00 UTC).

        Args:
            symbol: Asset symbol (e.g., 'BTC')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with funding rate data
        """
        # Check if asset has futures
        if symbol.upper() not in config.BINANCE_FUTURES_SYMBOLS:
            logger.warning(f"{symbol} does not have perpetual futures")
            return pd.DataFrame()

        binance_symbol = config.BINANCE_FUTURES_SYMBOLS[symbol.upper()]

        # Check cache
        cache_path = self._get_cache_path(symbol, 'funding', start_date, end_date)
        if use_cache and cache_path.exists():
            logger.info(f"Loading {symbol} funding rates from cache")
            return pd.read_parquet(cache_path)

        logger.info(f"Fetching {symbol} ({binance_symbol}) funding rates")

        # Convert dates
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)

        all_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            try:
                params = {
                    'symbol': binance_symbol,
                    'startTime': current_ts,
                    'endTime': end_ts,
                    'limit': 1000
                }

                resp = requests.get(
                    f"{self.FUTURES_BASE_URL}/fapi/v1/fundingRate",
                    params=params
                )

                if resp.status_code != 200:
                    logger.warning(f"Funding API error: {resp.status_code}")
                    break

                data = resp.json()
                if not data:
                    break

                all_data.extend(data)
                current_ts = data[-1]['fundingTime'] + 1

                self._rate_limit()

            except Exception as e:
                logger.error(f"Error fetching {symbol} funding: {e}")
                break

        if not all_data:
            logger.warning(f"No funding data for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
        df['funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
        df['symbol'] = symbol.upper()

        # Annualized rate (3 settlements/day * 365 days * 100 for %)
        df['funding_rate_annualized'] = df['funding_rate'] * 3 * 365 * 100

        df = df[['timestamp', 'symbol', 'funding_rate', 'funding_rate_annualized']]
        df.set_index('timestamp', inplace=True)

        # Cache
        df.to_parquet(cache_path)
        logger.info(f"Cached {len(df)} funding records for {symbol}")

        return df

    def fetch_all_assets(
        self,
        assets: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for all specified assets.

        Args:
            assets: List of assets (defaults to config.ALL_ASSETS)
            start_date: Start date (defaults to config.START_DATE)
            end_date: End date (defaults to config.END_DATE)

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        assets = assets or config.ALL_ASSETS
        start_date = start_date or config.START_DATE
        end_date = end_date or config.END_DATE

        logger.info(f"Fetching {len(assets)} assets from {start_date} to {end_date}")

        results = {}
        for symbol in assets:
            try:
                df = self.fetch_ohlcv(symbol, start_date, end_date)
                if not df.empty:
                    results[symbol] = df
                    logger.info(f"  {symbol}: {len(df)} records")
            except Exception as e:
                logger.error(f"  {symbol}: FAILED - {e}")

        logger.info(f"Successfully fetched {len(results)}/{len(assets)} assets")
        return results


def fetch_and_save_all_data():
    """Fetch all OHLCV and funding rate data for the asset universe."""
    fetcher = BinanceDataFetcher()

    print("=" * 60)
    print("DATA COLLECTION - FULL ASSET UNIVERSE")
    print("=" * 60)
    print(f"Assets: {config.ALL_ASSETS}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}")
    print("=" * 60)

    # Fetch OHLCV
    print("\n--- OHLCV DATA ---")
    ohlcv_data = fetcher.fetch_all_assets()

    # Fetch funding rates
    print("\n--- FUNDING RATES ---")
    funding_data = {}
    for symbol in config.BINANCE_FUTURES_SYMBOLS.keys():
        df = fetcher.fetch_funding_rates(
            symbol,
            config.START_DATE,
            config.END_DATE
        )
        if not df.empty:
            funding_data[symbol] = df
            print(f"  {symbol}: {len(df)} funding records")

    # Save combined datasets
    print("\n--- SAVING COMBINED DATA ---")

    # OHLCV
    all_ohlcv = pd.concat(ohlcv_data.values())
    ohlcv_file = config.DATA_DIR / 'all_ohlcv.parquet'
    all_ohlcv.to_parquet(ohlcv_file)
    print(f"  OHLCV: {ohlcv_file} ({len(all_ohlcv)} records)")

    # Funding
    if funding_data:
        all_funding = pd.concat(funding_data.values())
        funding_file = config.DATA_DIR / 'all_funding_rates.parquet'
        all_funding.to_parquet(funding_file)
        print(f"  Funding: {funding_file} ({len(all_funding)} records)")

    print("\n[DONE] Data collection complete")

    return ohlcv_data, funding_data


if __name__ == "__main__":
    fetch_and_save_all_data()
