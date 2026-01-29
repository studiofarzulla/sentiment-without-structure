"""
Liquidity Metrics Module
========================

Computes market microstructure/liquidity metrics from OHLCV data.
These are proper econometric measures that don't require tick-level data.

Metrics implemented:
1. Amihud (2002) Illiquidity Ratio - Price impact per unit volume
2. Roll (1984) Spread Estimator - Effective spread from return autocorrelation
3. Corwin-Schultz (2012) Spread - Spread from high-low prices
4. Kyle's Lambda - Price impact coefficient
5. Volume-based measures - Abnormal volume, volume volatility

References:
- Amihud, Y. (2002). Illiquidity and stock returns. JFM 5(1), 31-56.
- Roll, R. (1984). A simple implicit measure of bid-ask spread. JF 39(4), 1127-1139.
- Corwin, S. & Schultz, P. (2012). A simple way to estimate bid-ask spreads. JF 67(2), 719-760.
- Kyle, A. (1985). Continuous auctions and insider trading. Econometrica 53(6), 1315-1335.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats

from . import config


def amihud_illiquidity(
    returns: pd.Series,
    volume: pd.Series,
    price: Optional[pd.Series] = None,
    window: int = 21
) -> pd.Series:
    """
    Compute Amihud (2002) illiquidity ratio.

    ILLIQ = |R_t| / V_t

    Where:
        R_t = daily return
        V_t = daily dollar volume

    Higher values indicate lower liquidity (larger price impact per dollar traded).

    Args:
        returns: Daily return series
        volume: Daily volume series (in units or dollars)
        price: Optional price series to convert volume to dollar volume
        window: Rolling window for smoothing (default 21 = monthly)

    Returns:
        Amihud illiquidity ratio series
    """
    # Convert to dollar volume if price provided
    if price is not None:
        dollar_volume = volume * price
    else:
        dollar_volume = volume

    # Avoid division by zero
    dollar_volume = dollar_volume.replace(0, np.nan)

    # Raw Amihud ratio (scaled by 1e6 for readability)
    amihud = np.abs(returns) / dollar_volume * 1e6

    # Rolling average for stability
    if window > 1:
        amihud_smooth = amihud.rolling(window=window, min_periods=1).mean()
        return amihud_smooth

    return amihud


def roll_spread(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Compute Roll (1984) effective spread estimator.

    Roll shows that bid-ask spread induces negative autocorrelation in returns:
        Cov(R_t, R_{t-1}) = -c^2
    where c = half-spread

    Effective spread = 2 * sqrt(-Cov) if Cov < 0, else 0

    Args:
        returns: Daily return series
        window: Rolling window for covariance estimation

    Returns:
        Roll spread estimate series
    """
    def compute_roll_cov(x):
        if len(x) < 3:
            return np.nan
        # Covariance between consecutive returns
        cov = np.cov(x[:-1], x[1:])[0, 1]
        # Roll spread (only meaningful if cov < 0)
        if cov < 0:
            return 2 * np.sqrt(-cov)
        else:
            return 0.0

    roll = returns.rolling(window=window, min_periods=3).apply(compute_roll_cov, raw=True)

    return roll


def corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Compute Corwin-Schultz (2012) bid-ask spread estimator.

    Uses the ratio of high-low over two consecutive days to separate
    volatility from spread components.

    S = (2 * (exp(alpha) - 1)) / (1 + exp(alpha))

    where alpha is derived from high-low ratios.

    Args:
        high: Daily high price series
        low: Daily low price series

    Returns:
        Corwin-Schultz spread estimate series
    """
    # Beta: Variance estimator from single-day high-low
    beta = np.log(high / low) ** 2

    # Gamma: Two-day high-low for separating volatility
    high_2day = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low_2day = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = np.log(high_2day / low_2day) ** 2

    # Alpha calculation
    beta_sum = beta + beta.shift(1)
    alpha = (np.sqrt(2 * beta_sum) - np.sqrt(beta_sum)) / (3 - 2 * np.sqrt(2)) - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))

    # Ensure alpha is non-negative for spread calculation
    alpha = alpha.clip(lower=0)

    # Spread estimate
    spread = (2 * (np.exp(alpha) - 1)) / (1 + np.exp(alpha))

    # Cap at reasonable values (spread > 100% is unrealistic)
    spread = spread.clip(upper=1.0)

    return spread


def kyle_lambda(
    returns: pd.Series,
    volume: pd.Series,
    window: int = 21
) -> pd.Series:
    """
    Compute Kyle's Lambda (price impact coefficient).

    Lambda measures price impact per unit of order flow:
        R_t = lambda * sqrt(V_t) + epsilon

    Higher lambda = higher price impact = lower liquidity.

    This is a simplified version using OLS regression in rolling windows.

    Args:
        returns: Daily return series
        volume: Daily volume series
        window: Rolling window for regression

    Returns:
        Kyle's lambda series
    """
    # Signed order flow proxy (sqrt of volume * sign of return)
    signed_volume = np.sign(returns) * np.sqrt(np.abs(volume))

    def compute_lambda(idx):
        if idx < window:
            return np.nan
        r = returns.iloc[idx-window:idx].values
        sv = signed_volume.iloc[idx-window:idx].values

        # Remove NaNs
        mask = ~(np.isnan(r) | np.isnan(sv))
        if mask.sum() < 5:
            return np.nan

        r = r[mask]
        sv = sv[mask]

        # OLS: r = lambda * sv + epsilon
        try:
            slope, _, _, _, _ = stats.linregress(sv, r)
            return abs(slope)
        except:
            return np.nan

    lambdas = pd.Series(
        [compute_lambda(i) for i in range(len(returns))],
        index=returns.index
    )

    return lambdas


def volume_metrics(volume: pd.Series, window: int = 21) -> pd.DataFrame:
    """
    Compute volume-based liquidity metrics.

    Args:
        volume: Daily volume series
        window: Rolling window for baseline

    Returns:
        DataFrame with volume metrics
    """
    # Rolling mean and std
    vol_mean = volume.rolling(window=window, min_periods=1).mean()
    vol_std = volume.rolling(window=window, min_periods=1).std()

    # Abnormal volume (standardized)
    abnormal_vol = (volume - vol_mean) / vol_std

    # Volume coefficient of variation
    vol_cv = vol_std / vol_mean

    # Volume turnover (relative to rolling average)
    vol_turnover = volume / vol_mean

    return pd.DataFrame({
        'volume_mean': vol_mean,
        'volume_std': vol_std,
        'abnormal_volume': abnormal_vol,
        'volume_cv': vol_cv,
        'volume_turnover': vol_turnover
    })


def compute_all_liquidity_metrics(
    ohlcv: pd.DataFrame,
    window: int = 21
) -> pd.DataFrame:
    """
    Compute all liquidity metrics from OHLCV data.

    Args:
        ohlcv: DataFrame with columns: open, high, low, close, volume
        window: Rolling window for smoothing

    Returns:
        DataFrame with all liquidity metrics
    """
    # Standardize column names
    df = ohlcv.copy()
    df.columns = [c.lower() for c in df.columns]

    # Compute returns
    df['returns'] = df['close'].pct_change()

    # Dollar volume (if not already)
    if 'dollar_volume' not in df.columns:
        df['dollar_volume'] = df['volume'] * df['close']

    # Compute metrics
    metrics = pd.DataFrame(index=df.index)

    # Amihud
    metrics['amihud'] = amihud_illiquidity(
        df['returns'],
        df['dollar_volume'],
        window=window
    )

    # Roll spread
    metrics['roll_spread'] = roll_spread(df['returns'], window=window)

    # Corwin-Schultz spread
    metrics['cs_spread'] = corwin_schultz_spread(df['high'], df['low'])

    # Kyle's lambda
    metrics['kyle_lambda'] = kyle_lambda(df['returns'], df['volume'], window=window)

    # Volume metrics
    vol_metrics = volume_metrics(df['volume'], window=window)
    for col in vol_metrics.columns:
        metrics[col] = vol_metrics[col]

    # Add price/return columns for reference
    metrics['price'] = df['close']
    metrics['returns'] = df['returns']
    metrics['volume'] = df['volume']

    return metrics


def compute_event_liquidity_change(
    metrics: pd.DataFrame,
    event_date: str,
    pre_window: int = 30,
    post_window: int = 30
) -> Dict:
    """
    Compute pre/post event changes in liquidity metrics.

    Args:
        metrics: DataFrame with liquidity metrics
        event_date: Event date (YYYY-MM-DD)
        pre_window: Days before event
        post_window: Days after event

    Returns:
        Dictionary with changes and significance tests
    """
    event_dt = pd.to_datetime(event_date)

    # Define windows
    pre_start = event_dt - pd.Timedelta(days=pre_window)
    pre_end = event_dt - pd.Timedelta(days=1)
    post_start = event_dt + pd.Timedelta(days=1)
    post_end = event_dt + pd.Timedelta(days=post_window)

    # Split data
    pre_data = metrics[(metrics.index >= pre_start) & (metrics.index <= pre_end)]
    post_data = metrics[(metrics.index >= post_start) & (metrics.index <= post_end)]

    if len(pre_data) < 5 or len(post_data) < 5:
        return {'error': 'Insufficient data'}

    results = {'event_date': event_date}

    # Metrics to analyze
    liquidity_cols = ['amihud', 'roll_spread', 'cs_spread', 'kyle_lambda',
                      'abnormal_volume', 'volume_turnover']

    for col in liquidity_cols:
        if col not in metrics.columns:
            continue

        pre = pre_data[col].dropna()
        post = post_data[col].dropna()

        if len(pre) < 5 or len(post) < 5:
            continue

        # Statistics
        pre_mean = pre.mean()
        post_mean = post.mean()
        change = post_mean - pre_mean
        change_pct = (change / abs(pre_mean)) * 100 if pre_mean != 0 else np.nan

        # T-test
        t_stat, t_pval = stats.ttest_ind(pre, post)

        results[f'{col}_pre_mean'] = pre_mean
        results[f'{col}_post_mean'] = post_mean
        results[f'{col}_change'] = change
        results[f'{col}_change_pct'] = change_pct
        results[f'{col}_t_stat'] = t_stat
        results[f'{col}_t_pval'] = t_pval
        results[f'{col}_significant'] = t_pval < 0.05

    return results


class LiquidityAnalyzer:
    """Analyze liquidity metrics around events."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize analyzer."""
        self.data_dir = data_dir or Path(config.DATA_DIR)
        self.results_dir = config.ANALYSIS_RESULTS_DIR / 'liquidity'
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def analyze_event(
        self,
        ohlcv: pd.DataFrame,
        event_date: str,
        symbol: str,
        event_name: str,
        event_type: str
    ) -> Dict:
        """
        Analyze liquidity metrics around a single event.

        Args:
            ohlcv: OHLCV DataFrame for the symbol
            event_date: Event date
            symbol: Asset symbol
            event_name: Name of event
            event_type: Event type (Infrastructure/Regulatory)

        Returns:
            Dictionary with analysis results
        """
        # Compute metrics
        metrics = compute_all_liquidity_metrics(ohlcv)

        # Compute event impact
        results = compute_event_liquidity_change(
            metrics,
            event_date,
            pre_window=config.EVENT_WINDOW_PRE,
            post_window=config.EVENT_WINDOW_POST
        )

        results['symbol'] = symbol
        results['event_name'] = event_name
        results['event_type'] = event_type

        return results

    def summarize_results(self, results: List[Dict]) -> pd.DataFrame:
        """
        Summarize multiple event results into a table.

        Args:
            results: List of result dictionaries

        Returns:
            Summary DataFrame
        """
        df = pd.DataFrame(results)

        # Compute support rates for hypothesis
        # Hypothesis: Infrastructure events show larger liquidity deterioration than regulatory
        if 'event_type' in df.columns:
            infra = df[df['event_type'] == 'Infrastructure']
            reg = df[df['event_type'] == 'Regulatory']

            print("\n" + "=" * 60)
            print("LIQUIDITY ANALYSIS SUMMARY")
            print("=" * 60)

            for metric in ['amihud', 'roll_spread', 'cs_spread']:
                col = f'{metric}_change_pct'
                if col in df.columns:
                    infra_mean = infra[col].mean()
                    reg_mean = reg[col].mean()
                    print(f"\n{metric}:")
                    print(f"  Infrastructure mean change: {infra_mean:+.2f}%")
                    print(f"  Regulatory mean change:     {reg_mean:+.2f}%")
                    print(f"  Difference (Infra - Reg):   {infra_mean - reg_mean:+.2f}%")

        return df


if __name__ == "__main__":
    # Test with sample data
    print("Testing liquidity metrics computation...")

    # Generate sample OHLCV data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2024-01-01', periods=n, freq='D')
    price = 100 + np.cumsum(np.random.randn(n))
    price = np.maximum(price, 10)  # Ensure positive

    sample_data = pd.DataFrame({
        'open': price * (1 + np.random.randn(n) * 0.01),
        'high': price * (1 + np.abs(np.random.randn(n)) * 0.02),
        'low': price * (1 - np.abs(np.random.randn(n)) * 0.02),
        'close': price,
        'volume': np.abs(np.random.randn(n)) * 1e6 + 1e6
    }, index=dates)

    # Compute metrics
    metrics = compute_all_liquidity_metrics(sample_data)

    print("\nComputed metrics:")
    print(metrics.tail(10))

    # Test event analysis
    results = compute_event_liquidity_change(
        metrics,
        event_date='2024-02-15',
        pre_window=20,
        post_window=20
    )

    print("\nEvent analysis results:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    print("\n[SUCCESS] Liquidity metrics module working!")
