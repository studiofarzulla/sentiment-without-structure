#!/usr/bin/env python3
"""
Run Full Analysis
=================

Main script to run the complete Sentiment Without Structure analysis.

Usage:
    python scripts/run_full_analysis.py

Outputs:
    - outputs/tables/main_results.csv
    - outputs/tables/robustness_results.csv
    - outputs/tables/did_results.csv
    - outputs/figures/car_by_event_type.pdf
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.event_study import EventStudyAnalyzer, DifferenceInDifferences
from src.robustness import RobustnessChecker, results_to_dataframe
from src.liquidity_metrics import compute_all_liquidity_metrics, compute_event_liquidity_change


def load_events() -> List[Dict]:
    """Load validated events from events_master.json."""
    events_file = config.DATA_DIR / 'events_master.json'

    with open(events_file) as f:
        data = json.load(f)

    # Filter to valid events
    valid_events = [
        e for e in data['events']
        if e['meets_impact_threshold'] and e['has_sufficient_estimation_data']
    ]

    print(f"Loaded {len(valid_events)} valid events from {len(data['events'])} total")
    return valid_events


def load_returns_data(
    fetcher: BinanceDataFetcher,
    assets: List[str]
) -> Dict[str, pd.Series]:
    """Load and prepare return series for all assets."""
    returns_dict = {}

    for symbol in assets:
        try:
            df = fetcher.fetch_ohlcv(
                symbol,
                config.START_DATE,
                config.END_DATE
            )
            if not df.empty and 'returns' in df.columns:
                returns_dict[symbol] = df['returns'].dropna()
                print(f"  {symbol}: {len(returns_dict[symbol])} returns")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    return returns_dict


def run_main_event_study(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    analyzer: EventStudyAnalyzer
) -> pd.DataFrame:
    """Run main event study on all events."""
    print("\n" + "=" * 60)
    print("MAIN EVENT STUDY")
    print("=" * 60)

    results = []

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_type = event['type']
        event_name = event['title']

        for symbol, returns in returns_dict.items():
            result = analyzer.analyze_single_event(
                returns=returns,
                event_date=event_date,
                event_id=event_id,
                event_name=event_name,
                event_type=event_type,
                symbol=symbol
            )

            if result:
                results.append({
                    'event_id': result.event_id,
                    'event_date': result.event_date,
                    'event_type': result.event_type,
                    'event_name': result.event_name,
                    'symbol': result.symbol,
                    'car': result.car,
                    't_stat': result.car_t_stat,
                    'p_value': result.car_p_value,
                    'bootstrap_ci_low': result.car_bootstrap_ci_low,
                    'bootstrap_ci_high': result.car_bootstrap_ci_high,
                    'bootstrap_pvalue': result.car_bootstrap_pvalue,
                    'estimation_n': result.estimation_n
                })

    df = pd.DataFrame(results)

    # Summary by event type
    print("\n--- Summary by Event Type ---")
    for event_type in df['event_type'].unique():
        subset = df[df['event_type'] == event_type]
        car_mean = subset['car'].mean()
        n_sig = (subset['p_value'] < 0.05).sum()
        print(f"  {event_type}: CAR mean = {car_mean:.4f}, {n_sig}/{len(subset)} significant")

    return df


def run_did_analysis(results_df: pd.DataFrame) -> Dict:
    """Run Difference-in-Differences analysis."""
    print("\n" + "=" * 60)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS")
    print("=" * 60)

    # Split by event type
    infra = results_df[results_df['event_type'] == 'Infrastructure']['car'].values
    reg = results_df[results_df['event_type'] == 'Regulatory']['car'].values

    # For DiD, we need pre/post distinction
    # Use sign of CAR as proxy (positive = post > pre)
    infra_pre = np.zeros(len(infra))  # Baseline
    infra_post = infra
    reg_pre = np.zeros(len(reg))
    reg_post = reg

    # DiD estimate
    did_result = DifferenceInDifferences.compute_did(
        infra_pre, infra_post,
        reg_pre, reg_post
    )

    # Bootstrap
    bootstrap_did = DifferenceInDifferences.bootstrap_did(
        infra_pre, infra_post,
        reg_pre, reg_post,
        n_bootstrap=config.BOOTSTRAP_REPLICATIONS
    )

    print(f"\nDiD Estimate: {did_result['did_estimate']:.4f}")
    print(f"  Infrastructure effect: {did_result['infra_diff']:.4f}")
    print(f"  Regulatory effect: {did_result['reg_diff']:.4f}")
    print(f"  t-stat: {did_result['t_stat']:.4f}")
    print(f"  p-value: {did_result['p_value']:.4f}")
    print(f"\nBootstrap 95% CI: [{bootstrap_did['bootstrap_ci_low']:.4f}, {bootstrap_did['bootstrap_ci_high']:.4f}]")

    return {**did_result, **bootstrap_did}


def run_liquidity_analysis(
    fetcher: BinanceDataFetcher,
    events: List[Dict],
    assets: List[str]
) -> pd.DataFrame:
    """Run liquidity metrics analysis around events."""
    print("\n" + "=" * 60)
    print("LIQUIDITY METRICS ANALYSIS")
    print("=" * 60)

    results = []

    for event in events:
        event_date = event['date']
        event_type = event['type']

        for symbol in assets:
            try:
                # Fetch OHLCV
                event_dt = pd.to_datetime(event_date)
                start = (event_dt - pd.Timedelta(days=60)).strftime('%Y-%m-%d')
                end = (event_dt + pd.Timedelta(days=60)).strftime('%Y-%m-%d')

                ohlcv = fetcher.fetch_ohlcv(symbol, start, end)

                if ohlcv.empty:
                    continue

                # Compute liquidity metrics
                metrics = compute_all_liquidity_metrics(ohlcv)

                # Compute event impact
                impact = compute_event_liquidity_change(
                    metrics, event_date,
                    pre_window=30, post_window=30
                )

                if 'error' not in impact:
                    impact['event_id'] = event['event_id']
                    impact['event_type'] = event_type
                    impact['symbol'] = symbol
                    results.append(impact)

            except Exception as e:
                print(f"  Error {symbol}/{event_date}: {e}")

    df = pd.DataFrame(results)

    # Summary
    if not df.empty:
        print("\n--- Liquidity Changes by Event Type ---")
        for metric in ['amihud', 'roll_spread', 'cs_spread']:
            col = f'{metric}_change_pct'
            if col in df.columns:
                for event_type in df['event_type'].unique():
                    mean_change = df[df['event_type'] == event_type][col].mean()
                    print(f"  {event_type} - {metric}: {mean_change:+.2f}%")

    return df


def main():
    """Run complete analysis pipeline."""
    print("\n" + "=" * 70)
    print("SENTIMENT WITHOUT STRUCTURE - FULL ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Assets: {config.TIER1_ASSETS + config.TIER2_ASSETS}")
    print(f"Period: {config.START_DATE} to {config.END_DATE}")
    print("=" * 70)

    # Initialize components
    fetcher = BinanceDataFetcher()
    analyzer = EventStudyAnalyzer()
    robustness_checker = RobustnessChecker()

    # 1. Load events
    events = load_events()
    infra_events = [e for e in events if e['type'] == 'Infrastructure']
    reg_events = [e for e in events if e['type'] == 'Regulatory']
    print(f"\nEvents: {len(infra_events)} Infrastructure, {len(reg_events)} Regulatory")

    # 2. Load returns data
    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]  # Start with subset
    returns_dict = load_returns_data(fetcher, assets)

    if not returns_dict:
        print("ERROR: No return data loaded")
        return

    # 3. Main event study
    main_results = run_main_event_study(returns_dict, events, analyzer)

    # 4. DiD analysis
    did_results = run_did_analysis(main_results)

    # 5. Robustness checks
    main_summary = {
        'car': main_results['car'].mean(),
        'p_value': main_results['p_value'].mean()
    }
    robustness_results = robustness_checker.run_all_checks(
        returns_dict, events, main_summary
    )

    # 6. Liquidity analysis
    liquidity_results = run_liquidity_analysis(
        fetcher, events[:10], assets[:2]  # Subset for speed
    )

    # 7. Save results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Main results
    main_file = config.TABLES_DIR / 'main_results.csv'
    main_results.to_csv(main_file, index=False)
    print(f"  Main results: {main_file}")

    # DiD results
    did_file = config.TABLES_DIR / 'did_results.json'
    with open(did_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                   for k, v in did_results.items()}, f, indent=2)
    print(f"  DiD results: {did_file}")

    # Robustness
    robustness_df = results_to_dataframe(robustness_results)
    robustness_file = config.TABLES_DIR / 'robustness_results.csv'
    robustness_df.to_csv(robustness_file, index=False)
    print(f"  Robustness: {robustness_file}")

    # Liquidity
    if not liquidity_results.empty:
        liquidity_file = config.TABLES_DIR / 'liquidity_results.csv'
        liquidity_results.to_csv(liquidity_file, index=False)
        print(f"  Liquidity: {liquidity_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Summary statistics
    print("\n--- KEY FINDINGS ---")
    print(f"Total event-asset observations: {len(main_results)}")
    print(f"Mean CAR (Infrastructure): {main_results[main_results['event_type']=='Infrastructure']['car'].mean():.4f}")
    print(f"Mean CAR (Regulatory): {main_results[main_results['event_type']=='Regulatory']['car'].mean():.4f}")
    print(f"DiD Estimate: {did_results['did_estimate']:.4f} (p={did_results['p_value']:.4f})")

    n_robust = sum(r.consistent_with_main for results in robustness_results.values() for r in results)
    n_total = sum(len(results) for results in robustness_results.values())
    print(f"Robustness: {n_robust}/{n_total} checks consistent")


if __name__ == "__main__":
    main()
