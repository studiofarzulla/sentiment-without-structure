#!/usr/bin/env python3
"""
Run Reclassified Analysis
=========================

Dedicated script for clean comparison using 4-category event classification.
Addresses peer review feedback on event conflation.

Primary comparison: Infra_Negative (10) vs Reg_Negative (7)
- Both are negative-valence shocks
- Clean test of enforcement capacity hypothesis

Secondary comparison: Infra_Positive vs Reg_Positive
- How do markets respond to positive shocks by type?

Usage:
    python scripts/run_reclassified_analysis.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.event_study import (
    EventStudyAnalyzer, DifferenceInDifferences, ConstantMeanModel,
    MarketModel, EWMarketModel, compute_pre_event_car, run_market_model_analysis,
    run_ew_market_analysis
)
from src.robustness import RobustnessChecker, results_to_dataframe, EventBlockBootstrap


def load_reclassified_events() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """
    Load events from reclassified JSON.

    Returns:
        Tuple of (all events, events_by_type dict)
    """
    events_file = config.DATA_DIR / 'events_reclassified.json'

    with open(events_file) as f:
        data = json.load(f)

    # Filter to included events with sufficient data
    events = [
        e for e in data['events']
        if e.get('include_in_reanalysis', True)
        and e.get('meets_impact_threshold', False)
        and e.get('has_sufficient_estimation_data', True)
    ]

    # Group by detailed type
    events_by_type = {}
    for e in events:
        etype = e.get('type_detailed', e['type'])
        if etype not in events_by_type:
            events_by_type[etype] = []
        events_by_type[etype].append(e)

    print("\n--- Reclassified Event Summary ---")
    for etype, elist in sorted(events_by_type.items()):
        print(f"  {etype}: {len(elist)} events")

    return events, events_by_type


def load_returns_data(fetcher: BinanceDataFetcher, assets: List[str]) -> Dict[str, pd.Series]:
    """Load return series for all assets."""
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


def run_event_study_by_type(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    analyzer: EventStudyAnalyzer,
    label: str
) -> pd.DataFrame:
    """
    Run event study on a specific set of events.

    Args:
        returns_dict: Return series by symbol
        events: List of events to analyze
        analyzer: EventStudyAnalyzer instance
        label: Label for this analysis

    Returns:
        DataFrame with results
    """
    results = []

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_type = event.get('type_detailed', event['type'])
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

    if not df.empty:
        car_mean = df['car'].mean()
        car_std = df['car'].std()
        n_sig = (df['p_value'] < 0.05).sum()
        n_total = len(df)
        print(f"  {label}: CAR = {car_mean:.4f} (SD={car_std:.4f}), {n_sig}/{n_total} significant")

    return df


def compute_aggregate_stats(df: pd.DataFrame) -> Dict:
    """Compute aggregate statistics for a results DataFrame."""
    if df.empty:
        return {}

    from scipy import stats

    car = df['car'].values
    car_mean = car.mean()
    car_std = car.std()
    n = len(car)

    t_stat = car_mean / (car_std / np.sqrt(n))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))

    return {
        'n_observations': n,
        'n_events': df['event_id'].nunique(),
        'n_assets': df['symbol'].nunique(),
        'car_mean': car_mean,
        'car_std': car_std,
        'car_median': np.median(car),
        't_stat': t_stat,
        'p_value': p_value,
        'n_significant_05': (df['p_value'] < 0.05).sum(),
        'pct_significant_05': (df['p_value'] < 0.05).mean() * 100,
    }


def run_comparison_analysis(
    returns_dict: Dict[str, pd.Series],
    events_by_type: Dict[str, List[Dict]],
    analyzer: EventStudyAnalyzer
) -> Dict:
    """
    Run primary comparison: Infra_Negative vs Reg_Negative.

    This is the clean test of the enforcement capacity hypothesis.
    """
    print("\n" + "=" * 60)
    print("PRIMARY ANALYSIS: Negative Shocks Comparison")
    print("=" * 60)

    # Get events
    infra_neg = events_by_type.get('Infra_Negative', [])
    reg_neg = events_by_type.get('Reg_Negative', [])

    print(f"\nInfrastructure Failures: {len(infra_neg)} events")
    for e in infra_neg:
        print(f"  - [{e['event_id']}] {e['label']} ({e['date']})")

    print(f"\nRegulatory Enforcement: {len(reg_neg)} events")
    for e in reg_neg:
        print(f"  - [{e['event_id']}] {e['label']} ({e['date']})")

    # Run event studies
    print("\n--- Running Event Studies ---")
    infra_results = run_event_study_by_type(
        returns_dict, infra_neg, analyzer, "Infra_Negative"
    )
    reg_results = run_event_study_by_type(
        returns_dict, reg_neg, analyzer, "Reg_Negative"
    )

    # Aggregate stats
    infra_stats = compute_aggregate_stats(infra_results)
    reg_stats = compute_aggregate_stats(reg_results)

    # Difference-in-Differences
    print("\n--- Difference-in-Differences ---")

    if not infra_results.empty and not reg_results.empty:
        infra_car = infra_results['car'].values
        reg_car = reg_results['car'].values

        # Simple mean difference
        diff = infra_stats['car_mean'] - reg_stats['car_mean']

        # Welch's t-test for unequal variances
        from scipy import stats as sp_stats
        t_stat, p_value = sp_stats.ttest_ind(infra_car, reg_car, equal_var=False)

        print(f"\nInfra_Negative mean CAR: {infra_stats['car_mean']:.4f}")
        print(f"Reg_Negative mean CAR: {reg_stats['car_mean']:.4f}")
        print(f"Difference (Infra - Reg): {diff:.4f}")
        print(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.4f}")

        # Interpretation
        if p_value < 0.05:
            if diff > 0:
                print("\n>>> RESULT: Infrastructure failures show LARGER market impact")
                print("    (Supports enforcement capacity hypothesis)")
            else:
                print("\n>>> RESULT: Regulatory enforcement shows LARGER market impact")
                print("    (Contradicts enforcement capacity hypothesis - uncertainty resolution?)")
        else:
            print("\n>>> RESULT: No significant difference between event types")

    return {
        'infra_negative': {
            'results': infra_results,
            'stats': infra_stats
        },
        'reg_negative': {
            'results': reg_results,
            'stats': reg_stats
        }
    }


def run_secondary_analysis(
    returns_dict: Dict[str, pd.Series],
    events_by_type: Dict[str, List[Dict]],
    analyzer: EventStudyAnalyzer
) -> Dict:
    """
    Run secondary comparison: Infra_Positive vs Reg_Positive.

    Tests how markets respond to positive shocks by type.
    """
    print("\n" + "=" * 60)
    print("SECONDARY ANALYSIS: Positive Shocks Comparison")
    print("=" * 60)

    infra_pos = events_by_type.get('Infra_Positive', [])
    reg_pos = events_by_type.get('Reg_Positive', [])

    print(f"\nInfrastructure Upgrades: {len(infra_pos)} events")
    print(f"Regulatory Clarity: {len(reg_pos)} events")

    infra_results = run_event_study_by_type(
        returns_dict, infra_pos, analyzer, "Infra_Positive"
    )
    reg_results = run_event_study_by_type(
        returns_dict, reg_pos, analyzer, "Reg_Positive"
    )

    infra_stats = compute_aggregate_stats(infra_results)
    reg_stats = compute_aggregate_stats(reg_results)

    if infra_stats and reg_stats:
        print(f"\nInfra_Positive mean CAR: {infra_stats['car_mean']:.4f}")
        print(f"Reg_Positive mean CAR: {reg_stats['car_mean']:.4f}")

    return {
        'infra_positive': {
            'results': infra_results,
            'stats': infra_stats
        },
        'reg_positive': {
            'results': reg_results,
            'stats': reg_stats
        }
    }


def run_market_model_robustness(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict]
) -> Dict:
    """
    Run event study with market model (BTC as proxy) for robustness.

    This addresses the placebo anomaly by accounting for systematic
    bull market drift that constant mean model doesn't capture.
    """
    print("\n" + "=" * 60)
    print("MARKET MODEL ROBUSTNESS CHECK")
    print("=" * 60)
    print("Using BTC as market proxy for altcoins")

    # Run market model analysis on all events
    mm_results = run_market_model_analysis(
        returns_dict,
        events,
        market_proxy='BTC',
        event_window=(-5, 30)
    )

    if mm_results.empty:
        return {'error': 'No market model results'}

    # Summary by event type
    print("\n--- Market Model Results by Event Type ---")
    for etype in mm_results['event_type'].unique():
        subset = mm_results[mm_results['event_type'] == etype]
        car_mean = subset['car'].mean()
        n_sig = (subset['p_value'] < 0.05).sum()
        print(f"  {etype}: CAR = {car_mean:.4f}, {n_sig}/{len(subset)} significant")

    # Compare BTC (constant mean) vs altcoins (market model)
    btc_results = mm_results[mm_results['symbol'] == 'BTC']
    altcoin_results = mm_results[mm_results['symbol'] != 'BTC']

    print(f"\nBTC (constant mean): mean CAR = {btc_results['car'].mean():.4f}")
    print(f"Altcoins (market-adjusted): mean CAR = {altcoin_results['car'].mean():.4f}")

    # Get beta estimates for altcoins
    betas = mm_results[mm_results['beta'].notna()][['symbol', 'beta']].drop_duplicates()
    if not betas.empty:
        print("\n--- Market Betas (vs BTC) ---")
        for _, row in betas.iterrows():
            print(f"  {row['symbol']}: beta = {row['beta']:.3f}")

    return {
        'results': mm_results,
        'btc_mean_car': btc_results['car'].mean(),
        'altcoin_mean_car': altcoin_results['car'].mean(),
        'overall_mean_car': mm_results['car'].mean(),
        'betas': betas.to_dict('records') if not betas.empty else []
    }


def run_pre_event_analysis(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict]
) -> pd.DataFrame:
    """
    Compute pre-event CAR to detect anticipation effects.

    Regulatory events may be more anticipated than infrastructure failures.
    """
    print("\n" + "=" * 60)
    print("PRE-EVENT ANTICIPATION ANALYSIS")
    print("=" * 60)
    print("Window: [-30, -1] days before event")

    results = []

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_type = event.get('type_detailed', event.get('type', 'Unknown'))

        for symbol, returns in returns_dict.items():
            pre_result = compute_pre_event_car(
                returns, event_date,
                pre_window=(-30, -1)
            )

            if 'error' not in pre_result:
                results.append({
                    'event_id': event_id,
                    'event_date': event_date,
                    'event_type': event_type,
                    'symbol': symbol,
                    'pre_event_car': pre_result['pre_event_car'],
                    'pre_event_t_stat': pre_result['pre_event_t_stat'],
                    'pre_event_p_value': pre_result['pre_event_p_value']
                })

    df = pd.DataFrame(results)

    if df.empty:
        print("No pre-event results computed")
        return df

    # Summary by event type
    print("\n--- Pre-Event CAR by Event Type ---")
    for etype in df['event_type'].unique():
        subset = df[df['event_type'] == etype]
        car_mean = subset['pre_event_car'].mean()
        n_sig = (subset['pre_event_p_value'] < 0.05).sum()
        print(f"  {etype}: Pre-CAR = {car_mean:+.4f}, {n_sig}/{len(subset)} significant")

    # Compare infrastructure vs regulatory anticipation
    infra_types = ['Infra_Negative', 'Infra_Positive']
    reg_types = ['Reg_Negative', 'Reg_Positive']

    infra_pre = df[df['event_type'].isin(infra_types)]['pre_event_car'].values
    reg_pre = df[df['event_type'].isin(reg_types)]['pre_event_car'].values

    if len(infra_pre) > 5 and len(reg_pre) > 5:
        from scipy import stats as sp_stats
        t_stat, p_value = sp_stats.ttest_ind(infra_pre, reg_pre, equal_var=False)

        print(f"\n--- Anticipation Comparison ---")
        print(f"Infrastructure mean pre-CAR: {infra_pre.mean():+.4f}")
        print(f"Regulatory mean pre-CAR: {reg_pre.mean():+.4f}")
        print(f"Welch's t-test: t={t_stat:.4f}, p={p_value:.4f}")

        if p_value < 0.05:
            if reg_pre.mean() > infra_pre.mean():
                print("\n>>> Regulatory events show MORE anticipation than infrastructure")
            else:
                print("\n>>> Infrastructure events show MORE anticipation than regulatory")
        else:
            print("\n>>> No significant difference in anticipation between types")

    return df


def run_block_bootstrap_analysis(
    returns_dict: Dict[str, pd.Series],
    events_by_type: Dict[str, List[Dict]],
    model: ConstantMeanModel = None,
    window: Tuple[int, int] = (-5, 30)
) -> Dict:
    """
    Run event-level block bootstrap for proper inference.

    CRITICAL: This addresses the degrees-of-freedom inflation problem
    where pooling asset-events as i.i.d. observations inflates N.

    True N = number of events, not number of asset-event observations.
    """
    print("\n" + "=" * 60)
    print("EVENT-LEVEL BLOCK BOOTSTRAP ANALYSIS")
    print("=" * 60)
    print("Resampling entire events to respect cross-sectional correlation")

    if model is None:
        model = ConstantMeanModel()

    bootstrap = EventBlockBootstrap(n_bootstrap=5000)
    results = {}

    # Compute CARs by category
    for etype, events in events_by_type.items():
        if not events:
            continue

        event_cars = bootstrap.compute_event_cars(
            returns_dict, events, model, window
        )

        if len(event_cars) < 2:
            continue

        result = bootstrap.bootstrap_mean_car(event_cars)
        results[etype] = result

        print(f"\n{etype}:")
        print(f"  N events: {result.n_events}")
        print(f"  N observations: {result.n_observations}")
        print(f"  Mean CAR: {result.statistic:.4f}")
        print(f"  Bootstrap 95% CI: [{result.ci_low:.4f}, {result.ci_high:.4f}]")
        print(f"  Bootstrap SE: {result.se:.4f}")
        print(f"  Bootstrap p-value: {result.p_value:.4f}")

    # Run difference test: Infra_Negative vs Reg_Negative
    if 'Infra_Negative' in results and 'Reg_Negative' in results:
        print("\n--- DiD: Infra_Negative vs Reg_Negative ---")

        infra_events = events_by_type.get('Infra_Negative', [])
        reg_events = events_by_type.get('Reg_Negative', [])

        infra_cars = bootstrap.compute_event_cars(
            returns_dict, infra_events, model, window
        )
        reg_cars = bootstrap.compute_event_cars(
            returns_dict, reg_events, model, window
        )

        diff_result = bootstrap.bootstrap_difference_test(infra_cars, reg_cars)
        results['DiD_InfraVsReg'] = diff_result

        print(f"  Difference (Infra - Reg): {diff_result.statistic:.4f}")
        print(f"  Bootstrap 95% CI: [{diff_result.ci_low:.4f}, {diff_result.ci_high:.4f}]")
        print(f"  Bootstrap p-value: {diff_result.p_value:.4f}")

        if diff_result.ci_low <= 0 <= diff_result.ci_high:
            print("\n>>> Bootstrap CI crosses zero: NULL FINDING CONFIRMED")
        else:
            print(f"\n>>> Bootstrap CI does NOT cross zero: SIGNIFICANT at α=0.05")

    return results


def run_window_sensitivity_analysis(
    returns_dict: Dict[str, pd.Series],
    events_by_type: Dict[str, List[Dict]],
    model: ConstantMeanModel = None
) -> pd.DataFrame:
    """
    Test sensitivity to event window specification.

    Tighter windows [0,+1], [0,+3], [0,+5] are more conservative
    and less prone to confounding.
    """
    print("\n" + "=" * 60)
    print("WINDOW SENSITIVITY ANALYSIS")
    print("=" * 60)
    print("Testing if results hold across different event windows")

    windows = [
        (0, 1),     # Immediate (2 days)
        (0, 3),     # 3-day (4 days)
        (0, 5),     # 5-day (6 days)
        (-5, 30),   # Baseline (36 days)
    ]

    results = []

    for pre, post in windows:
        window_name = f"[{pre:+d}, {post:+d}]"
        print(f"\n--- Window: {window_name} ---")

        # Compute CARs directly without model's 5-day constraint
        for etype in ['Infra_Negative', 'Reg_Negative', 'Infra_Positive', 'Reg_Positive']:
            events = events_by_type.get(etype, [])
            if not events:
                continue

            cars = []
            for event in events:
                event_date = event['date']
                event_dt = pd.to_datetime(event_date)

                for symbol, returns in returns_dict.items():
                    returns_copy = returns.copy()
                    returns_copy.index = pd.to_datetime(returns_copy.index)

                    # Estimation period (use longer period for stability)
                    estimation_end = event_dt - pd.Timedelta(days=31)
                    estimation_start = estimation_end - pd.Timedelta(days=250)

                    estimation_returns = returns_copy[
                        (returns_copy.index >= estimation_start) &
                        (returns_copy.index <= estimation_end)
                    ].dropna()

                    if len(estimation_returns) < 60:  # Minimum 60 days
                        continue

                    expected_return = estimation_returns.mean()

                    # Event window
                    event_start = event_dt + pd.Timedelta(days=pre)
                    event_end = event_dt + pd.Timedelta(days=post)

                    event_returns = returns_copy[
                        (returns_copy.index >= event_start) &
                        (returns_copy.index <= event_end)
                    ].dropna()

                    # For short windows, accept fewer observations
                    min_obs = max(1, (post - pre + 1) // 2)
                    if len(event_returns) < min_obs:
                        continue

                    abnormal_returns = event_returns - expected_return
                    car = abnormal_returns.sum()
                    cars.append(car)

            if len(cars) < 5:
                print(f"  {etype}: Insufficient data ({len(cars)} observations)")
                continue

            cars = np.array(cars)
            car_mean = cars.mean()
            car_std = cars.std()
            n = len(cars)

            from scipy import stats as sp_stats
            t_stat = car_mean / (car_std / np.sqrt(n))
            p_value = 2 * (1 - sp_stats.t.cdf(abs(t_stat), n - 1))

            results.append({
                'window': window_name,
                'pre': pre,
                'post': post,
                'event_type': etype,
                'n_observations': n,
                'car_mean': car_mean,
                'car_std': car_std,
                't_stat': t_stat,
                'p_value': p_value,
                'significant_05': p_value < 0.05
            })

            sig_flag = '*' if p_value < 0.05 else ''
            print(f"  {etype}: CAR={car_mean:+.4f}, t={t_stat:.2f}, p={p_value:.3f}{sig_flag}")

    df = pd.DataFrame(results)

    # Summary: consistency check
    print("\n--- Window Consistency Check ---")
    for etype in ['Infra_Negative', 'Reg_Negative']:
        subset = df[df['event_type'] == etype]
        if subset.empty:
            continue
        signs = np.sign(subset['car_mean'].values)
        consistent = len(set(signs)) == 1
        print(f"  {etype}: {'CONSISTENT' if consistent else 'INCONSISTENT'} sign across windows")

    return df


def run_loo_major_events(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    model: ConstantMeanModel = None,
    major_event_labels: List[str] = None
) -> Dict:
    """
    Leave-one-out analysis specifically for major events (FTX, Terra).

    Tests whether results are driven by outliers.
    """
    print("\n" + "=" * 60)
    print("LEAVE-ONE-OUT: MAJOR EVENTS")
    print("=" * 60)
    print("Testing if FTX/Terra dominate Infra_Negative category")

    if model is None:
        model = ConstantMeanModel()

    if major_event_labels is None:
        major_event_labels = ['FTX bankrupt', 'Terra/UST crash']

    # Filter to Infra_Negative events
    infra_neg = [e for e in events if e.get('type_detailed') == 'Infra_Negative']

    if not infra_neg:
        print("No Infra_Negative events found")
        return {}

    # Baseline CAR
    baseline_cars = []
    for event in infra_neg:
        event_date = event['date']
        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, (-5, 30))
            if 'error' not in result:
                baseline_cars.append(result['car'])

    baseline_mean = np.mean(baseline_cars) if baseline_cars else np.nan
    print(f"\nBaseline (all events): mean CAR = {baseline_mean:.4f}")

    results = {'baseline': baseline_mean, 'excluded': {}}

    # LOO for each major event
    for label in major_event_labels:
        excluded_event = None
        for e in infra_neg:
            if label.lower() in e.get('label', '').lower():
                excluded_event = e
                break

        if excluded_event is None:
            print(f"\n{label}: NOT FOUND in Infra_Negative events")
            continue

        remaining = [e for e in infra_neg if e['event_id'] != excluded_event['event_id']]

        loo_cars = []
        for event in remaining:
            event_date = event['date']
            for symbol, returns in returns_dict.items():
                result = model.compute_abnormal_returns(returns, event_date, (-5, 30))
                if 'error' not in result:
                    loo_cars.append(result['car'])

        loo_mean = np.mean(loo_cars) if loo_cars else np.nan
        change = loo_mean - baseline_mean

        results['excluded'][label] = {
            'event_id': excluded_event['event_id'],
            'event_date': excluded_event['date'],
            'loo_mean': loo_mean,
            'change_from_baseline': change,
            'sign_flip': np.sign(loo_mean) != np.sign(baseline_mean) if not np.isnan(loo_mean) else False,
            'magnitude_change_pct': (change / abs(baseline_mean)) * 100 if baseline_mean != 0 else np.nan
        }

        sign_flag = '⚠️ SIGN FLIP' if results['excluded'][label]['sign_flip'] else ''
        print(f"\nExcluding {label} (ID={excluded_event['event_id']}):")
        print(f"  LOO mean CAR: {loo_mean:.4f}")
        print(f"  Change from baseline: {change:+.4f} ({results['excluded'][label]['magnitude_change_pct']:+.1f}%)")
        if sign_flag:
            print(f"  {sign_flag}")

    # Summary
    any_sign_flip = any(r['sign_flip'] for r in results['excluded'].values())
    print("\n--- LOO Summary ---")
    if any_sign_flip:
        print("⚠️ WARNING: Excluding a major event FLIPS the sign of Infra_Negative CAR")
        print("   Results may be driven by outliers - discuss as limitation")
    else:
        print("✓ Results ROBUST to major event exclusion")

    return results


def run_ew_market_robustness(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict]
) -> Dict:
    """
    Run analysis with equal-weighted market model for robustness.

    Addresses concern that BTC as market proxy may be problematic:
    1. BTC is in our sample (mechanical correlation)
    2. BTC dominates crypto market cap (~50%)
    """
    print("\n" + "=" * 60)
    print("EQUAL-WEIGHTED MARKET MODEL ROBUSTNESS")
    print("=" * 60)
    print("Using leave-one-out EW index instead of BTC as market proxy")

    ew_results = run_ew_market_analysis(
        returns_dict,
        events,
        event_window=(-5, 30)
    )

    if ew_results.empty:
        return {'error': 'No EW market results'}

    # Summary by event type
    print("\n--- EW Market Model Results by Event Type ---")
    for etype in ew_results['event_type'].unique():
        subset = ew_results[ew_results['event_type'] == etype]
        car_mean = subset['car'].mean()
        n_sig = (subset['p_value'] < 0.05).sum()
        print(f"  {etype}: CAR = {car_mean:.4f}, {n_sig}/{len(subset)} significant")

    # Average betas
    betas = ew_results[ew_results['beta'].notna()][['symbol', 'beta']].drop_duplicates()
    if not betas.empty:
        print("\n--- Average Betas (vs EW index) ---")
        for _, row in betas.iterrows():
            print(f"  {row['symbol']}: beta = {row['beta']:.3f}")

    return {
        'results': ew_results,
        'summary_by_type': ew_results.groupby('event_type')['car'].agg(['mean', 'std', 'count']).to_dict()
    }


def run_reg_positive_diagnostic(
    returns_dict: Dict[str, pd.Series],
    events_by_type: Dict[str, List[Dict]],
    model: ConstantMeanModel = None
) -> pd.DataFrame:
    """
    Diagnose the Reg_Positive anomaly.

    Reg_Positive shows negative CAR despite positive valence.
    Hypotheses:
    1. Pre-emption: Markets front-run, then "sell the news"
    2. Contamination: Coincides with negative shocks
    3. Classification error: Some events misclassified
    """
    print("\n" + "=" * 60)
    print("REG_POSITIVE DIAGNOSTIC")
    print("=" * 60)
    print("Investigating why positive regulatory news shows negative returns")

    if model is None:
        model = ConstantMeanModel()

    reg_pos = events_by_type.get('Reg_Positive', [])
    if not reg_pos:
        print("No Reg_Positive events")
        return pd.DataFrame()

    results = []

    for event in reg_pos:
        event_id = event['event_id']
        event_date = event['date']
        event_label = event.get('label', '')
        event_title = event.get('title', '')
        overlaps = event.get('overlapping_events', [])

        event_cars = []
        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, (-5, 30))
            if 'error' not in result:
                event_cars.append(result['car'])

        mean_car = np.mean(event_cars) if event_cars else np.nan

        results.append({
            'event_id': event_id,
            'date': event_date,
            'label': event_label,
            'title': event_title,
            'mean_car': mean_car,
            'n_assets': len(event_cars),
            'has_overlaps': len(overlaps) > 0,
            'overlap_ids': [o['event_id'] for o in overlaps] if overlaps else []
        })

    df = pd.DataFrame(results)

    print("\n--- Per-Event CAR for Reg_Positive ---")
    for _, row in df.iterrows():
        overlap_flag = ' [OVERLAP]' if row['has_overlaps'] else ''
        print(f"  [{row['event_id']}] {row['label']}: CAR={row['mean_car']:+.4f}{overlap_flag}")

    # Check for "sell the news" pattern
    positive_events = df[df['mean_car'] > 0]
    negative_events = df[df['mean_car'] < 0]

    print(f"\n--- Summary ---")
    print(f"  Events with positive CAR: {len(positive_events)}")
    print(f"  Events with negative CAR: {len(negative_events)}")
    print(f"  Events with overlaps: {df['has_overlaps'].sum()}")

    if len(negative_events) > len(positive_events):
        print("\n>>> FINDING: Majority of Reg_Positive events show NEGATIVE returns")
        print("    Possible explanations:")
        print("    1. 'Sell the news' - markets front-run clarity, then sell")
        print("    2. Event window captures unrelated negative shocks")
        print("    3. Regulatory clarity reduces speculative premium")

    return df


def run_placebo_analysis(
    returns_dict: Dict[str, pd.Series],
    robustness_checker: RobustnessChecker
) -> Dict:
    """
    Run placebo test to establish null distribution.

    The positive placebo CAR (+4.06%, p=0.013) from prior analysis
    suggests constant mean model may not account for bull market drift.
    """
    print("\n" + "=" * 60)
    print("PLACEBO TEST")
    print("=" * 60)

    placebo_result = robustness_checker.placebo_test(
        returns_dict,
        n_placebo=200  # More samples for stability
    )

    print(f"\nPlacebo CAR mean: {placebo_result.car_mean:.4f}")
    print(f"Placebo CAR std: {placebo_result.car_std:.4f}")
    print(f"t-stat: {placebo_result.t_stat:.4f}")
    print(f"p-value: {placebo_result.p_value:.4f}")

    if placebo_result.p_value < 0.05:
        print("\n>>> WARNING: Placebo test SIGNIFICANT")
        print("    Suggests systematic drift not captured by constant mean model")
        print("    Consider market-adjusted returns for robustness")
    else:
        print("\n>>> Placebo test non-significant (as expected)")

    return {
        'car_mean': placebo_result.car_mean,
        'car_std': placebo_result.car_std,
        't_stat': placebo_result.t_stat,
        'p_value': placebo_result.p_value,
        'percentiles': placebo_result.details.get('placebo_cars_percentiles', {})
    }


def save_results(
    primary: Dict,
    secondary: Dict,
    placebo: Dict,
    output_dir: Path
):
    """Save all results to CSV/JSON files."""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Primary comparison results
    if not primary['infra_negative']['results'].empty:
        f = output_dir / 'infra_negative_results.csv'
        primary['infra_negative']['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    if not primary['reg_negative']['results'].empty:
        f = output_dir / 'reg_negative_results.csv'
        primary['reg_negative']['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    # Secondary comparison results
    if not secondary['infra_positive']['results'].empty:
        f = output_dir / 'infra_positive_results.csv'
        secondary['infra_positive']['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    if not secondary['reg_positive']['results'].empty:
        f = output_dir / 'reg_positive_results.csv'
        secondary['reg_positive']['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    # Summary statistics
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'primary_comparison': {
            'infra_negative': primary['infra_negative']['stats'],
            'reg_negative': primary['reg_negative']['stats'],
        },
        'secondary_comparison': {
            'infra_positive': secondary['infra_positive']['stats'],
            'reg_positive': secondary['reg_positive']['stats'],
        },
        'placebo_test': placebo
    }

    f = output_dir / 'reclassified_summary.json'
    with open(f, 'w') as fp:
        json.dump(summary, fp, indent=2, default=str)
    print(f"  Saved: {f}")


def main():
    """Run complete reclassified analysis pipeline."""
    print("\n" + "=" * 70)
    print("SENTIMENT WITHOUT STRUCTURE - RECLASSIFIED ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAddressing peer review feedback:")
    print("  - Event conflation: Using 4-category classification")
    print("  - Primary test: Negative-valence events only")
    print("  - Block bootstrap for proper inference (event-level clustering)")
    print("  - Window sensitivity: [0,+1] to [-5,+30]")
    print("  - Leave-one-out on FTX/Terra")
    print("  - EW market proxy robustness")
    print("=" * 70)

    # Initialize
    fetcher = BinanceDataFetcher()
    analyzer = EventStudyAnalyzer()
    robustness_checker = RobustnessChecker()
    model = ConstantMeanModel()

    # Load reclassified events
    events, events_by_type = load_reclassified_events()

    # Load returns data
    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]  # BTC, ETH, SOL, ADA
    returns_dict = load_returns_data(fetcher, assets)

    if not returns_dict:
        print("ERROR: No return data loaded")
        return

    # Primary analysis: Negative shocks
    primary_results = run_comparison_analysis(
        returns_dict, events_by_type, analyzer
    )

    # Secondary analysis: Positive shocks
    secondary_results = run_secondary_analysis(
        returns_dict, events_by_type, analyzer
    )

    # ==== NEW: STATISTICAL RIGOR IMPROVEMENTS ====

    # 1. EVENT-LEVEL BLOCK BOOTSTRAP (CRITICAL)
    bootstrap_results = run_block_bootstrap_analysis(
        returns_dict, events_by_type, model
    )

    # 2. WINDOW SENSITIVITY (Tighter windows)
    window_sensitivity_results = run_window_sensitivity_analysis(
        returns_dict, events_by_type, model
    )

    # 3. LEAVE-ONE-OUT ON MAJOR EVENTS (FTX, Terra)
    all_events = [e for e in events if e.get('include_in_reanalysis', True)]
    loo_major_results = run_loo_major_events(
        returns_dict, all_events, model
    )

    # 4. EQUAL-WEIGHTED MARKET MODEL ROBUSTNESS
    ew_market_results = run_ew_market_robustness(
        returns_dict, all_events
    )

    # 5. REG_POSITIVE DIAGNOSTIC
    reg_pos_diagnostic = run_reg_positive_diagnostic(
        returns_dict, events_by_type, model
    )

    # ==== EXISTING ROBUSTNESS CHECKS ====

    # Market model (BTC proxy) robustness
    market_model_results = run_market_model_robustness(
        returns_dict, all_events
    )

    # Pre-event anticipation analysis
    pre_event_results = run_pre_event_analysis(
        returns_dict, all_events
    )

    # Placebo test
    placebo_results = run_placebo_analysis(
        returns_dict, robustness_checker
    )

    # ==== SAVE RESULTS ====
    save_results(
        primary_results,
        secondary_results,
        placebo_results,
        config.TABLES_DIR
    )

    # Save additional results
    if 'results' in market_model_results and not market_model_results['results'].empty:
        f = config.TABLES_DIR / 'market_model_results.csv'
        market_model_results['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    if not pre_event_results.empty:
        f = config.TABLES_DIR / 'pre_event_results.csv'
        pre_event_results.to_csv(f, index=False)
        print(f"  Saved: {f}")

    # Save new robustness results
    if not window_sensitivity_results.empty:
        f = config.TABLES_DIR / 'window_sensitivity_results.csv'
        window_sensitivity_results.to_csv(f, index=False)
        print(f"  Saved: {f}")

    if not reg_pos_diagnostic.empty:
        f = config.TABLES_DIR / 'reg_positive_diagnostic.csv'
        reg_pos_diagnostic.to_csv(f, index=False)
        print(f"  Saved: {f}")

    if 'results' in ew_market_results and not ew_market_results['results'].empty:
        f = config.TABLES_DIR / 'ew_market_results.csv'
        ew_market_results['results'].to_csv(f, index=False)
        print(f"  Saved: {f}")

    # Save bootstrap summary
    bootstrap_summary = {}
    for etype, result in bootstrap_results.items():
        if hasattr(result, 'statistic'):
            bootstrap_summary[etype] = {
                'n_events': result.n_events,
                'n_observations': result.n_observations,
                'mean_car': result.statistic,
                'ci_low': result.ci_low,
                'ci_high': result.ci_high,
                'se': result.se,
                'p_value': result.p_value
            }
    if bootstrap_summary:
        f = config.TABLES_DIR / 'bootstrap_summary.json'
        with open(f, 'w') as fp:
            json.dump(bootstrap_summary, fp, indent=2, default=str)
        print(f"  Saved: {f}")

    # Save LOO major events
    if loo_major_results:
        f = config.TABLES_DIR / 'loo_major_events.json'
        with open(f, 'w') as fp:
            json.dump(loo_major_results, fp, indent=2, default=str)
        print(f"  Saved: {f}")

    # ==== FINAL SUMMARY ====
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    # KP t-test results (pooled - for reference)
    infra_car = primary_results['infra_negative']['stats'].get('car_mean', np.nan)
    reg_car = primary_results['reg_negative']['stats'].get('car_mean', np.nan)
    print("\n1. POOLED T-TEST (reference only - degrees of freedom inflated):")
    print(f"   Infra_Negative mean CAR: {infra_car:.4f}")
    print(f"   Reg_Negative mean CAR: {reg_car:.4f}")
    print(f"   Difference: {infra_car - reg_car:.4f}")

    # Bootstrap results (PROPER INFERENCE)
    print("\n2. BLOCK BOOTSTRAP (proper event-level inference):")
    if 'Infra_Negative' in bootstrap_results:
        r = bootstrap_results['Infra_Negative']
        print(f"   Infra_Negative: {r.statistic:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] p={r.p_value:.4f}")
    if 'Reg_Negative' in bootstrap_results:
        r = bootstrap_results['Reg_Negative']
        print(f"   Reg_Negative: {r.statistic:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] p={r.p_value:.4f}")
    if 'DiD_InfraVsReg' in bootstrap_results:
        r = bootstrap_results['DiD_InfraVsReg']
        crosses_zero = r.ci_low <= 0 <= r.ci_high
        print(f"   DiD (Infra-Reg): {r.statistic:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] p={r.p_value:.4f}")
        print(f"   >>> CI crosses zero: {crosses_zero} → {'NULL FINDING' if crosses_zero else 'SIGNIFICANT'}")

    # LOO major events
    print("\n3. LEAVE-ONE-OUT (FTX/Terra):")
    if loo_major_results:
        any_flip = any(r['sign_flip'] for r in loo_major_results.get('excluded', {}).values())
        print(f"   Baseline Infra_Negative CAR: {loo_major_results.get('baseline', np.nan):.4f}")
        for label, info in loo_major_results.get('excluded', {}).items():
            flip_flag = " ⚠️ SIGN FLIP" if info['sign_flip'] else ""
            print(f"   Excl. {label}: {info['loo_mean']:.4f} ({info['magnitude_change_pct']:+.1f}%){flip_flag}")
        print(f"   >>> Robust to outliers: {not any_flip}")

    # Window sensitivity
    print("\n4. WINDOW SENSITIVITY:")
    if not window_sensitivity_results.empty:
        for etype in ['Infra_Negative', 'Reg_Negative']:
            subset = window_sensitivity_results[window_sensitivity_results['event_type'] == etype]
            if not subset.empty:
                signs = np.sign(subset['car_mean'].values)
                consistent = len(set(signs)) == 1
                print(f"   {etype}: {'CONSISTENT' if consistent else 'INCONSISTENT'} sign across windows")

    # Placebo
    print("\n5. PLACEBO TEST:")
    print(f"   p-value: {placebo_results['p_value']:.4f}")
    if placebo_results['p_value'] < 0.05:
        print("   ⚠️ WARNING: Significant - systematic drift may bias results")

    # Reg_Positive anomaly
    print("\n6. REG_POSITIVE ANOMALY:")
    if not reg_pos_diagnostic.empty:
        pos_count = (reg_pos_diagnostic['mean_car'] > 0).sum()
        neg_count = (reg_pos_diagnostic['mean_car'] < 0).sum()
        print(f"   Events with positive CAR: {pos_count}")
        print(f"   Events with negative CAR: {neg_count}")
        if neg_count > pos_count:
            print("   >>> 'Sell the news' pattern or contamination likely")


if __name__ == "__main__":
    main()
