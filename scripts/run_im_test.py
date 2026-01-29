#!/usr/bin/env python3
"""
Ibragimov-Müller Few-Cluster Test
=================================

Implements the Ibragimov-Müller (2010) t-statistic based inference for
few-cluster settings. With only 8 vs 7 events, standard bootstrap may
be unreliable. The IM test uses a simple t-test on cluster means.

Reference:
    Ibragimov, R., & Müller, U. K. (2010). t-Statistic based correlation
    and heterogeneity robust inference. Journal of Business & Economic
    Statistics, 28(4), 453-468.

Usage:
    python scripts/run_im_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.event_study import ConstantMeanModel


def load_events_by_type() -> dict:
    """Load events grouped by type_detailed."""
    events_file = config.DATA_DIR / 'events_reclassified.json'

    with open(events_file) as f:
        data = json.load(f)

    events_by_type = {}
    for e in data['events']:
        if not e.get('include_in_reanalysis', True):
            continue
        if not e.get('meets_impact_threshold', False):
            continue
        if not e.get('has_sufficient_estimation_data', True):
            continue

        etype = e.get('type_detailed', e['type'])
        if etype not in events_by_type:
            events_by_type[etype] = []
        events_by_type[etype].append(e)

    return events_by_type


def compute_event_level_cars(
    returns_dict: dict,
    events: list,
    model: ConstantMeanModel,
    window: tuple = (-5, 30)
) -> list:
    """
    Compute mean CAR for each event (averaging across assets).

    Returns list of event-level mean CARs.
    """
    event_cars = []

    for event in events:
        event_date = event['date']
        asset_cars = []

        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, window)
            if 'error' not in result:
                asset_cars.append(result['car'])

        if asset_cars:
            event_cars.append({
                'event_id': event['event_id'],
                'event_date': event_date,
                'event_name': event.get('label', ''),
                'mean_car': np.mean(asset_cars),
                'n_assets': len(asset_cars)
            })

    return event_cars


def ibragimov_muller_test(group1_means: np.ndarray, group2_means: np.ndarray) -> dict:
    """
    Ibragimov-Müller t-test on group means.

    The IM test is simply a t-test comparing means of cluster-level statistics.
    With few clusters (events), this is more robust than pooled approaches.

    Args:
        group1_means: Array of event-level mean CARs for group 1
        group2_means: Array of event-level mean CARs for group 2

    Returns:
        Dict with test statistics
    """
    n1 = len(group1_means)
    n2 = len(group2_means)

    mean1 = np.mean(group1_means)
    mean2 = np.mean(group2_means)

    var1 = np.var(group1_means, ddof=1)
    var2 = np.var(group2_means, ddof=1)

    # Welch's t-test (unequal variances)
    se_diff = np.sqrt(var1/n1 + var2/n2)
    t_stat = (mean1 - mean2) / se_diff

    # Welch-Satterthwaite degrees of freedom
    df_num = (var1/n1 + var2/n2)**2
    df_denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = df_num / df_denom

    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Confidence interval for difference
    t_crit = stats.t.ppf(0.975, df)
    ci_low = (mean1 - mean2) - t_crit * se_diff
    ci_high = (mean1 - mean2) + t_crit * se_diff

    return {
        'n_group1': n1,
        'n_group2': n2,
        'mean_group1': mean1,
        'mean_group2': mean2,
        'difference': mean1 - mean2,
        'se_difference': se_diff,
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant_05': p_value < 0.05
    }


def main():
    print("\n" + "=" * 70)
    print("IBRAGIMOV-MÜLLER FEW-CLUSTER TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nRationale: With only 8 vs 7 events, standard inference may be unreliable.")
    print("The IM test uses a simple t-test on event-level means for robustness.")
    print("=" * 70)

    # Load data
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()

    events_by_type = load_events_by_type()

    print("\n--- Event Counts ---")
    for etype, events in sorted(events_by_type.items()):
        print(f"  {etype}: {len(events)} events")

    # Load returns
    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]  # BTC, ETH, SOL, ADA
    returns_dict = {}

    for symbol in assets:
        try:
            df = fetcher.fetch_ohlcv(symbol, config.START_DATE, config.END_DATE)
            if not df.empty and 'returns' in df.columns:
                returns_dict[symbol] = df['returns'].dropna()
                print(f"  {symbol}: {len(returns_dict[symbol])} returns")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    # Compute event-level CARs
    print("\n--- Computing Event-Level CARs ---")

    infra_neg_events = events_by_type.get('Infra_Negative', [])
    reg_neg_events = events_by_type.get('Reg_Negative', [])

    infra_neg_cars = compute_event_level_cars(returns_dict, infra_neg_events, model)
    reg_neg_cars = compute_event_level_cars(returns_dict, reg_neg_events, model)

    print(f"\nInfra_Negative: {len(infra_neg_cars)} events with valid CARs")
    for e in infra_neg_cars:
        print(f"  [{e['event_id']}] {e['event_name']}: CAR = {e['mean_car']:+.4f}")

    print(f"\nReg_Negative: {len(reg_neg_cars)} events with valid CARs")
    for e in reg_neg_cars:
        print(f"  [{e['event_id']}] {e['event_name']}: CAR = {e['mean_car']:+.4f}")

    # Run IM test
    print("\n" + "=" * 70)
    print("IBRAGIMOV-MÜLLER TEST RESULTS")
    print("=" * 70)

    infra_means = np.array([e['mean_car'] for e in infra_neg_cars])
    reg_means = np.array([e['mean_car'] for e in reg_neg_cars])

    im_result = ibragimov_muller_test(infra_means, reg_means)

    print(f"\nInfra_Negative (N={im_result['n_group1']}):")
    print(f"  Mean of event-level CARs: {im_result['mean_group1']:.4f}")

    print(f"\nReg_Negative (N={im_result['n_group2']}):")
    print(f"  Mean of event-level CARs: {im_result['mean_group2']:.4f}")

    print(f"\nDifference (Infra - Reg): {im_result['difference']:+.4f}")
    print(f"SE of difference: {im_result['se_difference']:.4f}")
    print(f"t-statistic: {im_result['t_statistic']:.4f}")
    print(f"Degrees of freedom: {im_result['df']:.2f}")
    print(f"p-value (two-tailed): {im_result['p_value']:.4f}")
    print(f"95% CI: [{im_result['ci_low']:.4f}, {im_result['ci_high']:.4f}]")

    if im_result['significant_05']:
        print("\n>>> SIGNIFICANT at α=0.05")
    else:
        print("\n>>> NOT SIGNIFICANT at α=0.05")
        print("    CI crosses zero - cannot reject H0: no difference")

    # Compare to bootstrap
    print("\n--- Comparison with Bootstrap ---")
    bootstrap_file = config.TABLES_DIR / 'bootstrap_summary.json'
    if bootstrap_file.exists():
        with open(bootstrap_file) as f:
            bootstrap = json.load(f)

        if 'DiD_InfraVsReg' in bootstrap:
            bs = bootstrap['DiD_InfraVsReg']
            print(f"Bootstrap p-value: {bs['p_value']:.4f}")
            print(f"Bootstrap 95% CI: [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]")
            print(f"IM test p-value: {im_result['p_value']:.4f}")
            print(f"IM test 95% CI: [{im_result['ci_low']:.4f}, {im_result['ci_high']:.4f}]")

            if (im_result['p_value'] > 0.05) == (bs['p_value'] > 0.05):
                print("\n>>> CONVERGENT: Both methods reach same conclusion")
            else:
                print("\n>>> DIVERGENT: Methods disagree - investigate further")

    # Save results
    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'Ibragimov-Muller t-test on event-level means',
        'reference': 'Ibragimov & Muller (2010) JBES',
        'infra_negative': {
            'n_events': int(im_result['n_group1']),
            'event_cars': [e['mean_car'] for e in infra_neg_cars],
            'mean_car': im_result['mean_group1']
        },
        'reg_negative': {
            'n_events': int(im_result['n_group2']),
            'event_cars': [e['mean_car'] for e in reg_neg_cars],
            'mean_car': im_result['mean_group2']
        },
        'difference_test': {
            'difference': im_result['difference'],
            'se': im_result['se_difference'],
            't_statistic': im_result['t_statistic'],
            'df': im_result['df'],
            'p_value': im_result['p_value'],
            'ci_low': im_result['ci_low'],
            'ci_high': im_result['ci_high'],
            'significant_05': bool(im_result['significant_05'])
        }
    }

    output_file = config.TABLES_DIR / 'im_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
