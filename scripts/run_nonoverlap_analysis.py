#!/usr/bin/env python3
"""
Non-Overlap Robustness Analysis
===============================

Addresses reviewer concern about overlapping event windows confounding inference.

Two approaches:
1. Filter to events with 30+ day separation from other events
2. Use shorter [0,+5] window that reduces overlap contamination

Usage:
    python scripts/run_nonoverlap_analysis.py
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
from src.robustness import EventBlockBootstrap


def load_events_with_overlap_info() -> tuple:
    """Load events with overlap information."""
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

        # Add overlap flag
        e['has_overlap'] = len(e.get('overlapping_events', [])) > 0
        events_by_type[etype].append(e)

    return events_by_type, data.get('overlapping_pairs', [])


def filter_nonoverlapping_events(events: list) -> list:
    """Return only events with no overlapping events within 30 days."""
    return [e for e in events if not e.get('has_overlap', False)]


def compute_event_cars(
    returns_dict: dict,
    events: list,
    model: ConstantMeanModel,
    window: tuple
) -> list:
    """Compute CARs for events using specified window."""
    event_cars = []

    for event in events:
        event_date = event['date']
        event_dt = pd.to_datetime(event_date)
        asset_cars = []

        for symbol, returns in returns_dict.items():
            returns_copy = returns.copy()
            returns_copy.index = pd.to_datetime(returns_copy.index)

            # Estimation period
            estimation_end = event_dt - pd.Timedelta(days=31)
            estimation_start = estimation_end - pd.Timedelta(days=250)

            estimation_returns = returns_copy[
                (returns_copy.index >= estimation_start) &
                (returns_copy.index <= estimation_end)
            ].dropna()

            if len(estimation_returns) < 60:
                continue

            expected_return = estimation_returns.mean()

            # Event window
            event_start = event_dt + pd.Timedelta(days=window[0])
            event_end = event_dt + pd.Timedelta(days=window[1])

            event_returns = returns_copy[
                (returns_copy.index >= event_start) &
                (returns_copy.index <= event_end)
            ].dropna()

            min_obs = max(1, (window[1] - window[0] + 1) // 2)
            if len(event_returns) < min_obs:
                continue

            abnormal_returns = event_returns - expected_return
            car = abnormal_returns.sum()
            asset_cars.append(car)

        if asset_cars:
            event_cars.append({
                'event_id': event['event_id'],
                'event_date': event_date,
                'event_name': event.get('label', ''),
                'mean_car': np.mean(asset_cars),
                'n_assets': len(asset_cars),
                'has_overlap': event.get('has_overlap', False)
            })

    return event_cars


def run_comparison_test(group1_cars: list, group2_cars: list, label: str) -> dict:
    """Run Welch's t-test and compute summary statistics."""
    cars1 = np.array([e['mean_car'] for e in group1_cars])
    cars2 = np.array([e['mean_car'] for e in group2_cars])

    n1, n2 = len(cars1), len(cars2)
    mean1, mean2 = np.mean(cars1), np.mean(cars2)
    diff = mean1 - mean2

    if n1 < 2 or n2 < 2:
        return {
            'label': label,
            'n_group1': n1,
            'n_group2': n2,
            'error': 'Insufficient data for test'
        }

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(cars1, cars2, equal_var=False)

    # Confidence interval for difference
    se_diff = np.sqrt(np.var(cars1, ddof=1)/n1 + np.var(cars2, ddof=1)/n2)

    # Welch-Satterthwaite df
    var1, var2 = np.var(cars1, ddof=1), np.var(cars2, ddof=1)
    df_num = (var1/n1 + var2/n2)**2
    df_denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
    df = df_num / df_denom

    t_crit = stats.t.ppf(0.975, df)
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    return {
        'label': label,
        'n_group1': n1,
        'n_group2': n2,
        'mean_group1': mean1,
        'mean_group2': mean2,
        'difference': diff,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'significant_05': bool(p_value < 0.05),
        'ci_crosses_zero': bool(ci_low <= 0 <= ci_high)
    }


def main():
    print("\n" + "=" * 70)
    print("NON-OVERLAP ROBUSTNESS ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAddressing reviewer concern about overlapping event windows.")
    print("Two approaches:")
    print("  1. Filter to independent events (no overlaps within 30 days)")
    print("  2. Use shorter [0,+5] window to reduce contamination")
    print("=" * 70)

    # Load data
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()

    events_by_type, overlap_pairs = load_events_with_overlap_info()

    print(f"\nOverlapping pairs found: {len(overlap_pairs)}")

    # Load returns
    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]
    returns_dict = {}

    for symbol in assets:
        try:
            df = fetcher.fetch_ohlcv(symbol, config.START_DATE, config.END_DATE)
            if not df.empty and 'returns' in df.columns:
                returns_dict[symbol] = df['returns'].dropna()
                print(f"  {symbol}: {len(returns_dict[symbol])} returns")
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    results = {'analysis_date': datetime.now().isoformat()}

    # ========================================
    # APPROACH 1: Filter to non-overlapping events
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 1: NON-OVERLAPPING EVENTS ONLY")
    print("=" * 70)

    infra_neg_all = events_by_type.get('Infra_Negative', [])
    reg_neg_all = events_by_type.get('Reg_Negative', [])

    infra_neg_independent = filter_nonoverlapping_events(infra_neg_all)
    reg_neg_independent = filter_nonoverlapping_events(reg_neg_all)

    print(f"\nInfra_Negative: {len(infra_neg_all)} total → {len(infra_neg_independent)} independent")
    for e in infra_neg_all:
        overlap_flag = " [OVERLAP]" if e['has_overlap'] else ""
        incl_flag = "✓" if not e['has_overlap'] else "✗"
        print(f"  [{incl_flag}] {e['label']}{overlap_flag}")

    print(f"\nReg_Negative: {len(reg_neg_all)} total → {len(reg_neg_independent)} independent")
    for e in reg_neg_all:
        overlap_flag = " [OVERLAP]" if e['has_overlap'] else ""
        incl_flag = "✓" if not e['has_overlap'] else "✗"
        print(f"  [{incl_flag}] {e['label']}{overlap_flag}")

    # Compute CARs for independent events with main window
    infra_cars_indep = compute_event_cars(returns_dict, infra_neg_independent, model, (-5, 30))
    reg_cars_indep = compute_event_cars(returns_dict, reg_neg_independent, model, (-5, 30))

    if len(infra_cars_indep) >= 2 and len(reg_cars_indep) >= 2:
        test_indep = run_comparison_test(infra_cars_indep, reg_cars_indep, "Independent events [-5,+30]")

        print(f"\n--- Results: Independent Events Only ---")
        print(f"Infra_Negative (N={test_indep['n_group1']}): mean CAR = {test_indep['mean_group1']:.4f}")
        print(f"Reg_Negative (N={test_indep['n_group2']}): mean CAR = {test_indep['mean_group2']:.4f}")
        print(f"Difference: {test_indep['difference']:+.4f}")
        print(f"p-value: {test_indep['p_value']:.4f}")
        print(f"95% CI: [{test_indep['ci_low']:.4f}, {test_indep['ci_high']:.4f}]")

        if test_indep['ci_crosses_zero']:
            print("\n>>> CI crosses zero: NULL FINDING CONFIRMED with clean sample")
        else:
            print("\n>>> CI does NOT cross zero: SIGNIFICANT with clean sample")

        results['approach1_independent'] = test_indep
    else:
        print("\n>>> Insufficient independent events for test")
        results['approach1_independent'] = {'error': 'Insufficient data'}

    # ========================================
    # APPROACH 2: Shorter [0,+5] window on all events
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 2: SHORT WINDOW [0,+5] ON ALL EVENTS")
    print("=" * 70)
    print("Rationale: 6-day window reduces overlap contamination")

    infra_cars_short = compute_event_cars(returns_dict, infra_neg_all, model, (0, 5))
    reg_cars_short = compute_event_cars(returns_dict, reg_neg_all, model, (0, 5))

    print(f"\nInfra_Negative [0,+5]: {len(infra_cars_short)} events")
    for e in infra_cars_short:
        print(f"  [{e['event_id']}] {e['event_name']}: CAR = {e['mean_car']:+.4f}")

    print(f"\nReg_Negative [0,+5]: {len(reg_cars_short)} events")
    for e in reg_cars_short:
        print(f"  [{e['event_id']}] {e['event_name']}: CAR = {e['mean_car']:+.4f}")

    if len(infra_cars_short) >= 2 and len(reg_cars_short) >= 2:
        test_short = run_comparison_test(infra_cars_short, reg_cars_short, "All events [0,+5]")

        print(f"\n--- Results: Short Window ---")
        print(f"Infra_Negative (N={test_short['n_group1']}): mean CAR = {test_short['mean_group1']:.4f}")
        print(f"Reg_Negative (N={test_short['n_group2']}): mean CAR = {test_short['mean_group2']:.4f}")
        print(f"Difference: {test_short['difference']:+.4f}")
        print(f"p-value: {test_short['p_value']:.4f}")
        print(f"95% CI: [{test_short['ci_low']:.4f}, {test_short['ci_high']:.4f}]")

        if test_short['ci_crosses_zero']:
            print("\n>>> CI crosses zero: NULL FINDING CONFIRMED with short window")
        else:
            print("\n>>> CI does NOT cross zero: SIGNIFICANT with short window")

        results['approach2_short_window'] = test_short
    else:
        print("\n>>> Insufficient data for short window test")
        results['approach2_short_window'] = {'error': 'Insufficient data'}

    # ========================================
    # APPROACH 3: Combined - Independent events with short window
    # ========================================
    print("\n" + "=" * 70)
    print("APPROACH 3: INDEPENDENT EVENTS + SHORT WINDOW [0,+5]")
    print("=" * 70)
    print("Most conservative specification")

    infra_cars_both = compute_event_cars(returns_dict, infra_neg_independent, model, (0, 5))
    reg_cars_both = compute_event_cars(returns_dict, reg_neg_independent, model, (0, 5))

    if len(infra_cars_both) >= 2 and len(reg_cars_both) >= 2:
        test_both = run_comparison_test(infra_cars_both, reg_cars_both, "Independent events [0,+5]")

        print(f"\n--- Results: Most Conservative ---")
        print(f"Infra_Negative (N={test_both['n_group1']}): mean CAR = {test_both['mean_group1']:.4f}")
        print(f"Reg_Negative (N={test_both['n_group2']}): mean CAR = {test_both['mean_group2']:.4f}")
        print(f"Difference: {test_both['difference']:+.4f}")
        print(f"p-value: {test_both['p_value']:.4f}")
        print(f"95% CI: [{test_both['ci_low']:.4f}, {test_both['ci_high']:.4f}]")

        results['approach3_conservative'] = test_both
    else:
        print("\n>>> Insufficient data for conservative test")
        results['approach3_conservative'] = {'error': 'Insufficient data'}

    # ========================================
    # Summary comparison with baseline
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE")
    print("=" * 70)

    bootstrap_file = config.TABLES_DIR / 'bootstrap_summary.json'
    if bootstrap_file.exists():
        with open(bootstrap_file) as f:
            bootstrap = json.load(f)

        if 'DiD_InfraVsReg' in bootstrap:
            bs = bootstrap['DiD_InfraVsReg']
            print(f"\nBaseline (bootstrap, all events, [-5,+30]):")
            print(f"  p-value: {bs['p_value']:.4f}")
            print(f"  95% CI: [{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]")

    print(f"\nApproach 1 (independent events, [-5,+30]):")
    if 'approach1_independent' in results and 'p_value' in results['approach1_independent']:
        r = results['approach1_independent']
        print(f"  p-value: {r['p_value']:.4f}")
        print(f"  95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    print(f"\nApproach 2 (all events, [0,+5]):")
    if 'approach2_short_window' in results and 'p_value' in results['approach2_short_window']:
        r = results['approach2_short_window']
        print(f"  p-value: {r['p_value']:.4f}")
        print(f"  95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    print(f"\nApproach 3 (independent events, [0,+5]):")
    if 'approach3_conservative' in results and 'p_value' in results['approach3_conservative']:
        r = results['approach3_conservative']
        print(f"  p-value: {r['p_value']:.4f}")
        print(f"  95% CI: [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")

    # Check convergence
    all_null = True
    for key in ['approach1_independent', 'approach2_short_window', 'approach3_conservative']:
        if key in results and 'ci_crosses_zero' in results[key]:
            if not results[key]['ci_crosses_zero']:
                all_null = False

    if all_null:
        print("\n>>> ALL APPROACHES CONFIRM NULL FINDING")
        print("    Overlap does not explain away the null result")
    else:
        print("\n>>> SOME APPROACHES SHOW SIGNIFICANCE")
        print("    Overlap may partially explain the null finding")

    # Save results
    output_file = config.TABLES_DIR / 'nonoverlap_robustness.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
