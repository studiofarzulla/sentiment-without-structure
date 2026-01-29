#!/usr/bin/env python3
"""
Black Thursday Sensitivity Analysis
====================================

Reviewer concern: Black Thursday (March 2020) was a global macro event,
not a crypto infrastructure failure. Test sensitivity to exclusion.

Usage:
    python scripts/run_black_thursday_sensitivity.py
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


def load_events():
    """Load events from reclassified JSON."""
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


def compute_event_level_cars(returns_dict, events, model, window=(-5, 30)):
    """Compute event-level CARs (averaged across assets within each event)."""
    event_cars = {}

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        asset_cars = []

        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, window)
            if 'error' not in result:
                asset_cars.append(result['car'])

        if asset_cars:
            event_cars[event_id] = {
                'label': event.get('label', ''),
                'date': event_date,
                'mean_car': np.mean(asset_cars),
                'n_assets': len(asset_cars)
            }

    return event_cars


def bootstrap_comparison(event_cars_a, event_cars_b, n_bootstrap=5000, seed=42):
    """Event-equal-weighted bootstrap comparison."""
    rng = np.random.default_rng(seed)

    ids_a = list(event_cars_a.keys())
    ids_b = list(event_cars_b.keys())

    means_a = np.array([event_cars_a[eid]['mean_car'] for eid in ids_a])
    means_b = np.array([event_cars_b[eid]['mean_car'] for eid in ids_b])

    orig_diff = np.mean(means_a) - np.mean(means_b)

    boot_diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(means_a, size=len(means_a), replace=True)
        sample_b = rng.choice(means_b, size=len(means_b), replace=True)
        boot_diffs.append(np.mean(sample_a) - np.mean(sample_b))

    boot_diffs = np.array(boot_diffs)

    ci_low = np.percentile(boot_diffs, 2.5)
    ci_high = np.percentile(boot_diffs, 97.5)

    if orig_diff >= 0:
        p_value = 2 * np.mean(boot_diffs <= 0)
    else:
        p_value = 2 * np.mean(boot_diffs >= 0)
    p_value = min(p_value, 1.0)

    return {
        'mean_a': float(np.mean(means_a)),
        'mean_b': float(np.mean(means_b)),
        'difference': float(orig_diff),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'p_value': float(p_value),
        'n_a': len(ids_a),
        'n_b': len(ids_b)
    }


def main():
    print("\n" + "=" * 70)
    print("BLACK THURSDAY SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("Testing: Does excluding Black Thursday (2020-03-12) change results?")
    print("Concern: March 2020 was global macro liquidation, not crypto-specific")
    print("=" * 70)

    # Load data
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()
    events_by_type = load_events()

    # Load returns
    print("\n--- Loading Returns ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]
    returns_dict = {}

    for symbol in assets:
        try:
            df = fetcher.fetch_ohlcv(symbol, config.START_DATE, config.END_DATE)
            if not df.empty and 'returns' in df.columns:
                returns_dict[symbol] = df['returns'].dropna()
        except:
            pass

    print(f"  Loaded {len(returns_dict)} assets")

    # Get events
    infra_neg = events_by_type.get('Infra_Negative', [])
    reg_neg = events_by_type.get('Reg_Negative', [])

    # Find Black Thursday
    black_thursday = None
    for e in infra_neg:
        if '2020-03-12' in e['date'] or 'Black Thursday' in e.get('label', ''):
            black_thursday = e
            break

    if not black_thursday:
        print("\n>>> ERROR: Black Thursday not found in events!")
        return

    print(f"\n--- Black Thursday Event ---")
    print(f"  ID: {black_thursday['event_id']}")
    print(f"  Date: {black_thursday['date']}")
    print(f"  Label: {black_thursday.get('label', 'N/A')}")

    # Compute CARs with and without Black Thursday
    print("\n--- Computing Event-Level CARs ---")

    infra_cars_full = compute_event_level_cars(returns_dict, infra_neg, model)
    infra_cars_excl = compute_event_level_cars(
        returns_dict,
        [e for e in infra_neg if e['event_id'] != black_thursday['event_id']],
        model
    )
    reg_cars = compute_event_level_cars(returns_dict, reg_neg, model)

    print(f"\nInfra_Negative (full): {len(infra_cars_full)} events")
    print(f"Infra_Negative (excl. Black Thursday): {len(infra_cars_excl)} events")
    print(f"Reg_Negative: {len(reg_cars)} events")

    # Show Black Thursday's contribution
    bt_id = black_thursday['event_id']
    if bt_id in infra_cars_full:
        bt_car = infra_cars_full[bt_id]['mean_car']
        print(f"\nBlack Thursday CAR: {bt_car:+.1%}")

        other_cars = [v['mean_car'] for k, v in infra_cars_full.items() if k != bt_id]
        print(f"Other Infra_Neg mean: {np.mean(other_cars):+.1%}")

    # Bootstrap comparisons
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    result_full = bootstrap_comparison(infra_cars_full, reg_cars)
    result_excl = bootstrap_comparison(infra_cars_excl, reg_cars)

    print("\n| Specification | N Infra | Infra CAR | Reg CAR | Δ | p-value | Null? |")
    print("|---------------|---------|-----------|---------|---|---------|-------|")
    print(f"| With Black Thursday | {result_full['n_a']} | {result_full['mean_a']:+.1%} | {result_full['mean_b']:+.1%} | {result_full['difference']:+.1%} | {result_full['p_value']:.2f} | {'✓' if result_full['ci_low'] <= 0 <= result_full['ci_high'] else '✗'} |")
    print(f"| Excl. Black Thursday | {result_excl['n_a']} | {result_excl['mean_a']:+.1%} | {result_excl['mean_b']:+.1%} | {result_excl['difference']:+.1%} | {result_excl['p_value']:.2f} | {'✓' if result_excl['ci_low'] <= 0 <= result_excl['ci_high'] else '✗'} |")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    both_null = (result_full['ci_low'] <= 0 <= result_full['ci_high']) and \
                (result_excl['ci_low'] <= 0 <= result_excl['ci_high'])

    if both_null:
        print("\n>>> NULL FINDING ROBUST TO BLACK THURSDAY EXCLUSION")
        print("    Results don't depend on this classification choice")
    else:
        print("\n>>> WARNING: Results sensitive to Black Thursday!")
        print("    Classification matters - consider reporting both")

    # Save results
    results = {
        'analysis_date': datetime.now().isoformat(),
        'black_thursday_event': {
            'event_id': black_thursday['event_id'],
            'date': black_thursday['date'],
            'car': float(infra_cars_full.get(bt_id, {}).get('mean_car', np.nan))
        },
        'with_black_thursday': result_full,
        'excluding_black_thursday': result_excl,
        'robust_to_exclusion': bool(both_null)
    }

    output_file = config.TABLES_DIR / 'black_thursday_sensitivity.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")


if __name__ == "__main__":
    main()
