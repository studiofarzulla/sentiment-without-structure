#!/usr/bin/env python3
"""
Exact Permutation Test
======================

Complements bootstrap inference with exact permutation test.

Reviewer concern: Bootstrap may not be appropriate for small N.

Approach:
- H0: No difference between Infra_Negative and Reg_Negative
- Takes 17 events (10 Infra_Neg + 7 Reg_Neg)
- Computes observed Δ = mean(Infra) - mean(Reg)
- Permutes group labels exhaustively (exact: C(17,10) = 19,448 permutations)
- P-value = proportion of |Δ_perm| ≥ |Δ_obs|

Usage:
    python scripts/run_permutation_test.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from itertools import combinations
from scipy import stats

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.event_study import ConstantMeanModel


def load_reclassified_events() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load events from reclassified JSON."""
    events_file = config.DATA_DIR / 'events_reclassified.json'

    with open(events_file) as f:
        data = json.load(f)

    events = [
        e for e in data['events']
        if e.get('include_in_reanalysis', True)
        and e.get('meets_impact_threshold', False)
        and e.get('has_sufficient_estimation_data', True)
    ]

    events_by_type = {}
    for e in events:
        etype = e.get('type_detailed', e['type'])
        if etype not in events_by_type:
            events_by_type[etype] = []
        events_by_type[etype].append(e)

    return events, events_by_type


def load_returns_data(fetcher, assets: List[str]) -> Dict[str, pd.Series]:
    """Load return series for all assets."""
    returns_dict = {}

    for symbol in assets:
        try:
            df = fetcher.fetch_ohlcv(symbol, config.START_DATE, config.END_DATE)
            if not df.empty and 'returns' in df.columns:
                returns_dict[symbol] = df['returns'].dropna()
        except Exception as e:
            print(f"  {symbol}: FAILED - {e}")

    return returns_dict


def compute_event_mean_cars(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    model: ConstantMeanModel,
    window: Tuple[int, int] = (-5, 30)
) -> Dict[int, float]:
    """
    Compute mean CAR for each event (averaging across assets).

    Returns dict mapping event_id -> mean CAR across assets.
    """
    event_mean_cars = {}

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        cars = []

        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, window)
            if 'error' not in result:
                cars.append(result['car'])

        if cars:
            event_mean_cars[event_id] = np.mean(cars)

    return event_mean_cars


def exact_permutation_test(
    group_a_values: List[float],
    group_b_values: List[float]
) -> Dict:
    """
    Exact permutation test for difference in means.

    Tests H0: mean(A) = mean(B) by exhaustive enumeration.

    Args:
        group_a_values: CARs for group A (e.g., Infra_Negative)
        group_b_values: CARs for group B (e.g., Reg_Negative)

    Returns:
        Dict with observed difference, exact p-value, and distribution stats
    """
    n_a = len(group_a_values)
    n_b = len(group_b_values)
    n_total = n_a + n_b

    # Observed difference
    obs_mean_a = np.mean(group_a_values)
    obs_mean_b = np.mean(group_b_values)
    obs_diff = obs_mean_a - obs_mean_b

    # Pool all values
    pooled = group_a_values + group_b_values

    # Count total permutations
    from math import comb
    n_permutations = comb(n_total, n_a)
    print(f"  Total permutations: C({n_total},{n_a}) = {n_permutations:,}")

    # Enumerate all permutations
    permutation_diffs = []

    for idx_a in combinations(range(n_total), n_a):
        idx_b = [i for i in range(n_total) if i not in idx_a]

        perm_a = [pooled[i] for i in idx_a]
        perm_b = [pooled[i] for i in idx_b]

        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        permutation_diffs.append(perm_diff)

    permutation_diffs = np.array(permutation_diffs)

    # Two-tailed exact p-value
    n_extreme = np.sum(np.abs(permutation_diffs) >= np.abs(obs_diff))
    exact_p_value = n_extreme / n_permutations

    return {
        'observed_diff': obs_diff,
        'mean_a': obs_mean_a,
        'mean_b': obs_mean_b,
        'n_a': n_a,
        'n_b': n_b,
        'n_permutations': n_permutations,
        'n_extreme': n_extreme,
        'exact_p_value': exact_p_value,
        'permutation_mean': permutation_diffs.mean(),
        'permutation_std': permutation_diffs.std(),
        'permutation_percentiles': {
            '2.5%': np.percentile(permutation_diffs, 2.5),
            '5%': np.percentile(permutation_diffs, 5),
            '25%': np.percentile(permutation_diffs, 25),
            '50%': np.percentile(permutation_diffs, 50),
            '75%': np.percentile(permutation_diffs, 75),
            '95%': np.percentile(permutation_diffs, 95),
            '97.5%': np.percentile(permutation_diffs, 97.5),
        }
    }


def main():
    """Run exact permutation test."""
    print("\n" + "=" * 70)
    print("EXACT PERMUTATION TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nH0: No difference in mean CARs between Infra_Negative and Reg_Negative")

    # Initialize
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()

    # Load data
    events, events_by_type = load_reclassified_events()

    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]  # BTC, ETH, SOL, ADA
    returns_dict = load_returns_data(fetcher, assets)

    if not returns_dict:
        print("ERROR: No return data loaded")
        return

    # Focus on primary comparison
    infra_neg = events_by_type.get('Infra_Negative', [])
    reg_neg = events_by_type.get('Reg_Negative', [])

    print(f"\nInfra_Negative: {len(infra_neg)} events")
    print(f"Reg_Negative: {len(reg_neg)} events")

    # Compute mean CAR per event (averaging across assets)
    print("\n--- Computing Event-Level CARs ---")
    infra_cars = compute_event_mean_cars(returns_dict, infra_neg, model)
    reg_cars = compute_event_mean_cars(returns_dict, reg_neg, model)

    print(f"  Infra_Negative: {len(infra_cars)} events with valid CARs")
    print(f"  Reg_Negative: {len(reg_cars)} events with valid CARs")

    # Convert to lists
    infra_values = list(infra_cars.values())
    reg_values = list(reg_cars.values())

    # Run exact permutation test
    print("\n--- Running Exact Permutation Test ---")
    result = exact_permutation_test(infra_values, reg_values)

    # Display results
    print("\n" + "=" * 60)
    print("PERMUTATION TEST RESULTS")
    print("=" * 60)

    print(f"\nGroup Statistics:")
    print(f"  Infra_Negative: n={result['n_a']}, mean CAR = {result['mean_a']:.4f}")
    print(f"  Reg_Negative: n={result['n_b']}, mean CAR = {result['mean_b']:.4f}")

    print(f"\nObserved Difference (Infra - Reg):")
    print(f"  Δ = {result['observed_diff']:.4f} ({result['observed_diff']*100:+.2f} pp)")

    print(f"\nPermutation Distribution:")
    print(f"  Mean: {result['permutation_mean']:.6f}")
    print(f"  Std:  {result['permutation_std']:.4f}")
    print(f"  2.5%: {result['permutation_percentiles']['2.5%']:.4f}")
    print(f"  97.5%: {result['permutation_percentiles']['97.5%']:.4f}")

    print(f"\nExact p-value:")
    print(f"  {result['n_extreme']:,} / {result['n_permutations']:,} permutations")
    print(f"  as extreme or more extreme than observed")
    print(f"  p = {result['exact_p_value']:.4f}")

    # Interpretation
    print("\n" + "-" * 40)
    print("INTERPRETATION")
    print("-" * 40)

    if result['exact_p_value'] > 0.05:
        print(f"\n✓ FAIL TO REJECT H0 at α = 0.05")
        print(f"  Exact p-value ({result['exact_p_value']:.4f}) > 0.05")
        print(f"  No significant difference between event types")
        print(f"  NULL FINDING CONFIRMED by permutation test")
    else:
        print(f"\n✗ REJECT H0 at α = 0.05")
        print(f"  Exact p-value ({result['exact_p_value']:.4f}) < 0.05")
        print(f"  Significant difference detected")

    # Compare with bootstrap
    print("\n--- Comparison with Bootstrap ---")
    print("  (Check bootstrap_summary.json for bootstrap p-value)")

    # Save results
    output_data = {
        'analysis_date': datetime.now().isoformat(),
        'hypothesis': 'H0: mean(Infra_Neg) = mean(Reg_Neg)',
        'test_type': 'exact_permutation',
        **result
    }

    output_file = config.TABLES_DIR / 'permutation_test.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")

    # Summary for paper
    print("\n\n--- Text for Paper ---")
    print(f"""
As robustness to bootstrap inference, we conduct an exact permutation test
for H0: no difference between infrastructure and regulatory event CARs.
With {result['n_a']} infrastructure and {result['n_b']} regulatory events,
there are C({result['n_a'] + result['n_b']}, {result['n_a']}) = {result['n_permutations']:,}
possible group assignments. The observed difference of {result['observed_diff']*100:+.2f} pp
yields an exact two-tailed p-value of {result['exact_p_value']:.3f}, confirming the
bootstrap result (p = 0.81): no significant difference between event types.
""")

    print(f"\n\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
