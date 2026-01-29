#!/usr/bin/env python3
"""
Corrected Bootstrap Analysis
============================

Addresses reviewer concern: events should be equally weighted.

Original implementation pooled all asset-level CARs, giving more weight to
events with more assets. Correct approach: average within events FIRST,
then bootstrap event-level means.

Usage:
    python scripts/run_corrected_bootstrap.py
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


class CorrectedEventBlockBootstrap:
    """
    Event-level block bootstrap with EQUAL event weighting.

    Key difference from original: computes event-level CAR by averaging
    across assets FIRST, then resamples event means. Each event gets
    equal weight regardless of asset coverage.
    """

    def __init__(
        self,
        n_bootstrap: int = 5000,
        confidence: float = 0.95,
        random_state: int = config.RANDOM_SEED
    ):
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence
        self.rng = np.random.default_rng(random_state)

    def compute_event_level_cars(
        self,
        event_cars: dict
    ) -> dict:
        """
        Compute event-level CARs by averaging across assets within each event.

        This ensures each event gets equal weight regardless of asset coverage.

        Args:
            event_cars: Dict mapping event_id -> {symbol: car}

        Returns:
            Dict mapping event_id -> mean_car (single value per event)
        """
        event_means = {}
        for event_id, asset_cars in event_cars.items():
            if len(asset_cars) > 0:
                event_means[event_id] = np.mean(list(asset_cars.values()))
        return event_means

    def bootstrap_mean_car(
        self,
        event_cars: dict,
        n_bootstrap: int = None
    ) -> dict:
        """
        Bootstrap CI for mean CAR with equal event weighting.

        Process:
        1. Average CARs within each event (event-level means)
        2. Resample event-level means with replacement
        3. Compute mean of resampled event means

        This gives each event equal weight.
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        # Step 1: Compute event-level CARs
        event_means = self.compute_event_level_cars(event_cars)
        event_ids = list(event_means.keys())
        n_events = len(event_ids)

        if n_events < 2:
            return {
                'statistic': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'se': np.nan,
                'p_value': np.nan,
                'n_events': n_events,
                'method': 'event_equal_weighted_bootstrap',
                'error': 'Insufficient events'
            }

        # Original statistic: mean of event-level means
        event_mean_values = np.array([event_means[eid] for eid in event_ids])
        original_mean = np.mean(event_mean_values)

        # Step 2 & 3: Bootstrap by resampling event means
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample event IDs with replacement
            sampled_ids = self.rng.choice(event_ids, size=n_events, replace=True)

            # Get event-level means for sampled events
            sampled_means = [event_means[eid] for eid in sampled_ids]
            bootstrap_means.append(np.mean(sampled_means))

        bootstrap_means = np.array(bootstrap_means)

        # CI and p-value
        alpha = (1 - self.confidence) / 2
        ci_low = np.percentile(bootstrap_means, alpha * 100)
        ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)
        bootstrap_se = bootstrap_means.std()

        # Two-tailed p-value against zero
        if original_mean >= 0:
            p_value = 2 * np.mean(bootstrap_means <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_means >= 0)
        p_value = min(p_value, 1.0)

        return {
            'statistic': float(original_mean),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'se': float(bootstrap_se),
            'p_value': float(p_value),
            'n_events': n_events,
            'n_observations': sum(len(v) for v in event_cars.values()),
            'method': 'event_equal_weighted_bootstrap',
            'event_level_cars': {str(k): float(v) for k, v in event_means.items()}
        }

    def bootstrap_difference_test(
        self,
        event_cars_a: dict,
        event_cars_b: dict,
        n_bootstrap: int = None
    ) -> dict:
        """
        Bootstrap test for difference in mean CARs with equal event weighting.
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        # Compute event-level means for each group
        event_means_a = self.compute_event_level_cars(event_cars_a)
        event_means_b = self.compute_event_level_cars(event_cars_b)

        event_ids_a = list(event_means_a.keys())
        event_ids_b = list(event_means_b.keys())
        n_events_a = len(event_ids_a)
        n_events_b = len(event_ids_b)

        if n_events_a < 2 or n_events_b < 2:
            return {
                'statistic': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'p_value': np.nan,
                'n_events_a': n_events_a,
                'n_events_b': n_events_b,
                'method': 'event_equal_weighted_diff_bootstrap',
                'error': 'Insufficient events'
            }

        # Original difference
        mean_a = np.mean([event_means_a[eid] for eid in event_ids_a])
        mean_b = np.mean([event_means_b[eid] for eid in event_ids_b])
        original_diff = mean_a - mean_b

        # Bootstrap the difference
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            sampled_a = self.rng.choice(event_ids_a, size=n_events_a, replace=True)
            sampled_b = self.rng.choice(event_ids_b, size=n_events_b, replace=True)

            boot_mean_a = np.mean([event_means_a[eid] for eid in sampled_a])
            boot_mean_b = np.mean([event_means_b[eid] for eid in sampled_b])
            bootstrap_diffs.append(boot_mean_a - boot_mean_b)

        bootstrap_diffs = np.array(bootstrap_diffs)

        # CI and p-value
        alpha = (1 - self.confidence) / 2
        ci_low = np.percentile(bootstrap_diffs, alpha * 100)
        ci_high = np.percentile(bootstrap_diffs, (1 - alpha) * 100)
        bootstrap_se = bootstrap_diffs.std()

        if original_diff >= 0:
            p_value = 2 * np.mean(bootstrap_diffs <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= 0)
        p_value = min(p_value, 1.0)

        return {
            'statistic': float(original_diff),
            'ci_low': float(ci_low),
            'ci_high': float(ci_high),
            'se': float(bootstrap_se),
            'p_value': float(p_value),
            'n_events_a': n_events_a,
            'n_events_b': n_events_b,
            'mean_a': float(mean_a),
            'mean_b': float(mean_b),
            'method': 'event_equal_weighted_diff_bootstrap'
        }


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


def compute_event_cars(
    returns_dict: dict,
    events: list,
    model: ConstantMeanModel,
    window: tuple = (-5, 30)
) -> dict:
    """Compute CARs for each event-asset pair."""
    event_cars = {}

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_cars[event_id] = {}

        for symbol, returns in returns_dict.items():
            result = model.compute_abnormal_returns(returns, event_date, window)
            if 'error' not in result:
                event_cars[event_id][symbol] = result['car']

    return event_cars


def main():
    print("\n" + "=" * 70)
    print("CORRECTED BOOTSTRAP ANALYSIS (Equal Event Weighting)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAddressing reviewer concern: events should be equally weighted.")
    print("Original implementation pooled all asset-level CARs.")
    print("Correction: average within events FIRST, then bootstrap event means.")
    print("=" * 70)

    # Load data
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()
    bootstrap = CorrectedEventBlockBootstrap(n_bootstrap=5000)

    events_by_type = load_events()

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

    results = {
        'analysis_date': datetime.now().isoformat(),
        'method': 'event_equal_weighted_bootstrap',
        'note': 'Events averaged across assets FIRST, then bootstrapped. Each event weighted equally.'
    }

    # ========================================
    # NEGATIVE EVENTS ANALYSIS
    # ========================================
    print("\n" + "=" * 70)
    print("NEGATIVE EVENTS: CORRECTED BOOTSTRAP")
    print("=" * 70)

    infra_neg = events_by_type.get('Infra_Negative', [])
    reg_neg = events_by_type.get('Reg_Negative', [])

    print(f"\nInfra_Negative: {len(infra_neg)} events")
    print(f"Reg_Negative: {len(reg_neg)} events")

    # Compute CARs
    infra_cars = compute_event_cars(returns_dict, infra_neg, model)
    reg_cars = compute_event_cars(returns_dict, reg_neg, model)

    # Show event-level coverage
    print("\n--- Asset Coverage per Event ---")
    print("\nInfra_Negative:")
    for event in infra_neg:
        eid = event['event_id']
        n_assets = len(infra_cars.get(eid, {}))
        print(f"  [{eid}] {event.get('label', '')}: {n_assets} assets")

    print("\nReg_Negative:")
    for event in reg_neg:
        eid = event['event_id']
        n_assets = len(reg_cars.get(eid, {}))
        print(f"  [{eid}] {event.get('label', '')}: {n_assets} assets")

    # Bootstrap with equal weighting
    print("\n--- Corrected Bootstrap Results ---")

    infra_result = bootstrap.bootstrap_mean_car(infra_cars)
    reg_result = bootstrap.bootstrap_mean_car(reg_cars)
    diff_result = bootstrap.bootstrap_difference_test(infra_cars, reg_cars)

    print(f"\nInfra_Negative (N={infra_result['n_events']} events):")
    print(f"  Mean CAR: {infra_result['statistic']:+.4f}")
    print(f"  95% CI: [{infra_result['ci_low']:.4f}, {infra_result['ci_high']:.4f}]")
    print(f"  p-value: {infra_result['p_value']:.4f}")
    print(f"  Event-level CARs: {list(infra_result.get('event_level_cars', {}).values())}")

    print(f"\nReg_Negative (N={reg_result['n_events']} events):")
    print(f"  Mean CAR: {reg_result['statistic']:+.4f}")
    print(f"  95% CI: [{reg_result['ci_low']:.4f}, {reg_result['ci_high']:.4f}]")
    print(f"  p-value: {reg_result['p_value']:.4f}")
    print(f"  Event-level CARs: {list(reg_result.get('event_level_cars', {}).values())}")

    print(f"\nDifference (Infra - Reg):")
    print(f"  Diff: {diff_result['statistic']:+.4f}")
    print(f"  95% CI: [{diff_result['ci_low']:.4f}, {diff_result['ci_high']:.4f}]")
    print(f"  p-value: {diff_result['p_value']:.4f}")

    results['Infra_Negative'] = infra_result
    results['Reg_Negative'] = reg_result
    results['DiD_InfraVsReg'] = diff_result

    # ========================================
    # COMPARISON WITH ORIGINAL
    # ========================================
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs CORRECTED")
    print("=" * 70)

    original_file = config.TABLES_DIR / 'bootstrap_summary.json'
    if original_file.exists():
        with open(original_file) as f:
            original = json.load(f)

        print("\n| Metric | Original (obs-weighted) | Corrected (event-equal) |")
        print("|--------|------------------------|------------------------|")

        if 'Infra_Negative' in original:
            o = original['Infra_Negative']
            c = infra_result
            print(f"| Infra mean | {o.get('statistic', o.get('mean_car', 'N/A')):.4f} | {c['statistic']:.4f} |")
            print(f"| Infra p-value | {o['p_value']:.4f} | {c['p_value']:.4f} |")

        if 'Reg_Negative' in original:
            o = original['Reg_Negative']
            c = reg_result
            print(f"| Reg mean | {o.get('statistic', o.get('mean_car', 'N/A')):.4f} | {c['statistic']:.4f} |")
            print(f"| Reg p-value | {o['p_value']:.4f} | {c['p_value']:.4f} |")

        if 'DiD_InfraVsReg' in original:
            o = original['DiD_InfraVsReg']
            c = diff_result
            print(f"| DiD p-value | {o['p_value']:.4f} | {c['p_value']:.4f} |")

    # ========================================
    # CONCLUSION
    # ========================================
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    ci_crosses_zero = diff_result['ci_low'] <= 0 <= diff_result['ci_high']
    if ci_crosses_zero:
        print("\n>>> NULL FINDING CONFIRMED with corrected event-equal weighting")
        print("    CI for (Infra - Reg) still crosses zero")
    else:
        print("\n>>> FINDING CHANGES with corrected event-equal weighting!")
        print("    CI no longer crosses zero")

    results['conclusion'] = {
        'null_confirmed': bool(ci_crosses_zero),
        'original_method': 'observation_weighted',
        'corrected_method': 'event_equal_weighted'
    }

    # Save results
    output_file = config.TABLES_DIR / 'corrected_bootstrap_summary.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {output_file}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
