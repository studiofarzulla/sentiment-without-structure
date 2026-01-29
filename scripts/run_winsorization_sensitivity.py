#!/usr/bin/env python3
"""
Winsorization Sensitivity Analysis
==================================

Tests robustness of results to different return cap levels.

Reviewer concern: ±50% winsorization may be too aggressive or too permissive.

Levels tested:
- ±30% (conservative)
- ±50% (baseline - current)
- ±75% (permissive)
- Uncapped (with robust tests)

Usage:
    python scripts/run_winsorization_sensitivity.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple
from scipy import stats

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.robustness import EventBlockBootstrap


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


def cap_returns(returns: pd.Series, cap_level: float) -> pd.Series:
    """Cap returns at specified level (e.g., 0.30 for ±30%)."""
    if cap_level is None:
        return returns.copy()
    return returns.clip(lower=-cap_level, upper=cap_level)


def compute_car_with_cap(
    returns: pd.Series,
    event_date: str,
    cap_level: float,
    window: Tuple[int, int] = (-5, 30),
    estimation_window: int = 250,
    gap_window: int = 30
) -> Dict:
    """Compute CAR using capped returns."""
    returns_copy = returns.copy()
    returns_copy.index = pd.to_datetime(returns_copy.index)
    event_dt = pd.to_datetime(event_date)

    # Apply cap
    capped_returns = cap_returns(returns_copy, cap_level)

    # Estimation period
    est_end = event_dt - pd.Timedelta(days=gap_window + 1)
    est_start = est_end - pd.Timedelta(days=estimation_window)

    est_returns = capped_returns[(capped_returns.index >= est_start) &
                                  (capped_returns.index <= est_end)].dropna()

    if len(est_returns) < 60:
        return {'error': 'Insufficient estimation data'}

    expected_return = est_returns.mean()

    # Event window
    pre, post = window
    event_start = event_dt + pd.Timedelta(days=pre)
    event_end = event_dt + pd.Timedelta(days=post)

    event_returns = capped_returns[(capped_returns.index >= event_start) &
                                    (capped_returns.index <= event_end)].dropna()

    if len(event_returns) < 5:
        return {'error': 'Insufficient event window data'}

    abnormal_returns = event_returns - expected_return
    car = abnormal_returns.sum()

    return {'car': car, 'n_days': len(event_returns)}


def run_analysis_with_cap(
    returns_dict: Dict[str, pd.Series],
    events: List[Dict],
    cap_level: float,
    window: Tuple[int, int] = (-5, 30)
) -> Dict[int, Dict[str, float]]:
    """Run event study with specified cap level."""
    event_cars = {}

    for event in events:
        event_id = event['event_id']
        event_date = event['date']
        event_cars[event_id] = {}

        for symbol, returns in returns_dict.items():
            result = compute_car_with_cap(returns, event_date, cap_level, window)
            if 'error' not in result:
                event_cars[event_id][symbol] = result['car']

    return event_cars


def main():
    """Run winsorization sensitivity analysis."""
    print("\n" + "=" * 70)
    print("WINSORIZATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Cap levels to test
    cap_levels = {
        '±30%': 0.30,
        '±50% (baseline)': 0.50,
        '±75%': 0.75,
        'Uncapped': None
    }

    # Initialize
    fetcher = BinanceDataFetcher()
    bootstrap = EventBlockBootstrap(n_bootstrap=5000)

    # Load data
    events, events_by_type = load_reclassified_events()

    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]  # BTC, ETH, SOL, ADA
    returns_dict = load_returns_data(fetcher, assets)

    if not returns_dict:
        print("ERROR: No return data loaded")
        return

    # Focus on primary comparison: Infra_Negative vs Reg_Negative
    infra_neg = events_by_type.get('Infra_Negative', [])
    reg_neg = events_by_type.get('Reg_Negative', [])

    print(f"\nInfra_Negative: {len(infra_neg)} events")
    print(f"Reg_Negative: {len(reg_neg)} events")

    # Results storage
    results = []

    print("\n" + "=" * 60)
    print("RESULTS BY WINSORIZATION LEVEL")
    print("=" * 60)

    for cap_name, cap_level in cap_levels.items():
        print(f"\n--- {cap_name} ---")

        # Compute CARs for each group
        infra_cars = run_analysis_with_cap(returns_dict, infra_neg, cap_level)
        reg_cars = run_analysis_with_cap(returns_dict, reg_neg, cap_level)

        # Bootstrap inference for each group
        infra_result = bootstrap.bootstrap_mean_car(infra_cars)
        reg_result = bootstrap.bootstrap_mean_car(reg_cars)

        # Difference test
        diff_result = bootstrap.bootstrap_difference_test(infra_cars, reg_cars)

        print(f"\n  Infra_Negative:")
        print(f"    Mean CAR: {infra_result.statistic:.4f}")
        print(f"    95% CI: [{infra_result.ci_low:.4f}, {infra_result.ci_high:.4f}]")
        print(f"    p-value: {infra_result.p_value:.4f}")

        print(f"\n  Reg_Negative:")
        print(f"    Mean CAR: {reg_result.statistic:.4f}")
        print(f"    95% CI: [{reg_result.ci_low:.4f}, {reg_result.ci_high:.4f}]")
        print(f"    p-value: {reg_result.p_value:.4f}")

        print(f"\n  Difference (Infra - Reg):")
        print(f"    Δ: {diff_result.statistic:.4f}")
        print(f"    95% CI: [{diff_result.ci_low:.4f}, {diff_result.ci_high:.4f}]")
        print(f"    p-value: {diff_result.p_value:.4f}")

        crosses_zero = diff_result.ci_low <= 0 <= diff_result.ci_high
        print(f"    CI crosses zero: {crosses_zero} → {'NULL' if crosses_zero else 'SIGNIFICANT'}")

        results.append({
            'cap_level': cap_name,
            'cap_value': cap_level,
            'infra_car': infra_result.statistic,
            'infra_ci_low': infra_result.ci_low,
            'infra_ci_high': infra_result.ci_high,
            'infra_pvalue': infra_result.p_value,
            'reg_car': reg_result.statistic,
            'reg_ci_low': reg_result.ci_low,
            'reg_ci_high': reg_result.ci_high,
            'reg_pvalue': reg_result.p_value,
            'diff': diff_result.statistic,
            'diff_ci_low': diff_result.ci_low,
            'diff_ci_high': diff_result.ci_high,
            'diff_pvalue': diff_result.p_value,
            'ci_crosses_zero': crosses_zero
        })

    # Save results
    df = pd.DataFrame(results)
    output_file = config.TABLES_DIR / 'winsorization_sensitivity.csv'
    df.to_csv(output_file, index=False)
    print(f"\n\nSaved: {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_null = all(r['ci_crosses_zero'] for r in results)
    print(f"\nAll levels show null finding: {all_null}")

    if all_null:
        print("\n✓ ROBUST: Null finding holds across all winsorization levels")
    else:
        significant_levels = [r['cap_level'] for r in results if not r['ci_crosses_zero']]
        print(f"\n⚠️ WARNING: Some levels show significance: {significant_levels}")

    # LaTeX table output
    print("\n\n--- LaTeX Table ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Winsorization Sensitivity Analysis}")
    print(r"\label{tab:winsor_sens}")
    print(r"\begin{tabular}{@{}lrrrrr@{}}")
    print(r"\toprule")
    print(r"\textbf{Cap Level} & \textbf{Infra\_Neg CAR} & \textbf{Reg\_Neg CAR} & \textbf{$\Delta$} & \textbf{p-value} & \textbf{Null?} \\")
    print(r"\midrule")

    for r in results:
        null_mark = r'\checkmark' if r['ci_crosses_zero'] else ''
        baseline_mark = '*' if r['cap_level'] == '±50% (baseline)' else ''
        print(f"{r['cap_level']}{baseline_mark} & {r['infra_car']:.1%} & {r['reg_car']:.1%} & {r['diff']:+.1%} & {r['diff_pvalue']:.3f} & {null_mark} \\\\")

    print(r"\bottomrule")
    print(r"\multicolumn{6}{l}{\footnotesize * Baseline specification. $\Delta$ = Infra - Reg. Null = CI crosses zero.} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print(f"\n\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
