#!/usr/bin/env python3
"""
Event Selection Audit
=====================

Addresses reviewer concern about event selection bias.

Reviewer concern: Return threshold (|BTC| > 5%) conditions on outcome.

This script:
1. Classifies each event by selection criterion:
   - "return_threshold": Included primarily due to |BTC return| > 5%
   - "exogenous": Included due to impact/user threshold only (known major event)

2. Runs subsample re-analysis on EXOGENOUS-ONLY events:
   - Filters to events selected WITHOUT return threshold
   - Re-runs event study on pure subsample
   - Reports bootstrap CI and p-value for subsample
   - Compares to full sample results

Usage:
    python scripts/audit_event_selection.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple

from src import config
from src.data_fetcher import BinanceDataFetcher
from src.event_study import ConstantMeanModel
from src.robustness import EventBlockBootstrap


def load_reclassified_events() -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Load events from reclassified JSON."""
    events_file = config.DATA_DIR / 'events_reclassified.json'

    with open(events_file) as f:
        data = json.load(f)

    # Get ALL events with full metadata
    events = data['events']

    return events, data


def classify_selection_criterion(event: Dict) -> str:
    """
    Classify how an event was selected for inclusion.

    Returns:
        'exogenous': Event is a known major event (is_known_major_event=True)
                     regardless of return threshold
        'return_threshold': Event met return threshold but NOT a known major event
        'both': Both criteria met
        'excluded': Not meeting criteria
    """
    is_major = event.get('is_known_major_event', False)
    meets_auto = event.get('meets_auto_threshold', False)
    meets_impact = event.get('meets_impact_threshold', False)

    if not meets_impact:
        return 'excluded'

    if is_major and meets_auto:
        return 'both'
    elif is_major:
        return 'exogenous'
    elif meets_auto:
        return 'return_threshold'
    else:
        # meets_impact but neither auto nor major
        # This means it's a known major event that didn't hit return threshold
        return 'exogenous'


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


def main():
    """Run event selection audit."""
    print("\n" + "=" * 70)
    print("EVENT SELECTION AUDIT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nAddresses reviewer concern: Return threshold conditions on outcome")

    # Load events
    events, data = load_reclassified_events()

    # Part 1: Classify all events by selection criterion
    print("\n" + "=" * 60)
    print("PART 1: SELECTION CRITERIA BREAKDOWN")
    print("=" * 60)

    # Focus on events included in reanalysis
    included_events = [
        e for e in events
        if e.get('include_in_reanalysis', True)
        and e.get('meets_impact_threshold', False)
        and e.get('has_sufficient_estimation_data', True)
    ]

    # Classify each event
    classification = []
    for event in included_events:
        criterion = classify_selection_criterion(event)
        etype = event.get('type_detailed', event.get('type', 'Unknown'))

        classification.append({
            'event_id': event['event_id'],
            'label': event.get('label', ''),
            'date': event['date'],
            'type_detailed': etype,
            'is_known_major_event': event.get('is_known_major_event', False),
            'meets_auto_threshold': event.get('meets_auto_threshold', False),
            'selection_criterion': criterion,
            'same_day_return': event.get('impact_validation', {}).get('same_day_return', None),
            'three_day_return': event.get('impact_validation', {}).get('three_day_return', None),
        })

    df_class = pd.DataFrame(classification)

    # Summary by type and criterion
    print("\n--- All Included Events ---")
    print(f"Total events: {len(df_class)}")

    print("\n--- By Selection Criterion ---")
    for criterion in df_class['selection_criterion'].unique():
        n = (df_class['selection_criterion'] == criterion).sum()
        print(f"  {criterion}: {n} events")

    print("\n--- By Type × Selection Criterion ---")
    cross_tab = pd.crosstab(
        df_class['type_detailed'],
        df_class['selection_criterion'],
        margins=True
    )
    print(cross_tab)

    # Focus on negative events
    print("\n--- Negative Events Only (Primary Analysis) ---")
    neg_events = df_class[df_class['type_detailed'].isin(['Infra_Negative', 'Reg_Negative'])]

    cross_tab_neg = pd.crosstab(
        neg_events['type_detailed'],
        neg_events['selection_criterion'],
        margins=True
    )
    print(cross_tab_neg)

    # List exogenous-only events
    exogenous_neg = neg_events[neg_events['selection_criterion'].isin(['exogenous', 'both'])]
    print(f"\n--- Exogenous Negative Events ({len(exogenous_neg)}) ---")
    for _, row in exogenous_neg.iterrows():
        print(f"  [{row['event_id']}] {row['label']} ({row['type_detailed']}) - {row['selection_criterion']}")

    # Part 2: Subsample Analysis on Exogenous-Only Events
    print("\n" + "=" * 60)
    print("PART 2: EXOGENOUS-ONLY SUBSAMPLE ANALYSIS")
    print("=" * 60)

    # Initialize for subsample analysis
    fetcher = BinanceDataFetcher()
    model = ConstantMeanModel()
    bootstrap = EventBlockBootstrap(n_bootstrap=5000)

    print("\n--- Loading Returns Data ---")
    assets = config.TIER1_ASSETS + config.TIER2_ASSETS[:2]
    returns_dict = load_returns_data(fetcher, assets)

    if not returns_dict:
        print("ERROR: No return data loaded")
        return

    # Get exogenous-only events by category
    exogenous_ids = set(exogenous_neg['event_id'].values)

    infra_exog = [
        e for e in included_events
        if e['event_id'] in exogenous_ids
        and e.get('type_detailed') == 'Infra_Negative'
    ]

    reg_exog = [
        e for e in included_events
        if e['event_id'] in exogenous_ids
        and e.get('type_detailed') == 'Reg_Negative'
    ]

    # Also get full sample for comparison
    infra_full = [e for e in included_events if e.get('type_detailed') == 'Infra_Negative']
    reg_full = [e for e in included_events if e.get('type_detailed') == 'Reg_Negative']

    print(f"\nFull Sample:")
    print(f"  Infra_Negative: {len(infra_full)} events")
    print(f"  Reg_Negative: {len(reg_full)} events")

    print(f"\nExogenous-Only Subsample:")
    print(f"  Infra_Negative: {len(infra_exog)} events")
    print(f"  Reg_Negative: {len(reg_exog)} events")

    # Compute CARs for subsample
    def compute_event_cars(events_list):
        event_cars = {}
        for event in events_list:
            event_id = event['event_id']
            event_date = event['date']
            event_cars[event_id] = {}

            for symbol, returns in returns_dict.items():
                result = model.compute_abnormal_returns(returns, event_date, (-5, 30))
                if 'error' not in result:
                    event_cars[event_id][symbol] = result['car']

        return event_cars

    print("\n--- Computing CARs ---")

    # Full sample
    infra_full_cars = compute_event_cars(infra_full)
    reg_full_cars = compute_event_cars(reg_full)

    infra_full_result = bootstrap.bootstrap_mean_car(infra_full_cars)
    reg_full_result = bootstrap.bootstrap_mean_car(reg_full_cars)
    diff_full_result = bootstrap.bootstrap_difference_test(infra_full_cars, reg_full_cars)

    # Exogenous subsample
    infra_exog_cars = compute_event_cars(infra_exog)
    reg_exog_cars = compute_event_cars(reg_exog)

    infra_exog_result = bootstrap.bootstrap_mean_car(infra_exog_cars)
    reg_exog_result = bootstrap.bootstrap_mean_car(reg_exog_cars)

    # Only run difference test if we have enough events in both groups
    if len(infra_exog_cars) >= 2 and len(reg_exog_cars) >= 2:
        diff_exog_result = bootstrap.bootstrap_difference_test(infra_exog_cars, reg_exog_cars)
        has_diff_test = True
    else:
        has_diff_test = False
        print("\n  WARNING: Not enough exogenous events for difference test")

    # Display results
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    print("\n--- FULL SAMPLE ---")
    print(f"Infra_Negative (n={infra_full_result.n_events}):")
    print(f"  CAR: {infra_full_result.statistic:.4f}, 95% CI [{infra_full_result.ci_low:.4f}, {infra_full_result.ci_high:.4f}]")
    print(f"Reg_Negative (n={reg_full_result.n_events}):")
    print(f"  CAR: {reg_full_result.statistic:.4f}, 95% CI [{reg_full_result.ci_low:.4f}, {reg_full_result.ci_high:.4f}]")
    print(f"Difference:")
    print(f"  Δ: {diff_full_result.statistic:.4f}, 95% CI [{diff_full_result.ci_low:.4f}, {diff_full_result.ci_high:.4f}], p={diff_full_result.p_value:.4f}")

    print("\n--- EXOGENOUS-ONLY SUBSAMPLE ---")
    print(f"Infra_Negative (n={infra_exog_result.n_events}):")
    print(f"  CAR: {infra_exog_result.statistic:.4f}, 95% CI [{infra_exog_result.ci_low:.4f}, {infra_exog_result.ci_high:.4f}]")
    print(f"Reg_Negative (n={reg_exog_result.n_events}):")
    print(f"  CAR: {reg_exog_result.statistic:.4f}, 95% CI [{reg_exog_result.ci_low:.4f}, {reg_exog_result.ci_high:.4f}]")

    if has_diff_test:
        print(f"Difference:")
        print(f"  Δ: {diff_exog_result.statistic:.4f}, 95% CI [{diff_exog_result.ci_low:.4f}, {diff_exog_result.ci_high:.4f}], p={diff_exog_result.p_value:.4f}")

        exog_null = diff_exog_result.ci_low <= 0 <= diff_exog_result.ci_high
        full_null = diff_full_result.ci_low <= 0 <= diff_full_result.ci_high

        print(f"\n--- ROBUSTNESS CHECK ---")
        print(f"Full sample CI crosses zero: {full_null} (NULL)")
        print(f"Exogenous subsample CI crosses zero: {exog_null} (NULL)")

        if full_null and exog_null:
            print("\n✓ ROBUST: Null finding holds in exogenous-only subsample")
            print("  Selection on return threshold does NOT drive results")
        elif full_null and not exog_null:
            print("\n⚠️ CAUTION: Exogenous subsample shows significance")
            print("  But note smaller N may produce wider CIs")
        else:
            print("\n⚠️ Results differ between samples")

    # Save results
    results = {
        'analysis_date': datetime.now().isoformat(),
        'full_sample': {
            'infra_n': infra_full_result.n_events,
            'infra_car': infra_full_result.statistic,
            'infra_ci': [infra_full_result.ci_low, infra_full_result.ci_high],
            'reg_n': reg_full_result.n_events,
            'reg_car': reg_full_result.statistic,
            'reg_ci': [reg_full_result.ci_low, reg_full_result.ci_high],
            'diff': diff_full_result.statistic,
            'diff_ci': [diff_full_result.ci_low, diff_full_result.ci_high],
            'diff_pvalue': diff_full_result.p_value,
        },
        'exogenous_subsample': {
            'infra_n': infra_exog_result.n_events,
            'infra_car': infra_exog_result.statistic,
            'infra_ci': [infra_exog_result.ci_low, infra_exog_result.ci_high],
            'reg_n': reg_exog_result.n_events,
            'reg_car': reg_exog_result.statistic,
            'reg_ci': [reg_exog_result.ci_low, reg_exog_result.ci_high],
        },
        'selection_breakdown': cross_tab_neg.to_dict(),
        'exogenous_events': exogenous_neg.to_dict('records'),
    }

    if has_diff_test:
        results['exogenous_subsample']['diff'] = diff_exog_result.statistic
        results['exogenous_subsample']['diff_ci'] = [diff_exog_result.ci_low, diff_exog_result.ci_high]
        results['exogenous_subsample']['diff_pvalue'] = diff_exog_result.p_value

    # Save
    output_file = config.TABLES_DIR / 'event_selection_audit.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nSaved: {output_file}")

    # Save classification table
    df_class.to_csv(config.TABLES_DIR / 'event_selection_classification.csv', index=False)
    print(f"Saved: {config.TABLES_DIR / 'event_selection_classification.csv'}")

    # LaTeX tables
    print("\n\n--- LaTeX Table: Selection Criteria Breakdown ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Event Selection Criteria Breakdown}")
    print(r"\label{tab:selection_audit}")
    print(r"\begin{tabular}{@{}lrrr@{}}")
    print(r"\toprule")
    print(r"\textbf{Category} & \textbf{Exogenous$^a$} & \textbf{Return Threshold$^b$} & \textbf{Total} \\")
    print(r"\midrule")

    for etype in ['Infra_Negative', 'Reg_Negative']:
        subset = neg_events[neg_events['type_detailed'] == etype]
        exog = subset[subset['selection_criterion'].isin(['exogenous', 'both'])].shape[0]
        ret = subset[subset['selection_criterion'] == 'return_threshold'].shape[0]
        total = subset.shape[0]
        print(f"{etype.replace('_', r'\_')} & {exog} & {ret} & {total} \\\\")

    print(r"\bottomrule")
    print(r"\multicolumn{4}{l}{\footnotesize $^a$ Known major event, selected regardless of return.} \\")
    print(r"\multicolumn{4}{l}{\footnotesize $^b$ Selected primarily due to $|\text{BTC}| > 5\%$ return threshold.} \\")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n\n--- LaTeX Table: Subsample Comparison ---")
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Event Selection Robustness: Full Sample vs Exogenous-Only}")
    print(r"\label{tab:selection_robustness}")
    print(r"\begin{tabular}{@{}lrrrr@{}}")
    print(r"\toprule")
    print(r" & \multicolumn{2}{c}{\textbf{Full Sample}} & \multicolumn{2}{c}{\textbf{Exogenous Only}} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    print(r"\textbf{Category} & \textbf{N} & \textbf{CAR} & \textbf{N} & \textbf{CAR} \\")
    print(r"\midrule")
    print(f"Infra\\_Negative & {infra_full_result.n_events} & {infra_full_result.statistic:.1%} & {infra_exog_result.n_events} & {infra_exog_result.statistic:.1%} \\\\")
    print(f"Reg\\_Negative & {reg_full_result.n_events} & {reg_full_result.statistic:.1%} & {reg_exog_result.n_events} & {reg_exog_result.statistic:.1%} \\\\")
    print(r"\midrule")
    print(f"$\\Delta$ & --- & {diff_full_result.statistic:+.1%} & --- & ", end='')
    if has_diff_test:
        print(f"{diff_exog_result.statistic:+.1%} \\\\")
    else:
        print(r"--- \\")
    print(f"p-value & --- & {diff_full_result.p_value:.3f} & --- & ", end='')
    if has_diff_test:
        print(f"{diff_exog_result.p_value:.3f} \\\\")
    else:
        print(r"--- \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print(f"\n\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
