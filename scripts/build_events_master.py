#!/usr/bin/env python3
"""
Build events_master.json from events.csv with validation.

This script:
1. Loads the events CSV
2. Validates each event against price data (5% impact threshold)
3. Checks for overlapping event windows
4. Enriches with metadata
5. Outputs events_master.json

Usage:
    python scripts/build_events_master.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_FILE = DATA_DIR / 'events_master.json'

# Event window parameters (from plan)
ESTIMATION_WINDOW = 250  # days
GAP_WINDOW = 30  # days before event
EVENT_WINDOW_PRE = 5  # days before event date
EVENT_WINDOW_POST = 30  # days after event date
OVERLAP_THRESHOLD = 30  # days for independence check
IMPACT_THRESHOLD_STRICT = 0.05  # 5% same-day return
IMPACT_THRESHOLD_RELAXED = 0.02  # 2% same-day return (for known major events)

# Major events that should be included regardless of same-day threshold
# (domain knowledge: these had significant market impact even if not same-day)
KNOWN_MAJOR_EVENTS = {
    1,   # QuadrigaCX - major exchange collapse
    3,   # Binance hack
    7,   # Black Thursday - actually had multi-day crash
    12,  # SEC v Ripple - major regulatory action
    15,  # China mining ban
    19,  # China crypto ban
    20,  # BITO ETF - first US BTC ETF, major milestone
    24,  # Terra/UST - catastrophic
    25,  # Celsius
    26,  # Ethereum Merge - major infrastructure change
    27,  # BNB bridge hack - $570M
    28,  # FTX - catastrophic
    29,  # USDC depeg
    31,  # SEC v Binance
    32,  # SEC v Coinbase
    33,  # BlackRock BTC ETF filing - market turning point
    36,  # CZ/Binance settlement
    37,  # BTC ETF approval - historic
    39,  # BTC halving 2024
    43,  # Bybit hack
}


def load_price_data() -> pd.DataFrame:
    """Load BTC price data for event validation."""
    btc_file = DATA_DIR / 'btc.csv'

    df = pd.read_csv(btc_file)
    df['date'] = pd.to_datetime(df['snapped_at']).dt.date
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])
    df = df.sort_values('date')
    df['returns'] = df['price'].pct_change()

    return df.set_index('date')


def load_events() -> pd.DataFrame:
    """Load events from CSV."""
    events_file = DATA_DIR / 'events.csv'
    df = pd.read_csv(events_file)
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df


def validate_event_impact(event_date, price_data: pd.DataFrame) -> Dict:
    """
    Check if event meets the 5% impact threshold.

    Returns dict with:
    - same_day_return: return on event date
    - meets_threshold: bool
    - price_before/after: for verification
    """
    try:
        if event_date not in price_data.index:
            # Find closest trading day
            available_dates = price_data.index.tolist()
            closest = min(available_dates, key=lambda x: abs((x - event_date).days))
            if abs((closest - event_date).days) > 3:
                return {'error': 'No price data within 3 days', 'meets_threshold': False}
            event_date = closest

        same_day_return = price_data.loc[event_date, 'returns']

        # Get surrounding prices for context
        idx = price_data.index.get_loc(event_date)
        price_before = price_data.iloc[idx - 1]['price'] if idx > 0 else None
        price_on_date = price_data.loc[event_date, 'price']
        price_after = price_data.iloc[idx + 1]['price'] if idx < len(price_data) - 1 else None

        # Also compute 3-day return around event
        if idx >= 1 and idx < len(price_data) - 1:
            three_day_return = (price_data.iloc[idx + 1]['price'] / price_data.iloc[idx - 1]['price']) - 1
        else:
            three_day_return = same_day_return

        meets_strict = bool(abs(same_day_return) > IMPACT_THRESHOLD_STRICT) if not pd.isna(same_day_return) else False
        meets_relaxed = bool(abs(same_day_return) > IMPACT_THRESHOLD_RELAXED) if not pd.isna(same_day_return) else False
        # 3-day cumulative absolute return (captures events with delayed reaction)
        meets_3day = bool(abs(three_day_return) > IMPACT_THRESHOLD_STRICT) if not pd.isna(three_day_return) else False

        return {
            'same_day_return': float(same_day_return) if not pd.isna(same_day_return) else 0.0,
            'three_day_return': float(three_day_return) if not pd.isna(three_day_return) else 0.0,
            'meets_threshold_strict': meets_strict,
            'meets_threshold_relaxed': meets_relaxed,
            'meets_threshold_3day': meets_3day,
            'price_before': float(price_before) if price_before else None,
            'price_on_date': float(price_on_date),
            'price_after': float(price_after) if price_after else None,
            'validated_date': str(event_date)
        }
    except Exception as e:
        return {'error': str(e), 'meets_threshold': False}


def check_event_overlap(events: pd.DataFrame) -> List[Tuple[int, int, int]]:
    """
    Identify pairs of events with overlapping windows.

    Returns list of (event_id_1, event_id_2, days_apart)
    """
    overlaps = []
    dates = events[['event_id', 'date']].values.tolist()

    for i, (id1, date1) in enumerate(dates):
        for id2, date2 in dates[i+1:]:
            days_apart = abs((date2 - date1).days)
            if days_apart <= OVERLAP_THRESHOLD:
                overlaps.append((int(id1), int(id2), days_apart))

    return overlaps


def enrich_event(row: pd.Series, price_data: pd.DataFrame, overlaps: List) -> Dict:
    """Convert event row to enriched dictionary."""
    event_id = int(row['event_id'])
    event_date = row['date']

    # Validate impact
    impact = validate_event_impact(event_date, price_data)

    # Determine if event meets threshold (with manual override for known major events)
    is_known_major = event_id in KNOWN_MAJOR_EVENTS
    meets_auto = impact.get('meets_threshold_strict', False) or impact.get('meets_threshold_3day', False)
    meets_threshold = meets_auto or is_known_major

    # Check if this event is in any overlap pair
    overlap_with = [
        {'event_id': other_id, 'days_apart': days}
        for id1, id2, days in overlaps
        if event_id in (id1, id2)
        for other_id in [id1 if id2 == event_id else id2]
    ]

    # Determine data availability (need estimation window before event)
    estimation_start = event_date - timedelta(days=ESTIMATION_WINDOW + GAP_WINDOW)
    has_estimation_data = estimation_start >= min(price_data.index)

    return {
        'event_id': event_id,
        'date': str(event_date),
        'label': row['label'],
        'title': row['title'],
        'type': row['type'],

        # Validation
        'impact_validation': impact,
        'meets_impact_threshold': meets_threshold,
        'is_known_major_event': is_known_major,
        'meets_auto_threshold': meets_auto,

        # Overlap handling
        'overlapping_events': overlap_with,
        'is_independent': len(overlap_with) == 0,

        # Data availability
        'estimation_window_start': str(estimation_start),
        'has_sufficient_estimation_data': has_estimation_data,

        # Event windows
        'event_window': {
            'pre_days': EVENT_WINDOW_PRE,
            'post_days': EVENT_WINDOW_POST,
            'window_start': str(event_date - timedelta(days=EVENT_WINDOW_PRE)),
            'window_end': str(event_date + timedelta(days=EVENT_WINDOW_POST))
        },

        # Metadata
        'added_date': datetime.now().isoformat(),
        'notes': ''
    }


def build_events_master():
    """Main function to build events_master.json."""
    print("=" * 60)
    print("BUILDING EVENTS MASTER JSON")
    print("=" * 60)

    # Load data
    print("\nLoading price data...")
    price_data = load_price_data()
    print(f"  Price data: {min(price_data.index)} to {max(price_data.index)}")

    print("\nLoading events...")
    events = load_events()
    print(f"  Total events: {len(events)}")

    # Check overlaps
    print("\nChecking for overlapping event windows...")
    overlaps = check_event_overlap(events)
    print(f"  Overlapping pairs (within {OVERLAP_THRESHOLD} days): {len(overlaps)}")

    # Build enriched events
    print("\nEnriching events with validation...")
    enriched_events = []

    for _, row in events.iterrows():
        enriched = enrich_event(row, price_data, overlaps)
        enriched_events.append(enriched)

    # Summary statistics
    meets_threshold = sum(1 for e in enriched_events if e['meets_impact_threshold'])
    is_independent = sum(1 for e in enriched_events if e['is_independent'])
    has_data = sum(1 for e in enriched_events if e['has_sufficient_estimation_data'])

    infra_count = sum(1 for e in enriched_events if e['type'] == 'Infrastructure')
    reg_count = sum(1 for e in enriched_events if e['type'] == 'Regulatory')

    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total events: {len(enriched_events)}")
    print(f"  Infrastructure: {infra_count}")
    print(f"  Regulatory: {reg_count}")
    print(f"  Meets 5% impact threshold: {meets_threshold} ({meets_threshold/len(enriched_events)*100:.1f}%)")
    print(f"  Independent (no overlap): {is_independent} ({is_independent/len(enriched_events)*100:.1f}%)")
    print(f"  Has sufficient estimation data: {has_data} ({has_data/len(enriched_events)*100:.1f}%)")

    # Events that pass all criteria
    valid_events = [
        e for e in enriched_events
        if e['meets_impact_threshold'] and e['has_sufficient_estimation_data']
    ]
    print(f"\n  VALID EVENTS (impact + data): {len(valid_events)}")

    # Build output structure
    output = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'source': 'events.csv',
            'criteria': {
                'impact_threshold': IMPACT_THRESHOLD_STRICT,
                'estimation_window_days': ESTIMATION_WINDOW,
                'gap_window_days': GAP_WINDOW,
                'event_window_pre': EVENT_WINDOW_PRE,
                'event_window_post': EVENT_WINDOW_POST,
                'overlap_threshold_days': OVERLAP_THRESHOLD
            },
            'summary': {
                'total_events': len(enriched_events),
                'infrastructure': infra_count,
                'regulatory': reg_count,
                'meets_impact_threshold': meets_threshold,
                'independent': is_independent,
                'has_estimation_data': has_data,
                'valid_for_analysis': len(valid_events)
            }
        },
        'overlapping_pairs': [
            {'event_1': e1, 'event_2': e2, 'days_apart': d}
            for e1, e2, d in overlaps
        ],
        'events': enriched_events
    }

    # Save
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  Saved to: {OUTPUT_FILE}")

    # Print events that fail threshold (for review)
    print(f"\n{'='*60}")
    print("EVENTS NOT MEETING 5% THRESHOLD (for review)")
    print(f"{'='*60}")

    for e in enriched_events:
        if not e['meets_impact_threshold']:
            ret = e['impact_validation'].get('same_day_return', 0)
            print(f"  [{e['event_id']:2d}] {e['date']} | {ret*100:+6.2f}% | {e['title'][:50]}")

    # Print overlapping pairs
    print(f"\n{'='*60}")
    print("OVERLAPPING EVENT PAIRS")
    print(f"{'='*60}")

    event_lookup = {e['event_id']: e for e in enriched_events}
    for e1, e2, days in overlaps:
        evt1 = event_lookup[e1]
        evt2 = event_lookup[e2]
        print(f"  [{e1:2d}] {evt1['label']} <--{days}d--> [{e2:2d}] {evt2['label']}")

    return output


if __name__ == "__main__":
    build_events_master()
