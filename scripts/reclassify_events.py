#!/usr/bin/env python3
"""
Reclassify events into 4 categories based on peer review feedback.

Categories:
- Infra_Negative: Infrastructure failures, hacks, collapses
- Infra_Positive: Upgrades, halvings, positive infrastructure
- Reg_Negative: Bans, enforcement, restrictions
- Reg_Positive: Approvals, clarity, adoption

This addresses the conflation issue identified in peer review.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / 'data'

# Manual reclassification based on event content
RECLASSIFICATION = {
    # Infrastructure NEGATIVE (failures, hacks, collapses)
    1: 'Infra_Negative',   # QuadrigaCX collapse
    3: 'Infra_Negative',   # Binance hack 2019
    7: 'Infra_Negative',   # Black Thursday crash
    18: 'Infra_Negative',  # Poly Network hack
    24: 'Infra_Negative',  # Terra/UST collapse
    25: 'Infra_Negative',  # Celsius freeze
    27: 'Infra_Negative',  # BNB bridge hack
    28: 'Infra_Negative',  # FTX bankruptcy
    29: 'Infra_Negative',  # USDC depeg
    43: 'Infra_Negative',  # Bybit hack 2025

    # Infrastructure POSITIVE (upgrades, scheduled events)
    5: 'Infra_Positive',   # Litecoin halving
    8: 'Infra_Positive',   # Bitcoin halving 2020
    11: 'Infra_Positive',  # ETH2 Beacon Chain
    17: 'Infra_Positive',  # ETH EIP-1559
    26: 'Infra_Positive',  # ETH Merge
    30: 'Infra_Positive',  # ETH Shanghai
    38: 'Infra_Positive',  # ETH Dencun
    39: 'Infra_Positive',  # Bitcoin halving 2024
    47: 'Infra_Positive',  # ETH Pectra

    # Infrastructure NEUTRAL/MIXED (remove from analysis)
    13: 'Exclude',  # Tesla buys BTC - corporate action, not infrastructure
    14: 'Exclude',  # Coinbase listing - exchange IPO, not infrastructure event
    21: 'Exclude',  # BTC Taproot - minor upgrade

    # Regulatory NEGATIVE (bans, enforcement)
    12: 'Reg_Negative',  # SEC v Ripple
    15: 'Reg_Negative',  # China mining ban
    19: 'Reg_Negative',  # China total ban
    22: 'Reg_Negative',  # Kazakhstan internet shutdown
    31: 'Reg_Negative',  # SEC v Binance
    32: 'Reg_Negative',  # SEC v Coinbase

    # Regulatory POSITIVE (approvals, clarity)
    6: 'Reg_Positive',   # China blockchain endorsement
    16: 'Reg_Positive',  # El Salvador BTC legal tender
    20: 'Reg_Positive',  # BITO ETF launch
    23: 'Reg_Positive',  # US crypto Executive Order
    33: 'Reg_Positive',  # BlackRock ETF filing
    34: 'Reg_Positive',  # Grayscale wins court case
    35: 'Reg_Positive',  # MiCA passage
    36: 'Reg_Negative',  # Binance/CZ settlement (enforcement action)
    37: 'Reg_Positive',  # BTC ETF approval
    40: 'Reg_Positive',  # ETH ETF approval
    41: 'Reg_Positive',  # MiCA phase 1
    42: 'Reg_Positive',  # ETH ETFs trading
    44: 'Reg_Positive',  # SEC drops Coinbase
    45: 'Reg_Positive',  # OCC interpretive letter
    46: 'Reg_Positive',  # SEC stablecoin clarity
    48: 'Reg_Positive',  # GENIUS Act
    49: 'Reg_Positive',  # SEC in-kind ETPs
    50: 'Reg_Positive',  # XRP case ends
}


def reclassify_events():
    """Load events and add new classification."""
    events_file = DATA_DIR / 'events_master.json'

    with open(events_file) as f:
        data = json.load(f)

    # Add new classification
    for event in data['events']:
        event_id = event['event_id']
        new_type = RECLASSIFICATION.get(event_id)

        if new_type:
            event['type_original'] = event['type']
            event['type_detailed'] = new_type
            event['include_in_reanalysis'] = new_type != 'Exclude'
        else:
            event['type_detailed'] = event['type']
            event['include_in_reanalysis'] = True

    # Summary
    from collections import Counter
    detailed_counts = Counter(e['type_detailed'] for e in data['events'])

    print("=" * 60)
    print("EVENT RECLASSIFICATION")
    print("=" * 60)
    print("\nDetailed Classification Counts:")
    for cat, count in sorted(detailed_counts.items()):
        print(f"  {cat}: {count}")

    # Filter for reanalysis
    included = [e for e in data['events'] if e.get('include_in_reanalysis', True)]
    excluded = [e for e in data['events'] if not e.get('include_in_reanalysis', True)]

    print(f"\nIncluded in reanalysis: {len(included)}")
    print(f"Excluded: {len(excluded)}")

    for e in excluded:
        print(f"  - [{e['event_id']}] {e['label']}")

    # Hypothesis-relevant subset (failures vs enforcement)
    infra_neg = [e for e in included if e['type_detailed'] == 'Infra_Negative']
    reg_neg = [e for e in included if e['type_detailed'] == 'Reg_Negative']

    print(f"\n--- HYPOTHESIS-RELEVANT SUBSET ---")
    print(f"Infrastructure Failures (negative shocks): {len(infra_neg)}")
    print(f"Regulatory Enforcement (negative shocks): {len(reg_neg)}")

    # Save updated events
    output_file = DATA_DIR / 'events_reclassified.json'
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved to: {output_file}")

    return data


if __name__ == "__main__":
    reclassify_events()
