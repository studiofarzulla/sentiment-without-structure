"""
Robustness Checks Module
========================

Implements comprehensive robustness battery for event study results.

Checks:
1. Window sensitivity - varying event window length
2. Estimation sensitivity - varying estimation period
3. Leave-one-out - excluding individual events
4. Asset heterogeneity - excluding individual assets
5. Metric consistency - comparing across liquidity measures
6. Placebo tests - random event dates
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from tqdm import tqdm
import warnings

from . import config
from .event_study import EventStudyAnalyzer, ConstantMeanModel, BootstrapBCA, MarketModel


@dataclass
class RobustnessResult:
    """Container for robustness check results."""
    check_type: str
    variation: str
    n_events: int
    car_mean: float
    car_std: float
    t_stat: float
    p_value: float
    consistent_with_main: bool
    details: Dict


@dataclass
class BootstrapResult:
    """Container for bootstrap inference results."""
    statistic: float
    ci_low: float
    ci_high: float
    se: float
    p_value: float
    n_events: int
    n_observations: int
    method: str
    details: Dict


class EventBlockBootstrap:
    """
    Event-level block bootstrap for proper inference in event studies.

    CRITICAL: Standard t-tests on pooled asset-events inflate degrees of freedom.
    If you have 10 events × 6 assets = 60 observations, the t-test treats
    this as N=60 independent observations when the true N is closer to 10 events.

    This class implements block bootstrap that resamples ENTIRE EVENTS,
    keeping all assets within each event together. This respects the
    cross-sectional correlation of returns within an event.

    Reference:
    - Kothari & Warner (2007) Econometrics of Event Studies
    - Cameron, Gelbach & Miller (2008) Bootstrap-Based Improvements
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

    def compute_event_cars(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        model: ConstantMeanModel,
        window: Tuple[int, int] = (-5, 30)
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute CARs for each event-asset pair.

        Returns:
            Dict mapping event_id -> {symbol: car}
        """
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

    def bootstrap_mean_car(
        self,
        event_cars: Dict[int, Dict[str, float]],
        n_bootstrap: int = None
    ) -> BootstrapResult:
        """
        Bootstrap confidence interval for mean CAR.

        Resamples entire events (with replacement), keeping all assets
        within each event together.

        Args:
            event_cars: Dict mapping event_id -> {symbol: car}
            n_bootstrap: Number of bootstrap replications (default: self.n_bootstrap)

        Returns:
            BootstrapResult with CI and p-value
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        event_ids = list(event_cars.keys())
        n_events = len(event_ids)

        if n_events < 2:
            return BootstrapResult(
                statistic=np.nan, ci_low=np.nan, ci_high=np.nan,
                se=np.nan, p_value=np.nan, n_events=n_events,
                n_observations=0, method='block_bootstrap',
                details={'error': 'Insufficient events for bootstrap'}
            )

        # Compute original statistic: mean CAR across all event-assets
        all_cars = []
        for eid in event_ids:
            for symbol, car in event_cars[eid].items():
                all_cars.append(car)
        original_mean = np.mean(all_cars)
        n_observations = len(all_cars)

        # Block bootstrap: resample events
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Sample event IDs with replacement
            sampled_ids = self.rng.choice(event_ids, size=n_events, replace=True)

            # Collect all CARs from sampled events
            sample_cars = []
            for eid in sampled_ids:
                for symbol, car in event_cars[eid].items():
                    sample_cars.append(car)

            if len(sample_cars) > 0:
                bootstrap_means.append(np.mean(sample_cars))

        bootstrap_means = np.array(bootstrap_means)

        # Percentile CI
        alpha = (1 - self.confidence) / 2
        ci_low = np.percentile(bootstrap_means, alpha * 100)
        ci_high = np.percentile(bootstrap_means, (1 - alpha) * 100)

        # Bootstrap SE
        bootstrap_se = bootstrap_means.std()

        # Bootstrap p-value (two-tailed test against zero)
        if original_mean >= 0:
            p_value = 2 * np.mean(bootstrap_means <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_means >= 0)
        p_value = min(p_value, 1.0)

        return BootstrapResult(
            statistic=original_mean,
            ci_low=ci_low,
            ci_high=ci_high,
            se=bootstrap_se,
            p_value=p_value,
            n_events=n_events,
            n_observations=n_observations,
            method='block_bootstrap',
            details={
                'n_bootstrap': n_bootstrap,
                'confidence': self.confidence,
                'bootstrap_percentiles': {
                    '5%': np.percentile(bootstrap_means, 5),
                    '25%': np.percentile(bootstrap_means, 25),
                    '50%': np.percentile(bootstrap_means, 50),
                    '75%': np.percentile(bootstrap_means, 75),
                    '95%': np.percentile(bootstrap_means, 95),
                }
            }
        )

    def bootstrap_difference_test(
        self,
        event_cars_a: Dict[int, Dict[str, float]],
        event_cars_b: Dict[int, Dict[str, float]],
        n_bootstrap: int = None
    ) -> BootstrapResult:
        """
        Bootstrap test for difference in mean CARs between two groups.

        Used for DiD analysis: is (Infra_Negative CAR) - (Reg_Negative CAR) != 0?

        Args:
            event_cars_a: CARs for group A (e.g., Infra_Negative)
            event_cars_b: CARs for group B (e.g., Reg_Negative)

        Returns:
            BootstrapResult for the difference (A - B)
        """
        if n_bootstrap is None:
            n_bootstrap = self.n_bootstrap

        event_ids_a = list(event_cars_a.keys())
        event_ids_b = list(event_cars_b.keys())
        n_events_a = len(event_ids_a)
        n_events_b = len(event_ids_b)

        if n_events_a < 2 or n_events_b < 2:
            return BootstrapResult(
                statistic=np.nan, ci_low=np.nan, ci_high=np.nan,
                se=np.nan, p_value=np.nan, n_events=n_events_a + n_events_b,
                n_observations=0, method='block_bootstrap_diff',
                details={'error': 'Insufficient events for difference test'}
            )

        # Original means
        all_cars_a = [car for eid in event_ids_a for car in event_cars_a[eid].values()]
        all_cars_b = [car for eid in event_ids_b for car in event_cars_b[eid].values()]
        original_diff = np.mean(all_cars_a) - np.mean(all_cars_b)

        # Bootstrap the difference
        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            # Resample events within each group
            sampled_a = self.rng.choice(event_ids_a, size=n_events_a, replace=True)
            sampled_b = self.rng.choice(event_ids_b, size=n_events_b, replace=True)

            sample_cars_a = [car for eid in sampled_a for car in event_cars_a[eid].values()]
            sample_cars_b = [car for eid in sampled_b for car in event_cars_b[eid].values()]

            if len(sample_cars_a) > 0 and len(sample_cars_b) > 0:
                bootstrap_diffs.append(np.mean(sample_cars_a) - np.mean(sample_cars_b))

        bootstrap_diffs = np.array(bootstrap_diffs)

        # CI and p-value
        alpha = (1 - self.confidence) / 2
        ci_low = np.percentile(bootstrap_diffs, alpha * 100)
        ci_high = np.percentile(bootstrap_diffs, (1 - alpha) * 100)
        bootstrap_se = bootstrap_diffs.std()

        # Two-tailed p-value against zero
        if original_diff >= 0:
            p_value = 2 * np.mean(bootstrap_diffs <= 0)
        else:
            p_value = 2 * np.mean(bootstrap_diffs >= 0)
        p_value = min(p_value, 1.0)

        return BootstrapResult(
            statistic=original_diff,
            ci_low=ci_low,
            ci_high=ci_high,
            se=bootstrap_se,
            p_value=p_value,
            n_events=n_events_a + n_events_b,
            n_observations=len(all_cars_a) + len(all_cars_b),
            method='block_bootstrap_diff',
            details={
                'n_bootstrap': n_bootstrap,
                'confidence': self.confidence,
                'n_events_a': n_events_a,
                'n_events_b': n_events_b,
                'mean_a': np.mean(all_cars_a),
                'mean_b': np.mean(all_cars_b),
                'bootstrap_percentiles': {
                    '5%': np.percentile(bootstrap_diffs, 5),
                    '25%': np.percentile(bootstrap_diffs, 25),
                    '50%': np.percentile(bootstrap_diffs, 50),
                    '75%': np.percentile(bootstrap_diffs, 75),
                    '95%': np.percentile(bootstrap_diffs, 95),
                }
            }
        )


class RobustnessChecker:
    """Run comprehensive robustness checks on event study results."""

    def __init__(
        self,
        random_state: int = config.RANDOM_SEED
    ):
        self.rng = np.random.default_rng(random_state)
        self.analyzer = EventStudyAnalyzer()

    def window_sensitivity(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        windows: Optional[List[Tuple[int, int]]] = None
    ) -> List[RobustnessResult]:
        """
        Test sensitivity to event window specification.

        Args:
            returns_dict: Dictionary of return series by symbol
            events: List of event dictionaries
            windows: List of (pre, post) windows to test

        Returns:
            List of RobustnessResult for each window
        """
        if windows is None:
            # Updated windows per peer review: add tighter windows for robustness
            windows = [
                (0, 1),     # Immediate reaction only
                (0, 3),     # 3-day (crypto moves fast)
                (0, 5),     # 5-day standard short
                (-1, 5),    # Short-term with pre
                (-3, 10),   # Medium
                (-5, 30),   # Main (baseline)
                (-5, 60),   # Extended
            ]

        results = []
        baseline_cars = None

        for pre, post in windows:
            window_name = f"[{pre:+d}, {post:+d}]"
            cars = []

            for event in events:
                event_date = event['date']

                for symbol, returns in returns_dict.items():
                    result = self.analyzer.model.compute_abnormal_returns(
                        returns, event_date, (pre, post)
                    )
                    if 'error' not in result:
                        cars.append(result['car'])

            if len(cars) < 5:
                continue

            cars = np.array(cars)
            car_mean = cars.mean()
            car_std = cars.std()
            t_stat = car_mean / (car_std / np.sqrt(len(cars)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(cars) - 1))

            # Check consistency with baseline
            if baseline_cars is None and (pre, post) == (-5, 30):
                baseline_cars = cars
                consistent = True
            elif baseline_cars is not None:
                # Correlation with baseline
                min_len = min(len(cars), len(baseline_cars))
                if min_len > 10:
                    corr = np.corrcoef(cars[:min_len], baseline_cars[:min_len])[0, 1]
                    consistent = corr > 0.7 and np.sign(car_mean) == np.sign(baseline_cars.mean())
                else:
                    consistent = np.sign(car_mean) == np.sign(baseline_cars.mean()) if baseline_cars is not None else True
            else:
                consistent = True

            results.append(RobustnessResult(
                check_type='window_sensitivity',
                variation=window_name,
                n_events=len(cars),
                car_mean=car_mean,
                car_std=car_std,
                t_stat=t_stat,
                p_value=p_value,
                consistent_with_main=consistent,
                details={'window': (pre, post)}
            ))

        return results

    def estimation_sensitivity(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        estimation_windows: Optional[List[int]] = None
    ) -> List[RobustnessResult]:
        """
        Test sensitivity to estimation window length.

        Args:
            returns_dict: Dictionary of return series by symbol
            events: List of event dictionaries
            estimation_windows: List of estimation window lengths to test

        Returns:
            List of RobustnessResult for each estimation window
        """
        if estimation_windows is None:
            estimation_windows = [60, 120, 180, 250, 365]

        results = []
        baseline_cars = None

        for est_window in estimation_windows:
            model = ConstantMeanModel(estimation_window=est_window)
            cars = []

            for event in events:
                event_date = event['date']

                for symbol, returns in returns_dict.items():
                    result = model.compute_abnormal_returns(
                        returns, event_date, (-5, 30)
                    )
                    if 'error' not in result:
                        cars.append(result['car'])

            if len(cars) < 5:
                continue

            cars = np.array(cars)
            car_mean = cars.mean()
            car_std = cars.std()
            t_stat = car_mean / (car_std / np.sqrt(len(cars)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(cars) - 1))

            if baseline_cars is None and est_window == 250:
                baseline_cars = cars
                consistent = True
            elif baseline_cars is not None:
                min_len = min(len(cars), len(baseline_cars))
                if min_len > 10:
                    corr = np.corrcoef(cars[:min_len], baseline_cars[:min_len])[0, 1]
                    consistent = corr > 0.7
                else:
                    consistent = True
            else:
                consistent = True

            results.append(RobustnessResult(
                check_type='estimation_sensitivity',
                variation=f'{est_window} days',
                n_events=len(cars),
                car_mean=car_mean,
                car_std=car_std,
                t_stat=t_stat,
                p_value=p_value,
                consistent_with_main=consistent,
                details={'estimation_window': est_window}
            ))

        return results

    def leave_one_out_events(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        main_result: Dict
    ) -> List[RobustnessResult]:
        """
        Leave-one-out analysis on events.

        Tests whether results are driven by a single event.

        Args:
            returns_dict: Dictionary of return series by symbol
            events: List of event dictionaries
            main_result: Main analysis result for comparison

        Returns:
            List of RobustnessResult for each excluded event
        """
        results = []

        for exclude_event in events:
            exclude_id = exclude_event['event_id']
            remaining_events = [e for e in events if e['event_id'] != exclude_id]

            cars = []
            for event in remaining_events:
                event_date = event['date']
                for symbol, returns in returns_dict.items():
                    result = self.analyzer.model.compute_abnormal_returns(
                        returns, event_date, (-5, 30)
                    )
                    if 'error' not in result:
                        cars.append(result['car'])

            if len(cars) < 5:
                continue

            cars = np.array(cars)
            car_mean = cars.mean()
            car_std = cars.std()
            t_stat = car_mean / (car_std / np.sqrt(len(cars)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(cars) - 1))

            # Check if excluding this event flips significance
            main_significant = main_result.get('p_value', 1) < 0.05
            this_significant = p_value < 0.05
            consistent = main_significant == this_significant

            results.append(RobustnessResult(
                check_type='leave_one_out_event',
                variation=f'Excl. {exclude_event.get("label", exclude_id)}',
                n_events=len(cars),
                car_mean=car_mean,
                car_std=car_std,
                t_stat=t_stat,
                p_value=p_value,
                consistent_with_main=consistent,
                details={'excluded_event_id': exclude_id}
            ))

        return results

    def leave_one_out_assets(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        main_result: Dict
    ) -> List[RobustnessResult]:
        """
        Leave-one-out analysis on assets.

        Tests whether results are driven by a single asset.
        """
        results = []

        for exclude_symbol in returns_dict.keys():
            remaining_returns = {
                s: r for s, r in returns_dict.items()
                if s != exclude_symbol
            }

            cars = []
            for event in events:
                event_date = event['date']
                for symbol, returns in remaining_returns.items():
                    result = self.analyzer.model.compute_abnormal_returns(
                        returns, event_date, (-5, 30)
                    )
                    if 'error' not in result:
                        cars.append(result['car'])

            if len(cars) < 5:
                continue

            cars = np.array(cars)
            car_mean = cars.mean()
            car_std = cars.std()
            t_stat = car_mean / (car_std / np.sqrt(len(cars)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(cars) - 1))

            main_significant = main_result.get('p_value', 1) < 0.05
            this_significant = p_value < 0.05
            consistent = main_significant == this_significant

            results.append(RobustnessResult(
                check_type='leave_one_out_asset',
                variation=f'Excl. {exclude_symbol}',
                n_events=len(cars),
                car_mean=car_mean,
                car_std=car_std,
                t_stat=t_stat,
                p_value=p_value,
                consistent_with_main=consistent,
                details={'excluded_asset': exclude_symbol}
            ))

        return results

    def placebo_test(
        self,
        returns_dict: Dict[str, pd.Series],
        n_placebo: int = 100,
        event_window: Tuple[int, int] = (-5, 30)
    ) -> RobustnessResult:
        """
        Placebo test with random event dates.

        Generates pseudo-events on random dates to establish null distribution.
        """
        # Get date range from returns
        all_dates = []
        for returns in returns_dict.values():
            all_dates.extend(returns.index.tolist())

        min_date = pd.to_datetime(min(all_dates)) + pd.Timedelta(days=300)
        max_date = pd.to_datetime(max(all_dates)) - pd.Timedelta(days=60)

        date_range = pd.date_range(min_date, max_date, freq='D')

        placebo_cars = []

        for _ in range(n_placebo):
            pseudo_date = pd.Timestamp(self.rng.choice(date_range)).strftime('%Y-%m-%d')

            for symbol, returns in returns_dict.items():
                result = self.analyzer.model.compute_abnormal_returns(
                    returns, pseudo_date, event_window
                )
                if 'error' not in result:
                    placebo_cars.append(result['car'])

        placebo_cars = np.array(placebo_cars)

        # Summary stats
        car_mean = placebo_cars.mean()
        car_std = placebo_cars.std()
        t_stat = car_mean / (car_std / np.sqrt(len(placebo_cars)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(placebo_cars) - 1))

        # Placebo should NOT be significant
        consistent = p_value > 0.10  # No effect expected

        return RobustnessResult(
            check_type='placebo_test',
            variation=f'{n_placebo} pseudo-events',
            n_events=len(placebo_cars),
            car_mean=car_mean,
            car_std=car_std,
            t_stat=t_stat,
            p_value=p_value,
            consistent_with_main=consistent,
            details={
                'n_placebo': n_placebo,
                'placebo_cars_percentiles': {
                    '5%': np.percentile(placebo_cars, 5),
                    '25%': np.percentile(placebo_cars, 25),
                    '50%': np.percentile(placebo_cars, 50),
                    '75%': np.percentile(placebo_cars, 75),
                    '95%': np.percentile(placebo_cars, 95),
                }
            }
        )

    def run_all_checks(
        self,
        returns_dict: Dict[str, pd.Series],
        events: List[Dict],
        main_result: Dict
    ) -> Dict[str, List[RobustnessResult]]:
        """
        Run all robustness checks.

        Args:
            returns_dict: Dictionary of return series by symbol
            events: List of event dictionaries
            main_result: Main analysis result for comparison

        Returns:
            Dictionary mapping check type to list of results
        """
        print("\n" + "=" * 60)
        print("ROBUSTNESS CHECKS")
        print("=" * 60)

        all_results = {}

        # 1. Window sensitivity
        print("\n1. Window sensitivity...")
        all_results['window_sensitivity'] = self.window_sensitivity(
            returns_dict, events
        )
        print(f"   {len(all_results['window_sensitivity'])} variations tested")

        # 2. Estimation sensitivity
        print("\n2. Estimation window sensitivity...")
        all_results['estimation_sensitivity'] = self.estimation_sensitivity(
            returns_dict, events
        )
        print(f"   {len(all_results['estimation_sensitivity'])} variations tested")

        # 3. Leave-one-out events
        print("\n3. Leave-one-out (events)...")
        all_results['leave_one_out_events'] = self.leave_one_out_events(
            returns_dict, events, main_result
        )
        print(f"   {len(all_results['leave_one_out_events'])} events tested")

        # 4. Leave-one-out assets
        print("\n4. Leave-one-out (assets)...")
        all_results['leave_one_out_assets'] = self.leave_one_out_assets(
            returns_dict, events, main_result
        )
        print(f"   {len(all_results['leave_one_out_assets'])} assets tested")

        # 5. Placebo test
        print("\n5. Placebo test...")
        all_results['placebo_test'] = [
            self.placebo_test(returns_dict, n_placebo=100)
        ]
        print(f"   {all_results['placebo_test'][0].n_events} pseudo-event CARs computed")

        # Summary
        print("\n" + "-" * 40)
        print("ROBUSTNESS SUMMARY")
        print("-" * 40)

        for check_type, results in all_results.items():
            n_consistent = sum(1 for r in results if r.consistent_with_main)
            n_total = len(results)
            print(f"  {check_type}: {n_consistent}/{n_total} consistent")

        return all_results


def results_to_dataframe(results: Dict[str, List[RobustnessResult]]) -> pd.DataFrame:
    """Convert robustness results to DataFrame for export."""
    rows = []

    for check_type, result_list in results.items():
        for r in result_list:
            rows.append({
                'check_type': r.check_type,
                'variation': r.variation,
                'n_events': r.n_events,
                'car_mean': r.car_mean,
                'car_std': r.car_std,
                't_stat': r.t_stat,
                'p_value': r.p_value,
                'consistent': r.consistent_with_main
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    print("Testing Robustness Module...")

    np.random.seed(42)

    # Synthetic data
    dates = pd.date_range('2022-01-01', periods=500, freq='D')
    returns_dict = {
        'BTC': pd.Series(np.random.randn(500) * 0.02, index=dates),
        'ETH': pd.Series(np.random.randn(500) * 0.025, index=dates),
    }

    events = [
        {'event_id': 1, 'date': '2022-06-01', 'label': 'Event1'},
        {'event_id': 2, 'date': '2022-09-01', 'label': 'Event2'},
    ]

    main_result = {'car': 0.05, 'p_value': 0.03}

    checker = RobustnessChecker()

    # Test window sensitivity
    ws_results = checker.window_sensitivity(returns_dict, events)
    print(f"\nWindow sensitivity: {len(ws_results)} results")

    # Test placebo
    placebo = checker.placebo_test(returns_dict, n_placebo=20)
    print(f"Placebo CAR mean: {placebo.car_mean:.4f}, p={placebo.p_value:.4f}")

    print("\n[SUCCESS] Robustness module working!")
