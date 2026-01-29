# Sentiment Without Structure

**Differential Market Responses to Infrastructure vs Regulatory Events in Cryptocurrency Markets**

[![DOI](https://img.shields.io/badge/DOI-pending-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract

We investigate differential market responses to infrastructure versus regulatory events in cryptocurrency markets using event study methodology with 4-category event classification. From 50 candidate events (2019–2025), 31 meet our impact and estimation-data criteria across 4 cryptocurrencies: BTC, ETH, SOL, and ADA.

**Primary finding**: Infrastructure failures (N=8) produce mean CAR of −7.6% and regulatory enforcement (N=7) produces mean CAR of −11.1%. The difference of +3.6 pp has 95% CI [−25.3%, +30.9%], p = 0.81. **This is a null finding**: markets respond similarly to both shock types when controlling for event valence.

## Key Contributions

1. **Methodological**: 4-category event classification (infrastructure/regulatory × positive/negative)
2. **Statistical**: Event-level block bootstrap respecting cross-sectional correlation
3. **Empirical**: Null finding robust across 8 alternative specifications

## Repository Structure

```
├── paper/
│   ├── sentiment-without-structure.tex    # LaTeX source
│   ├── sentiment-without-structure.pdf    # Compiled paper
│   └── references.bib                     # Bibliography
├── src/
│   ├── config.py                          # Configuration
│   ├── data_fetcher.py                    # Binance OHLCV fetcher
│   ├── event_study.py                     # Event study models
│   └── robustness.py                      # Bootstrap & robustness checks
├── scripts/
│   ├── run_main_analysis.py               # Primary analysis
│   ├── run_corrected_bootstrap.py         # Event-equal-weighted bootstrap
│   ├── run_im_test.py                     # Ibragimov-Müller test
│   ├── run_nonoverlap_analysis.py         # Overlap robustness
│   └── run_black_thursday_sensitivity.py  # Classification sensitivity
├── data/
│   └── events_reclassified.json           # Event sample with classifications
└── outputs/
    └── tables/                            # JSON results for all analyses
```

## Replication

```bash
# Install dependencies
pip install pandas numpy scipy requests

# Run main analysis
python scripts/run_main_analysis.py

# Run robustness checks
python scripts/run_corrected_bootstrap.py
python scripts/run_im_test.py
python scripts/run_nonoverlap_analysis.py
```

## Software

- Python 3.11
- NumPy 1.26, SciPy 1.12, pandas 2.1
- All tests: two-sided α = 0.05
- Bootstrap: B = 5,000 replications, percentile CIs

## Citation

```bibtex
@article{farzulla2026sentiment,
  title={Sentiment Without Structure: Differential Market Responses to
         Infrastructure vs Regulatory Events in Cryptocurrency Markets},
  author={Farzulla, Murad},
  journal={Working Paper},
  year={2026},
  url={https://github.com/studiofarzulla/sentiment-without-structure}
}
```

## Author

**Murad Farzulla**
King's College London
[ORCID: 0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)

## License

MIT License - see [LICENSE](LICENSE) for details.
