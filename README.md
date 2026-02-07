# Sentiment Without Structure

**Differential Liquidity Response to Infrastructure vs Regulatory Events in Cryptocurrency Markets**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18099609-blue.svg)](https://doi.org/10.5281/zenodo.18099609)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-With_Editor-yellow.svg)](https://doi.org/10.5281/zenodo.18099609)

**Working Paper DAI-2507** | [Dissensus AI](https://dissensus.ai)

## Abstract

We investigate differential market responses to infrastructure versus regulatory events in cryptocurrency markets using event study methodology with 4-category event classification. From 50 candidate events (2019--2025), 31 meet inclusion criteria across Bitcoin, Ethereum, Solana, and Cardano. We employ constant mean and market-adjusted models with event-level block bootstrap confidence intervals that properly account for cross-sectional correlation.

Our primary comparison focuses on negative-valence events: infrastructure failures (8 events) versus regulatory enforcement (7 events). Infrastructure failures produce mean Cumulative Abnormal Return (CAR) of -7.6% (95% CI: [-25.8%, +11.3%]) and regulatory enforcement produces mean CAR of -11.1% (CI: [-31.0%, +10.7%]). The difference of +3.6 percentage points has CI [-25.3%, +30.9%], p = 0.81 -- a null finding indicating markets respond similarly to both shock types when controlling for valence.

Robustness checks confirm consistent results across window specifications, leave-one-out exclusion of major events (FTX, Terra), and alternative market model specifications. The 4-category classification addresses prior conflation of upgrades with failures. This exploratory analysis should be treated as hypothesis-generating.

## Key Findings

| Finding | Result |
|---------|--------|
| Infrastructure failures (N=8) mean CAR | -7.6% |
| Regulatory enforcement (N=7) mean CAR | -11.1% |
| Difference | +3.6 pp (p = 0.81) |
| Primary result | Null finding -- markets respond similarly to both shock types |
| Robustness | Consistent across 8 alternative specifications |

## Repository Structure

```
sentiment-without-structure/
├── paper/                           # LaTeX source and compiled PDF
├── src/                             # Analysis code
│   ├── config.py                    # Configuration
│   ├── data_fetcher.py              # Binance OHLCV fetcher
│   ├── event_study.py               # Event study models
│   └── robustness.py                # Bootstrap & robustness checks
├── scripts/                         # Analysis scripts
│   ├── run_main_analysis.py         # Primary analysis
│   ├── run_corrected_bootstrap.py   # Event-equal-weighted bootstrap
│   ├── run_im_test.py               # Ibragimov-Muller test
│   ├── run_nonoverlap_analysis.py   # Overlap robustness
│   └── run_black_thursday_sensitivity.py  # Classification sensitivity
├── data/                            # Event sample with classifications
└── outputs/                         # JSON results for all analyses
```

## Replication

```bash
pip install pandas numpy scipy requests

python scripts/run_main_analysis.py
python scripts/run_corrected_bootstrap.py
python scripts/run_im_test.py
python scripts/run_nonoverlap_analysis.py
```

## Keywords

Cryptocurrency, Event Study, Regulation, Infrastructure Risk, Block Bootstrap

## Citation

```bibtex
@article{farzulla2026sentiment,
  title={Sentiment Without Structure: Differential Liquidity Response to Infrastructure vs Regulatory Events in Cryptocurrency Markets},
  author={Farzulla, Murad},
  year={2026},
  doi={10.5281/zenodo.18099609},
  url={https://doi.org/10.5281/zenodo.18099609},
  note={With Editor at Digital Finance (Springer)}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
