# Same Returns, Different Risks

**How Cryptocurrency Markets Process Infrastructure vs Regulatory Shocks**

[![arXiv](https://img.shields.io/badge/arXiv-2602.07046-b31b1b.svg)](https://arxiv.org/abs/2602.07046)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18099609-blue.svg)](https://doi.org/10.5281/zenodo.18099609)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status](https://img.shields.io/badge/Status-Transferring_to_Comp._Econ.-orange.svg)]()

**Working Paper DAI-2507** | [Dissensus AI](https://dissensus.ai)

## Abstract

Cryptocurrency markets face two structurally distinct threat vectors -- infrastructure failures (exchange hacks, protocol exploits, bridge collapses) and regulatory enforcement (SEC actions, exchange shutdowns, trading bans) -- yet whether markets price these risks differently remains an open question. Using a 4-category event classification (infrastructure/regulatory x positive/negative) applied to 31 events across Bitcoin, Ethereum, Solana, and Cardano (2019--2025), we find no statistically significant difference in return-level market response: infrastructure failures produce mean CAR of -7.6% versus -11.1% for regulatory enforcement (difference = +3.6 pp, p = 0.81). However, a companion GJR-GARCH-X analysis of the same event taxonomy reveals that infrastructure events generate 5.7x larger conditional variance impacts than regulatory shocks (p = 0.0008, Cohen's d = 2.753), with crisis-regime amplification reaching 5x for infrastructure sensitivity while regulatory effects remain flat across regimes. Together, these results establish that the market's risk differentiation mechanism operates through the second moment, not the first: returns appear interchangeable, but the variance structure reveals sharply distinct processing of bounded (infrastructure) versus unbounded (regulatory) uncertainty.

## Key Findings

| Finding | Result |
|---------|--------|
| Infrastructure failures (N=8) mean CAR | -7.6% |
| Regulatory enforcement (N=7) mean CAR | -11.1% |
| Difference | +3.6 pp (p = 0.81) |
| Primary result | Null return-level finding -- markets respond similarly |
| Companion GARCH result | 5.7x infrastructure variance multiplier (p = 0.0008) |
| Joint interpretation | Risk differentiation lives in the second moment |
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

## Companion Paper

This paper forms a pair with [Infrastructure vs Regulatory Shocks: Asymmetric Volatility Response in Cryptocurrency Markets](https://doi.org/10.21203/rs.3.rs-8323026/v1) (DAI-2506), which examines the same event taxonomy through GJR-GARCH-X conditional variance modeling.

## Keywords

Cryptocurrency, Event Study, Block Bootstrap, Volatility Asymmetry, Infrastructure Risk, Multi-Moment Analysis

## Citation

```bibtex
@article{farzulla2025samereturnsdifferentrisks,
  title={Same Returns, Different Risks: How Cryptocurrency Markets Process Infrastructure vs Regulatory Shocks},
  author={Farzulla, Murad},
  journal={arXiv preprint arXiv:2602.07046},
  year={2025},
  doi={10.48550/arXiv.2602.07046}
}
```

## Authors

- **Murad Farzulla** -- [Dissensus AI](https://dissensus.ai) & King's College London
  - ORCID: [0009-0002-7164-8704](https://orcid.org/0009-0002-7164-8704)
  - Email: murad@dissensus.ai

## License

Paper content: [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
