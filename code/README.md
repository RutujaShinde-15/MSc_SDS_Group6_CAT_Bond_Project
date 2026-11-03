# China replication of Tang, Wen, Ling & Zhang (2023), Risks 11: 151

Rebuilds the paper's whole pipeline from the public CFDID database (Cui et al. 2026).

Run (Colab or local, Python 3 with numpy, scipy, pandas, matplotlib, statsmodels, sympy):

    cd code
    python run_all.py

| Step | Paper part | Output |
|---|---|---|
| 00_compile.py | Sec. 3 data | data/clean/events.csv (328 province-level major rainstorms), annual_counts.csv |
| 02_descriptive.py | Sec. 3.1, Table 3, Fig. 2 | Table 3: paper vs new compilation vs team v1 |
| 03_thresholds.py | Sec. 3.2, Figs. 3-4 | thresholds from scratch vs (160, 12, 15) |
| 04_marginals.py | Eq. 12-13, Table 4, Fig. 5 | Beta-GP fits at both threshold sets |
| 05_copula.py | Sec. 3.3, Table 5 | nested Gumbel/Clayton/Frank by 3-D MLE, bootstrap KS |
| 06_frequency.py | Sec. 3.4, Figs. 6-7 | ADF, ARMA, forecast 2021-23 vs actual |
| 07_price.py | Sec. 3.5, Tables 6-7, Fig. 8 | Monte Carlo prices with CIR discounting |
| 08_section3.py | all of Sec. 3 | Final Tables 3-7 and Figures 2-8 in outputs/section3 |

cat_model.py holds every formula; config.py every setting and the paper's published numbers.
test_toolbox.py checks the copula code on simulated data.
