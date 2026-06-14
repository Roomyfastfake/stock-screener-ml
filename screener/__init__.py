"""Stock screener package.

Modules:
  indicators  pure pandas/numpy indicators (no lookahead)
  score       winsorized cross-sectional z-scores -> weighted composite
  bayes       conjugate Bayesian regression for factor weights (step 1)
  screen      build feature table + rank universe; signal_fn for backtest
  backtest    long-only top-N monthly rebalance, no lookahead
  data        yfinance download (the only networked module)
"""
