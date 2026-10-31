# # Day 3: Stock Trend Analysis and Dashboard
#
# ## Analysis
# Generate a synthetic price series, calculate 20- and 50-day moving averages, and summarize returns and drawdown.
#
# ## Dashboard
# Review price momentum, daily-return distribution, and annual returns. This simulated market series is not a trading signal.
#
# **Dependencies:** NumPy, pandas, Matplotlib.

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
project_dir = Path.cwd()
if project_dir.name == "visualizations":
    project_dir = project_dir.parent
elif (project_dir / "daily_projects").is_dir():
    project_dir = project_dir / "daily_projects"
data_dir = project_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Analysis
stock = pd.DataFrame(index=pd.bdate_range("2022-01-03", periods=750))
stock["daily_return"] = rng.normal(0.00035, 0.012, len(stock))
stock["close"] = 100 * np.exp(stock.daily_return.cumsum())
stock["ma_20"] = stock.close.rolling(20).mean()
stock["ma_50"] = stock.close.rolling(50).mean()
annual_returns = stock.close.resample("YE").last().pct_change().dropna() * 100
volatility = stock.daily_return.std() * np.sqrt(252) * 100
max_drawdown = ((stock.close / stock.close.cummax()) - 1).min() * 100
print(f"Latest close: ${stock.close.iloc[-1]:.2f} | Annualized volatility: {volatility:.1f}% | Max drawdown: {max_drawdown:.1f}%")
print("Annual returns (%):\n", annual_returns.round(2))
output_path = data_dir / "day_03_stock.csv"
stock.rename_axis("date").reset_index().to_csv(output_path, index=False)
print(f"Saved analysis data: {output_path}")
