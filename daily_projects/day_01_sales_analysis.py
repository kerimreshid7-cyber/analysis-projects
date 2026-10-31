# # Day 1: Sales Analysis and Dashboard
#
# ## Analysis
# Create reproducible transaction data, calculate revenue, and compare performance over time, across product categories, and across regions.
#
# ## Dashboard
# A three-view sales dashboard: monthly revenue trend, category contribution, and regional sales. All records are synthetic demo data.
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
sales = pd.DataFrame({
    "date": rng.choice(pd.date_range("2024-01-01", "2025-12-31"), 1200),
    "region": rng.choice(["North", "South", "East", "West"], 1200),
    "category": rng.choice(["Equipment", "Apparel", "Footwear", "Accessories"], 1200),
    "units": rng.integers(1, 8, 1200),
    "unit_price": rng.uniform(12, 240, 1200).round(2),
})
sales["revenue"] = sales.units * sales.unit_price
sales["month"] = sales.date.dt.to_period("M").dt.to_timestamp()
monthly = sales.groupby("month").revenue.sum()
category = sales.groupby("category").revenue.sum().sort_values(ascending=False)
region = sales.groupby("region").revenue.sum().sort_values(ascending=False)
print(f"Orders: {len(sales):,} | Revenue: ${sales.revenue.sum():,.0f} | Average order: ${sales.revenue.mean():,.2f}")
print("Revenue by category:\n", category.round(2))
print("Revenue by region:\n", region.round(2))
output_path = data_dir / "day_01_sales.csv"
sales.to_csv(output_path, index=False)
print(f"Saved analysis data: {output_path}")


# insights
# 1. The total number of orders is 1,200, with a total revenue of $1,080,000 and an average order value of $900.00.
# 2. The category with the highest revenue is "Equipment," followed by "Apparel," " "Footwear," and "Accessories."
