# # Day 2: Customer Segmentation and Dashboard
#
# ## Analysis
# Aggregate customer spend, purchase frequency, and average order value; standardize the features and explore four KMeans segments.
#
# ## Dashboard
# Compare customer profiles, segment sizes, and average spend. Cluster labels are arbitrary and are not customer truth; all records are synthetic.
#
# **Dependencies:** NumPy, pandas, Matplotlib, scikit-learn.

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
project_dir = Path.cwd()
if project_dir.name == "visualizations":
    project_dir = project_dir.parent
elif (project_dir / "daily_projects").is_dir():
    project_dir = project_dir / "daily_projects"
data_dir = project_dir / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Analysis
orders = pd.DataFrame({"customer_id": rng.integers(1, 401, 2400), "order_value": rng.gamma(3, 32, 2400)})
customers = orders.groupby("customer_id").agg(spend=("order_value", "sum"), orders=("order_value", "size"))
customers["average_order"] = customers.spend / customers.orders
features = customers[["spend", "orders", "average_order"]]
scaled = StandardScaler().fit_transform(features)
customers["segment"] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(scaled)
profile = customers.groupby("segment").agg(customers=("spend", "size"), average_spend=("spend", "mean"), average_orders=("orders", "mean"))
print(f"Customers: {len(customers):,} | Segments: 4 | Largest segment: {profile.customers.max():,}")
print("Segment profile:\n", profile.round(1))
output_path = data_dir / "day_02_customers.csv"
customers.rename_axis("customer_id").reset_index().to_csv(output_path, index=False)
print(f"Saved analysis data: {output_path}")
