# # Day 4: Sentiment Analysis and Dashboard
#
# ## Analysis
# Generate fictional social posts and classify their tone with a transparent positive/negative word lexicon. Summarize engagement and frequent terms.
#
# ## Dashboard
# Compare sentiment counts, common terms, and engagement by sentiment. This baseline does not understand sarcasm or context; no real posts are collected.
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
positive = {"love", "great", "excellent", "easy", "helpful", "happy", "fast", "recommend"}
negative = {"hate", "poor", "broken", "slow", "difficult", "refund", "angry", "disappointing"}
templates = {
    "positive": ["Love the great service, it was fast and helpful.", "Excellent experience, happy to recommend.", "Great update: easy to use and very helpful."],
    "neutral": ["The latest update is available today.", "I used the service this week and shared feedback.", "The product arrived and I checked the details."],
    "negative": ["Poor service, the app is slow and broken.", "Disappointing experience; difficult to use.", "I am angry and want a refund for this broken item."],
}
labels = rng.choice(list(templates), 900, p=[0.42, 0.30, 0.28])
posts = pd.DataFrame({"text": [rng.choice(templates[label]) for label in labels], "engagement": rng.integers(1, 900, 900)})
def score(text):
    words = text.lower().replace(",", "").replace(".", "").replace(";", "").split()
    return sum(word in positive for word in words) - sum(word in negative for word in words)
posts["score"] = posts.text.map(score)
posts["sentiment"] = np.select([posts.score > 0, posts.score < 0], ["Positive", "Negative"], default="Neutral")
print(f"Posts: {len(posts):,} | Positive: {(posts.sentiment == 'Positive').mean():.1%} | Negative: {(posts.sentiment == 'Negative').mean():.1%}")
print("Sentiment counts:\n", posts.sentiment.value_counts())
output_path = data_dir / "day_04_posts.csv"
posts.to_csv(output_path, index=False)
print(f"Saved analysis data: {output_path}")
