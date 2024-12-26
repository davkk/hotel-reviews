from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from hotel_reviews import common

tqdm.pandas()
args = common.parse_args()
colors, markers = common.setup_pyplot()

df = pd.read_csv(args.infile, keep_default_na=False, na_filter=False)
if args.test:
    df = df.head(n=10)

df["rating"] = df.rating.astype(int)
df["length_pos"] = df["review_pos"].str.split().str.len()
df["length_neg"] = df["review_neg"].str.split().str.len()

pos = df[["length_pos", "rating"]].groupby("rating").mean()
neg = df[["length_neg", "rating"]].groupby("rating").mean()

ax = pos.plot(kind="bar", color=colors[0], alpha=0.7, label="Positive")
neg.plot(kind="bar", ax=ax, color=colors[1], alpha=0.7, label="Negative")

ax.set_title("Average number of words vs. Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Average number of words")

ax.legend(["Positive", "Negative"])

plt.tight_layout()

plt.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
