from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from textblob import TextBlob
from tqdm.auto import tqdm

from hotel_reviews import common

tqdm.pandas()
args = common.parse_args()
common.setup_pyplot()

df = pd.read_csv(args.infile, keep_default_na=False, na_filter=False)
if args.test:
    df = df.head(n=10)

neg = df[df["review_neg"].str.split().str.len() >= 20]
pos = df[df["review_pos"].str.split().str.len() >= 20]


def get_polarity(text: str):
    blob = TextBlob(text)
    return blob.sentiment.polarity  # type: ignore


plt.plot(neg["review_neg"].apply(get_polarity), neg["rating"], ".", label="Negative", alpha=0.7)
plt.plot(pos["review_pos"].apply(get_polarity), pos["rating"], ".", label="Positive", alpha=0.7)

plt.title("Polarity vs. Rating (reviews with 20 or more words)")
plt.xlabel("Polarity")
plt.ylabel("Rating")

plt.xticks(np.arange(-1, 1.5, 0.5))

plt.legend(frameon=True)

plt.tight_layout()
plt.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
