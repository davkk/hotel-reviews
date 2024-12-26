from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from hotel_reviews import common

args = common.parse_args()
colors, markers = common.setup_pyplot()

df = pd.read_csv(args.infile)

print(df[["review_pos", "review_neg"]].count().plot(kind="bar"))

labels = ["Positive", "Negative"]
plt.xticks(range(len(labels)), labels, rotation=45)

plt.title("Positive vs. Negative reviews")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
