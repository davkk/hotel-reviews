from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from wordcloud import WordCloud

from hotel_reviews import common

tqdm.pandas()
args = common.parse_args()
common.setup_pyplot()

df = pd.read_csv(args.infile, keep_default_na=False, na_filter=False)
if args.test:
    df = df.head(n=10)

df = df[
    (df["phrase"].str.split().str.len() > 1) & (df["phrase"].str.split().str.len() < 3)
]

counts = (
    df.groupby("rating")[["polarity", "phrase"]]
    .value_counts()
    .reset_index(name="count")
)  # type: ignore
counts = counts.groupby("rating").head(30)
counts = counts[counts["count"] > 1]


def create_clouds(title: str, axs, df: pd.DataFrame):
    for ax, rating in zip(axs, range(1, 11)):
        wc = WordCloud(background_color="white", width=300, height=300)
        freqs = df[df["rating"] == rating].set_index("phrase")["count"].to_dict()
        if len(freqs.items()) > 0:
            wc.generate_from_frequencies(freqs)
            ax.imshow(
                wc.recolor(colormap=plt.get_cmap("berlin")),
                interpolation="bilinear",
            )
        else:
            data = np.ones((200, 200))
            ax.imshow(data, cmap="gray", interpolation="nearest", vmin=0, vmax=1)

        ax.set_title(f"{dict(pos="Positive", neg="Negative")[title]}, {rating}/10")
        ax.axis("off")


fig, axs = plt.subplots(nrows=2, ncols=10, figsize=(16, 4))

for idx, (polarity, df) in enumerate(counts.groupby("polarity")):
    create_clouds(polarity, axs[idx], df)

fig.suptitle("Most frequently used words (positive and negative) vs. rating")

# fig.tight_layout()
fig.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
