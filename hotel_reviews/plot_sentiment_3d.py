from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from textblob import TextBlob
from tqdm.auto import tqdm

from hotel_reviews import common

common.setup_pyplot()

args = common.parse_args()
tqdm.pandas()

df = pd.read_csv(args.infile, keep_default_na=False, na_filter=False)
if args.test:
    df = df.head(n=10)


df["review"] = df.review_neg + " " + df.review_pos
df[["polarity", "subjectivity"]] = df.review.apply(
    lambda text: pd.Series(TextBlob(text).sentiment)
)

min_words = 20
df = df[df["review"].str.split().str.len() >= min_words]

print()
print(f"{df.polarity.corr(df.subjectivity)=}")

fig = plt.figure(figsize=(7, 5))
gs = GridSpec(3, 5, figure=fig)
ax_3d = fig.add_subplot(gs[:, 1:], projection="3d")
ax_rat_pol = fig.add_subplot(gs[0, 0])
ax_rat_sub = fig.add_subplot(gs[1, 0])
ax_pol_sub = fig.add_subplot(gs[2, 0])

sc = ax_3d.scatter(
    df.polarity,
    df.subjectivity,
    df.rating,
    c=df.rating,
    cmap=plt.get_cmap("rainbow"),
    norm=Normalize(df.rating.min(), df.rating.max()),
)
fig.colorbar(sc, ax=ax_3d, shrink=0.4, pad=0.1)

ax_3d.set_xlabel("Polarity", labelpad=-3)
ax_3d.tick_params(axis="x", pad=-3)
ax_3d.set_ylabel("Subjectivity", labelpad=-3)
ax_3d.tick_params(axis="y", pad=-3)
ax_3d.set_zlabel("Rating", labelpad=-5)
ax_3d.tick_params(axis="z", pad=-1)

df.plot("polarity", "rating", s=1, kind="scatter", ax=ax_rat_pol)
ax_rat_pol.set_title(f"corr = {df.polarity.corr(df.rating):.5f}")
ax_rat_pol.set_xticks(range(-1, 2))

df.plot("subjectivity", "rating", s=1, kind="scatter", ax=ax_rat_sub)
ax_rat_sub.set_title(f"corr = {df.subjectivity.corr(df.rating):.5f}")

df.plot("polarity", "subjectivity", s=1, kind="scatter", ax=ax_pol_sub)
ax_pol_sub.set_title(f"corr = {df.polarity.corr(df.subjectivity):.5f}")
ax_pol_sub.set_xticks(range(-1, 2))

fig.suptitle(
    f"Rating vs. Polarity vs. Subjectivity (reviews containing at least {min_words} words)"
)

fig.tight_layout()
fig.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
