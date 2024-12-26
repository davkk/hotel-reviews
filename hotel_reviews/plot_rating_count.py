from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from hotel_reviews import common

args = common.parse_args()
colors, markers = common.setup_pyplot()

df = pd.read_csv(args.infile)

df["rating"].hist(
    bins=range(1, 11),
    orientation="horizontal",
)

plt.title("Review Rating (1-10)")
plt.xlabel("Count")
plt.ylabel("Rating")

plt.yticks(range(1, 11))

plt.tight_layout()
plt.savefig(f"figures/{Path(__file__).stem}.pdf")
# plt.show()
