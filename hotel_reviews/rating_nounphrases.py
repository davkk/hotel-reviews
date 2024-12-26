import pandas as pd
from textblob import TextBlob
from tqdm.auto import tqdm

from hotel_reviews import common

tqdm.pandas()
args = common.parse_args()

df = pd.read_csv(args.infile, keep_default_na=False, na_filter=False)
if args.test:
    df = df.head(n=10)


polarities = []
ratings = []
phrases = []

for _, (text_pos, text_neg, rating) in tqdm(
    df[["review_pos", "review_neg", "rating"]].iterrows(),  # type: ignore
    total=len(df),
):
    blob_pos = TextBlob(text_pos)
    blob_neg = TextBlob(text_neg)

    for phrase in blob_neg.noun_phrases:  # type: ignore
        polarities.append("neg")
        ratings.append(int(rating))
        phrases.append(phrase)

    for phrase in blob_pos.noun_phrases:  # type: ignore
        polarities.append("pos")
        ratings.append(int(rating))
        phrases.append(phrase)

pd.DataFrame(
    dict(
        polarity=polarities,
        phrase=phrases,
        rating=ratings,
    )
).to_csv(args.outfile, index=False)
