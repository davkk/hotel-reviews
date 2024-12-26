import pandas as pd
import spacy
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

from hotel_reviews import common

nlp = spacy.load("en_core_web_sm")
tqdm.pandas()

args = common.parse_args()

# Index(['index', 'review_title', 'reviewed_at', 'reviewed_by', 'images',
#        'crawled_at', 'url', 'hotel_name', 'hotel_url', 'avg_rating',
#        'nationality', 'rating', 'review_text', 'raw_review_text', 'tags',
#        'meta'],
raw_data = pd.read_csv(args.infile)
if args.test:
    raw_data = raw_data.head(n=10)

columns = ["hotel_name", "nationality", "rating"]
final = raw_data[columns].copy()


def preprocess_text(text: str):
    if type(text) is not str:
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.is_alpha
    ]
    return " ".join(tokens)


def parse_review(html: str):
    if type(html) is not str:
        return pd.Series(["", ""])
    soup = BeautifulSoup(html, "html.parser")
    neg = soup.find("p", class_="review_neg")
    neg = neg.text if neg is not None else ""
    pos = soup.find("p", class_="review_pos")
    pos = pos.text if pos is not None else ""
    return pd.Series([preprocess_text(neg), preprocess_text(pos)])


final["review_title"] = raw_data["review_title"].progress_apply(preprocess_text)
final[
    [
        "review_neg",
        "review_pos",
    ]
] = raw_data["raw_review_text"].progress_apply(parse_review)
final = final.dropna(how="any").reset_index(drop=True)

final.to_csv(args.outfile)
