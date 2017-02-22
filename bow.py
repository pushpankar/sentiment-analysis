import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)

# Verify data
print(train.shape)
print(train.columns.values)
print(train["review"][0])


def review_to_words(raw_review):
    """ Function to convert a raw review to a string of words"""
    # Remove html
    review_text = BeautifulSoup(raw_review).get_text()

    # Remove non-letter
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()

    # Get rid of stop words
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if w not in stops]

    return (" ".join(meaningful_words))


clean_review = review_to_words(train["review"][0])
print(clean_review)

num_reviews = train["review"].size
clean_train_reviews = []

for i in xrange(0, num_reviews):
    clean_train_reviews.append(review_to_words(train["review"][i]))
