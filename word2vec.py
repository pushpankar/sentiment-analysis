import pandas as pd
import re
from bs4 import BeautifulSoup

train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                    delimiter="\t", quoting=3)


def review_to_list(raw_review):
    """ convert a raw review to a list of words"""
    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()

    return words


reviews = []

for i in range(train["review"].size):
    reviews.append(review_to_list(train["review"][i]))


print(reviews[0])


def build_dictionary(review_list):
    dictionary = dict()
    for review in review_list:
        for word in review:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def build_data(dictionary, review_list):
    data = list()
    for review in review_list:
        review_as_num = []
        for word in review:
            word_index = dictionary[word]
            review_as_num.append(word_index)
        data.append(review_as_num)
    return data


dictionary = build_dictionary(reviews)
data = build_data(dictionary, reviews)
print(data[0])
