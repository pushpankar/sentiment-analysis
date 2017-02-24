import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup


def review_to_list(raw_review, remove_stopwords):
    """ convert a raw review to a list of words"""
    # Remove HTML
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()

    if remove_stopwords:
        # Get rid of stop words
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]

    return words


def build_dictionary(review_list):
    dictionary = dict()
    reverse_dictionary = dict()
    for review in review_list:
        for word in review:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    for key, value in dictionary.items():
        reverse_dictionary[value] = key
    return dictionary, reverse_dictionary


def build_data(dictionary, review_list):
    data = list()
    for review in review_list:
        review_as_num = []
        for word in review:
            word_index = dictionary[word]
            review_as_num.append(word_index)
        data.append(review_as_num)
    return data


def get_data(remove_stopwords=False):
    reviews = []
    train = pd.read_csv("data/labeledTrainData.tsv", header=0,
                        delimiter="\t", quoting=3)

    for i in range(train["review"].size):
        reviews.append(review_to_list(train["review"][i], remove_stopwords))

    dictionary, reverse_dictionary = build_dictionary(reviews)
    data = build_data(dictionary, reviews)
    return data, train["sentiment"], dictionary, reverse_dictionary
