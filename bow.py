from data import get_data
import pickle
import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data, sentiments, dictionary, reverse_dict = get_data(remove_stopwords=True)

# Save DataFrame to avoid counting of words multiple times
if not os.path.isfile('data/bow.pickle'):
    # Create dataframe
    review_df = pd.DataFrame(0, columns=[0, 1], index=dictionary.keys())

    # create Bag of words
    for review, sentiment in zip(data, sentiments):
        for word in review:
            review_df.loc[reverse_dict[word]][sentiment] += 1

    with open('data/bow.pickle', 'wb') as f:
        pickle.dump(review_df, f)
else:
    with open('data/bow.pickle', 'rb') as f:
        review_df = pickle.load(f)


# classify using Naive bayes
pred_sentiment = []
# Count number of words in each class
counts = review_df.sum(axis=0)
print(counts)
for review in data:
    # Compute prior of a class
    prior_positive = np.log(counts[0]) - np.log(counts[0] + counts[1])
    prior_negative = np.log(counts[1]) - np.log(counts[0] + counts[1])
    for word in review:
        prior_negative += np.log(review_df.loc[reverse_dict[word]][0] + 1) - np.log(counts[0])
        prior_positive += np.log(review_df.loc[reverse_dict[word]][1] + 1) - np.log(counts[1])
    if prior_positive > prior_negative:
        pred_sentiment.append(1)
    else:
        pred_sentiment.append(0)

# Calculate scores
accuracy = accuracy_score(sentiments, pred_sentiment)
precision = precision_score(sentiments, pred_sentiment)
recall = recall_score(sentiments, pred_sentiment)
fmeasure = f1_score(sentiments, pred_sentiment)
print(accuracy)
print(precision)
print(recall)
print(fmeasure)
