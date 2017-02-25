from data import get_data
import pickle
import pandas as pd
import os

data, sentiments, dictionary, reverse_dict = get_data(remove_stopwords=True)

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
counts = review_df.sum(axis=0)
review_df = review_df / counts
print(counts)
prior_positive = counts[0] / (counts[0] + counts[1])
prior_negative = counts[1] / (counts[0] + counts[1])
for review in data[:10]:
    for word in review:
        prior_negative *= review_df.loc[reverse_dict[word]][0]
        prior_positive *= review_df.loc[reverse_dict[word]][1]
    print(prior_negative)
    print(prior_positive)
    if prior_positive > prior_negative:
        pred_sentiment.append(1)
    else:
        pred_sentiment.append(0)

print(pred_sentiment)
print(sentiments[:10])
correct_pred = pd.Series(pred_sentiment) == sentiments[:10]
accuracy = correct_pred.sum() / len(sentiments)
print(accuracy)
