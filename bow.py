from data import get_data
import pickle
import pandas as pd
import os

data, sentiments, dictionary, reverse_dict = get_data(remove_stopwords=True)

if not os.path.isfile('data/bow.pickle'):
    # Create dataframe
    review_df = pd.DataFrame(0, columns=[0, 1], index=dictionary.keys())
    print(review_df.head())

    # create Bag of words
    for review, sentiment in zip(data, sentiments):
        for word in review:
            review_df.loc[reverse_dict[word]][sentiment] += 1

    review_df = review_df.divide(review_df.sum(axis=1), axis=0)

    with open('data/bow.pickle', 'wb') as f:
        pickle.dump(review_df, f)
else:
    with open('data/bow.pickle', 'rb') as f:
        review_df = pickle.load(f)

print(review_df.head(35))

# classify using Naive bayes
pred_sentiment = []
for review in data:
    p_positive = 1
    p_negative = 1
    for word in review:
        p_negative *= review_df.loc[reverse_dict[word]][0]
        p_positive *= review_df.loc[reverse_dict[word]][1]
    if p_positive > p_negative:
        pred_sentiment.append(1)
    else:
        pred_sentiment.append(0)

correct_pred = pd.Series(pred_sentiment) == sentiments
accuracy = correct_pred.sum() / len(sentiments)
print(accuracy)
