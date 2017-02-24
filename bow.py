from data import get_data
import pickle
import pandas as pd
import os

if not os.path.isfile('data/bow.pickle'):
    data, sentiments, dictionary, reverse_dict = get_data(remove_stopwords=True)

    # Create dataframe
    review_df = pd.DataFrame(0, columns=[0, 1], index=dictionary.keys())
    print(review_df.head())

    # create Bag of words
    for review, sentiment in zip(data, sentiments):
        for word in review:
            review_df.loc[reverse_dict[word]][sentiment] += 1

    with open('data/bow.pickle', 'wb') as f:
        pickle.dump(review_df, f)
else:
    with open('data/bow.pickle', 'rb') as f:
        review_df = pickle.load(f)

print(review_df.head(35))
print(review_df[1].argmax())
