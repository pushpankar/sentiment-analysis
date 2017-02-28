import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


np.random.seed(8)
top_words = 5000  # Vocubulary size
# Read Data
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
max_words = 512  # Max words in a review
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)

# define model
model = Sequential()
model.add(Embedding(top_words, 32, input_length=max_words))
model.add(Convolution1D(nb_filter=128, filter_length=3, border_mode='same',
                        activation='relu'))
model.add(MaxPooling1D(pool_length=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['precision', 'recall', 'accuracy', 'fmeasure'])
print(model.summary)

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=128, verbose=1)
scores = model.evaluate(X_test, y_test)
for score in scores[1:]:
    print(score)
