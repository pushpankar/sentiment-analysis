from __future__ import print_function
import numpy as np

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

np.random.seed(108)
max_features = 30000
maxlen = 512
batch_size = 32

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
print("train size-{}".format(len(X_train)))
print("test size-{}".format(len(X_test)))

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

print('Build model')
model = Sequential()
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout_W=0.2, dropout_U=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy',
              metrics=['precision', 'recall', 'accuracy', 'fmeasure'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score: {}'.format(score))
