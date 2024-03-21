import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

# Load your datasets
train_data = pd.read_csv('train_data.csv')
val_data = pd.read_csv('val_data.csv')
test_data = pd.read_csv('test_data.csv')

# Tokenize and Pad sequences
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_data['processed_content'].values)
X_train = tokenizer.texts_to_sequences(train_data['processed_content'].values)
X_train = pad_sequences(X_train)

X_val = tokenizer.texts_to_sequences(val_data['processed_content'].values)
X_val = pad_sequences(X_val, maxlen=X_train.shape[1])

X_test = tokenizer.texts_to_sequences(test_data['processed_content'].values)
X_test = pad_sequences(X_test, maxlen=X_train.shape[1])

# Build the LSTM model
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X_train.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# Train the model
Y_train = pd.get_dummies(train_data['type']).values
Y_val = pd.get_dummies(val_data['type']).values

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, validation_data=(X_val, Y_val))

# Evaluate the model
Y_test = pd.get_dummies(test_data['type']).values
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

