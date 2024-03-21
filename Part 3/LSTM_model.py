import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm  # import the tqdm library

# Load your datasets
train_data = pd.read_csv('train_data_combined.csv')
val_data = pd.read_csv('val_data.csv')

train_data['processed_content'] = train_data['processed_content'].astype(str)
val_data['processed_content'] = val_data['processed_content'].astype(str)

# Tokenize and Pad sequences
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(train_data['processed_content'].values)
X_train = tokenizer.texts_to_sequences(tqdm(train_data['processed_content'].values))  # add progress bar
X_train = pad_sequences(X_train)

X_val = tokenizer.texts_to_sequences(tqdm(val_data['processed_content'].values))  # add progress bar
X_val = pad_sequences(X_val, maxlen=X_train.shape[1])

# Build the LSTM model
embed_dim = 128
lstm_out = 196
input_length = 1000  # Set a fixed input_length for the Embedding layer

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = input_length))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

# Train the model
Y_train = pd.get_dummies(train_data['category']).values
Y_val = pd.get_dummies(val_data['category']).values

batch_size = 32
model.fit(X_train, Y_train, epochs = 7, batch_size=batch_size, validation_data=(X_val, Y_val))

# Evaluate the model
score,acc = model.evaluate(X_val, Y_val, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

