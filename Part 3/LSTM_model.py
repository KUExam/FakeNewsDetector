## This model is not tested and visualized in the report, 
## as we decided to use BI_LSTM instead

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric, Precision, Recall
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tensorflow as tf

# Load your datasets
train_data = pd.read_csv('train_data.csv').fillna('')
val_data = pd.read_csv('val_data.csv').fillna('')

train_data['processed_content'] = train_data['processed_content'].fillna('')
val_data['processed_content'] = val_data['processed_content'].fillna('')

train_data['processed_content'] = train_data['processed_content'].astype(str)
val_data['processed_content'] = val_data['processed_content'].astype(str)

# Initialize the tokenizer
max_features = 10000  # This is the size of the vocabulary
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['processed_content'])

# Convert texts to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(tqdm(train_data['processed_content']))
X_val_sequences = tokenizer.texts_to_sequences(tqdm(val_data['processed_content']))

# Padding sequences to ensure uniform length
maxlen = 250
X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, truncating='post', padding='post')
X_val_padded = pad_sequences(X_val_sequences, maxlen=maxlen, truncating='post', padding='post')

# Convert labels to integers and then to binary class matrix
Y_train = train_data['category'].map({'fake': 0, 'reliable': 1}).values
Y_val = val_data['category'].map({'fake': 0, 'reliable': 1}).values

# Convert labels to a format suitable for binary classification
Y_train = to_categorical(Y_train, num_classes=2)
Y_val = to_categorical(Y_val, num_classes=2)

embed_dim = 128
lstm_out = 196

if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

model = Sequential()
model.add(Embedding(max_features, embed_dim)) 
model.add(SpatialDropout1D(0.2))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax', dtype='float32'))  # Using 2 because of to_categorical

class F1Score(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state() 
        self.recall.reset_state()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', F1Score()])
epochs = 3
batch_size = 200

history = model.fit(X_train_padded, Y_train, epochs=epochs, batch_size=batch_size, 
                    validation_data=(X_val_padded, Y_val))

# Get model predictions for the validation set
val_predictions_prob = model.predict(X_val_padded)
val_predictions = val_predictions_prob.argmax(axis=-1)

y_val = Y_val.argmax(axis=-1)

cm = confusion_matrix(y_val, val_predictions)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

evaluation = model.evaluate(X_val_padded, Y_val, batch_size=batch_size)
print(f'Validation Loss: {evaluation[0]}')
print(f'Validation Accuracy: {evaluation[1]}')