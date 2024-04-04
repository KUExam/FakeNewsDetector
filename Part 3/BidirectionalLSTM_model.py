import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric, Precision, Recall
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tqdm import tqdm
import tensorflow as tf
import os

# Load datasets
def load_and_preprocess(file_path):
    data = pd.read_csv(file_path).fillna('')
    data['processed_content'] = data['processed_content'].fillna('').astype(str)
    return data

train_data = load_and_preprocess('train_data.csv')
val_data = load_and_preprocess('val_data.csv')
test_data = load_and_preprocess('test_data.csv')
LIAR_data = load_and_preprocess('train_liar_update.csv')


# Tokenization and padding
max_features = 10000  # Size of the vocabulary
maxlen = 250  # Maximum length of the sequences
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['processed_content'])

# Assuming your tokenizer and maxlen are already defined
def tokenize_and_pad(text_series, tokenizer, maxlen=250):
    # This function converts text data into padded sequences
    sequences = tokenizer.texts_to_sequences(text_series)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')
    return padded_sequences

# Convert texts to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(tqdm(train_data['processed_content']))
X_val_sequences = tokenizer.texts_to_sequences(tqdm(val_data['processed_content']))
X_test_sequences = tokenizer.texts_to_sequences(tqdm(test_data['processed_content']))
X_new_sequences = tokenizer.texts_to_sequences(LIAR_data['processed_content'])

# Padding sequences to ensure uniform length
X_train_padded = tokenize_and_pad(train_data['processed_content'], tokenizer, maxlen)
X_val_padded = tokenize_and_pad(val_data['processed_content'], tokenizer, maxlen)
X_test_padded = tokenize_and_pad(test_data['processed_content'], tokenizer, maxlen)
X_new_padded = pad_sequences(X_new_sequences, maxlen=maxlen, padding='post', truncating='post')


# Convert labels to integers and then to binary class matrix
Y_train = train_data['category'].map({'fake': 0, 'reliable': 1}).values
Y_val = val_data['category'].map({'fake': 0, 'reliable': 1}).values

# Convert labels to a format suitable for binary classification
Y_train = to_categorical(train_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_val = to_categorical(val_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)

# Convert the test and LIAR labels to integers and then to binary class matrix
Y_test = to_categorical(test_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_new = to_categorical(LIAR_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_padded, Y_val))

embed_dim = 128
lstm_out = 196

epochs = 1
batch_size = 200

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE).cache()
val_dataset = val_dataset.batch(batch_size).prefetch(AUTOTUNE).cache()

if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)


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

# Define and compile model
model = Sequential()
model.add(Embedding(max_features, embed_dim))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(2, activation='softmax', dtype='float32'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])

# Train the model
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
    
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

# Get model predictions for the test and LIAR set
test_predictions_prob = model.predict(X_test_padded)
test_predictions = test_predictions_prob.argmax(axis=-1)
new_predictions_prob = model.predict(X_new_padded)
new_predictions = new_predictions_prob.argmax(axis=-1)

y_test = Y_test.argmax(axis=-1)
y_new = Y_new.argmax(axis=-1)

# Calculate Precision and Recall
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)

# Calculate Precision, Recall, and F1 Score for the LIAR dataset
LIAR_precision = precision_score(y_new, new_predictions)
LIAR_recall = recall_score(y_new, new_predictions)
LIAR_f1 = f1_score(y_new, new_predictions)

print(f'LIAR Dataset Precision: {LIAR_precision}')
print(f'LIAR Dataset Recall: {LIAR_recall}')
print(f'LIAR Dataset F1 Score: {LIAR_f1}')

# Generate and visualize the confusion matrix for the LIAR dataset
new_cm = confusion_matrix(y_new, new_predictions)

sns.heatmap(new_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - New Dataset')
plt.show()

# Calculate F1 Score
f1 = f1_score(y_test, test_predictions)

print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 Score: {f1}')

# Generate a confusion matrix for the test dataset
test_cm = confusion_matrix(y_test, test_predictions)

# Visualize the confusion matrix for test dataset
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Test Dataset')
plt.show()