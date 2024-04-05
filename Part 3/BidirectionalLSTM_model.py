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

def load_and_preprocess(file_path):
    """
    Loads a CSV file and preprocesses its content for NLP tasks.

    Parameters:
    - file_path (str): The path to the CSV file to be loaded.

    Returns:
    - pandas.DataFrame: A DataFrame with non-null, string-typed 'processed_content'.
    """
    data = pd.read_csv(file_path).fillna('')
    data['processed_content'] = data['processed_content'].fillna('').astype(str)
    return data

# Load and preprocess datasets
train_data = load_and_preprocess('train_data.csv')
val_data = load_and_preprocess('val_data.csv')
test_data = load_and_preprocess('test_data.csv')
LIAR_data = load_and_preprocess('train_liar_update.csv')

# Initialize the tokenizer for text processing
max_features = 10000  # Size of the vocabulary
maxlen = 250  # Maximum length of the sequences
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['processed_content'])

def tokenize_and_pad(text_series, tokenizer, maxlen=250):
    """
    Tokenizes and pads a series of texts to a uniform length.

    Parameters:
    - text_series (pandas.Series): The series of texts to be tokenized and padded.
    - tokenizer (Tokenizer): The Keras Tokenizer instance.
    - maxlen (int, optional): The maximum length of the sequences. Defaults to 250.

    Returns:
    - numpy.ndarray: An array of tokenized and padded sequences.
    """
    sequences = tokenizer.texts_to_sequences(text_series)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')
    return padded_sequences

# Convert texts to sequences of integers and then pad them
X_train_padded = tokenize_and_pad(train_data['processed_content'], tokenizer, maxlen)
X_val_padded = tokenize_and_pad(val_data['processed_content'], tokenizer, maxlen)
X_test_padded = tokenize_and_pad(test_data['processed_content'], tokenizer, maxlen)
X_new_padded = tokenize_and_pad(LIAR_data['processed_content'], tokenizer, maxlen)

# Convert labels to a format suitable for binary classification (one-hot encoding)
Y_train = to_categorical(train_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_val = to_categorical(val_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_test = to_categorical(test_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_new = to_categorical(LIAR_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)

# Defining batch size and epoch amounts.
batch_size = 200
epochs = 4

# Preparing TensorFlow datasets for efficient loading
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_padded, Y_val))
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

# Mixed precision policy for performance optimization on GPUs
if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

# Custom F1Score metric class definition
class F1Score(Metric):
    """
    Custom F1 Score metric for Keras models, combining precision and recall.
    """
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

# Setting model hyperparameters:
# embed_dim: Defines the size of the word embedding vectors. Each word is converted into a 128-dimensional vector. 
# lstm_out: Specifies the number of units in the LSTM layer
embed_dim = 128
lstm_out = 196

# Model architecture
model = Sequential([
    Embedding(max_features, embed_dim),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)),
    Dense(2, activation='softmax', dtype='float32')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', F1Score()])

# Model training
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
 
# Obtaining predictions for the validation dataset.
val_predictions_prob = model.predict(X_val_padded)
val_predictions = val_predictions_prob.argmax(axis=-1)

# Convert one-hot encoded validation labels back to their original integer form.
y_val = Y_val.argmax(axis=-1)

# Generating a confusion matrix to compare actual and predicted labels.
cm = confusion_matrix(y_val, val_predictions)

# Visualizing the confusion matrix using a heatmap.
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Evaluating the model on the validation dataset to get loss and accuracy.
evaluation = model.evaluate(X_val_padded, Y_val, batch_size=batch_size)
print(f'Validation Loss: {evaluation[0]}')
print(f'Validation Accuracy: {evaluation[1]}')

# Predicting outcomes for the test and LIAR datasets.
test_predictions_prob = model.predict(X_test_padded)
test_predictions = test_predictions_prob.argmax(axis=-1)
new_predictions_prob = model.predict(X_new_padded)
new_predictions = new_predictions_prob.argmax(axis=-1)

# Converting one-hot encoded test and LIAR dataset labels back to integer form.
y_test = Y_test.argmax(axis=-1)
y_new = Y_new.argmax(axis=-1)

# Calculating precision and recall for the test dataset.
precision = precision_score(y_test, test_predictions)
recall = recall_score(y_test, test_predictions)

# Calculating precision, recall, and F1 score for the LIAR dataset.
LIAR_precision = precision_score(y_new, new_predictions)
LIAR_recall = recall_score(y_new, new_predictions)
LIAR_f1 = f1_score(y_new, new_predictions)

print(f'LIAR Dataset Precision: {LIAR_precision}')
print(f'LIAR Dataset Recall: {LIAR_recall}')
print(f'LIAR Dataset F1 Score: {LIAR_f1}')

# Generating and visualizing the confusion matrix for the LIAR dataset.
new_cm = confusion_matrix(y_new, new_predictions)
sns.heatmap(new_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - New Dataset')
plt.show()

# Calculating and printing F1 score for the test dataset.
f1 = f1_score(y_test, test_predictions)
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 Score: {f1}')

# Generating and visualizing the confusion matrix for the test dataset.
test_cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Test Dataset')
plt.show()
