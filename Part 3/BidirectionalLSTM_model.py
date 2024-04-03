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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Force TensorFlow to use GPU 0

if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # If GPUs are found, try to set the first one as the only visible device
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load your datasets
train_data = pd.read_csv('train_data.csv').fillna('')
val_data = pd.read_csv('val_data.csv').fillna('')
test_data = pd.read_csv('test_data.csv').fillna('')

train_data['processed_content'] = train_data['processed_content'].fillna('')
val_data['processed_content'] = val_data['processed_content'].fillna('')
test_data['processed_content'] = test_data['processed_content'].fillna('')

train_data['processed_content'] = train_data['processed_content'].astype(str)
val_data['processed_content'] = val_data['processed_content'].astype(str)
test_data['processed_content'] = test_data['processed_content'].astype(str)


# Initialize the tokenizer
max_features = 10000  # This is the size of the vocabulary
tokenizer = Tokenizer(num_words=max_features, oov_token="<OOV>")
tokenizer.fit_on_texts(train_data['processed_content'])

# Assuming your tokenizer and maxlen are already defined
def tokenize_and_pad(text_series, tokenizer, maxlen=250):
    # This function converts text data into padded sequences
    sequences = tokenizer.texts_to_sequences(text_series)
    padded_sequences = pad_sequences(sequences, maxlen=maxlen, truncating='post', padding='post')
    return padded_sequences

maxlen = 250  

# Convert texts to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(tqdm(train_data['processed_content']))
X_val_sequences = tokenizer.texts_to_sequences(tqdm(val_data['processed_content']))

X_test_sequences = tokenizer.texts_to_sequences(tqdm(test_data['processed_content']))
X_test_padded = tokenize_and_pad(test_data['processed_content'], tokenizer, maxlen)

# Padding sequences to ensure uniform length
X_train_padded = tokenize_and_pad(train_data['processed_content'], tokenizer, maxlen)
X_val_padded = tokenize_and_pad(val_data['processed_content'], tokenizer, maxlen)


# Convert labels to integers and then to binary class matrix
Y_train = train_data['category'].map({'fake': 0, 'reliable': 1}).values
Y_val = val_data['category'].map({'fake': 0, 'reliable': 1}).values

# Convert labels to a format suitable for binary classification
Y_train = to_categorical(train_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)
Y_val = to_categorical(val_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)

# Convert the test labels to integers and then to binary class matrix
Y_test = to_categorical(test_data['category'].map({'fake': 0, 'reliable': 1}).values, num_classes=2)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_padded, Y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_padded, Y_val))

embed_dim = 128
lstm_out = 196

epochs = 4
batch_size = 200

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE).cache()
val_dataset = val_dataset.batch(batch_size).prefetch(AUTOTUNE).cache()

if tf.config.experimental.list_physical_devices('GPU'):
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

model = Sequential()
model.add(Embedding(max_features, embed_dim))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2)))  # Wrapped in Bidirectional
model.add(Dense(2, activation='softmax', dtype='float32'))

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

history = model.fit(train_dataset,
                    epochs=epochs,
                    validation_data=val_dataset)

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

# Evaluate the model on the test set
test_evaluation = model.evaluate(X_test_padded, Y_test, batch_size=batch_size)
print(f'Test Loss: {test_evaluation[0]}')
print(f'Test Accuracy: {test_evaluation[1]}')

# Get model predictions for the test set
test_predictions_prob = model.predict(X_test_padded)
test_predictions = test_predictions_prob.argmax(axis=-1)

y_test = Y_test.argmax(axis=-1)

# Generate a confusion matrix for the test dataset
test_cm = confusion_matrix(y_test, test_predictions)

# Visualize the confusion matrix
sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Test Dataset')
plt.show()