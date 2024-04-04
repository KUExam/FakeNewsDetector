import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read our training, validation and Liar datasets.
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
liar_df = pd.read_csv('train_liar_update.csv')  # Load the Liar dataset

# Fill NaN values with an empty string in the 'processed_content' column
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
liar_df['processed_content'] = liar_df['processed_content'].fillna('')  # Do the same for the Liar dataset

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

# Fit on training data and transform the training data
X_train_tfidf = vectorizer.fit_transform(train_df['processed_content'])

# Transform the validation and Liar data
X_val_tfidf = vectorizer.transform(val_df['processed_content'])
X_liar_tfidf = vectorizer.transform(liar_df['processed_content'])  # Transform the Liar data

y_train = train_df['category'].map({'reliable': 1, 'fake': 0}).values
y_val = val_df['category'].map({'reliable': 1, 'fake': 0}).values
y_liar = liar_df['category'].map({'reliable': 1, 'fake': 0}).values  # Do the same for the Liar dataset

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Liar dataset predictions
liar_predictions = model.predict(X_liar_tfidf)  # Make predictions on the Liar dataset
print(f"Liar Dataset Accuracy: {accuracy_score(y_liar, liar_predictions)}")
print(classification_report(y_liar, liar_predictions))

# We make a confusion matrix to give a clearer picture of our model's performance by showing the true positives, true negatives, false positives, and false negatives.
cm = confusion_matrix(y_liar, liar_predictions)  # Make a confusion matrix for the Liar dataset
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['fake', 'reliable'], yticklabels=['fake', 'reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
