import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read our training, validation and test datasets.
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('train_liar_update.csv')

# Fill NaN values with an empty string in the 'processed_content' column
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
test_df['processed_content'] = test_df['processed_content'].fillna('')

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit on training data and transform the training data
X_train_tfidf = vectorizer.fit_transform(train_df['processed_content'])

# Transform the validation and test data
X_val_tfidf = vectorizer.transform(val_df['processed_content'])
X_test_tfidf = vectorizer.transform(test_df['processed_content'])

y_train = train_df['category'].map({'reliable': 1, 'fake': 0}).values
y_val = val_df['category'].map({'reliable': 1, 'fake': 0}).values
y_test = test_df['category'].map({'reliable': 1, 'fake': 0}).values

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Test set predictions
test_predictions = model.predict(X_test_tfidf)
print(f"Test Accuracy: {accuracy_score(y_test, test_predictions)}")
print(classification_report(y_test, test_predictions))

# We make a confusion matrix to give a clearer picture of our model's performance by showing the true positives, true negatives, false positives, and false negatives.
cm = confusion_matrix(y_test, test_predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['fake', 'reliable'], yticklabels=['fake', 'reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()