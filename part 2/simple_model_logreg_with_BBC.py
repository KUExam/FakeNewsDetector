import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Read the original and additional training datasets
train_df1 = pd.read_csv('train_data.csv')
train_df2 = pd.read_csv('assigment2_clean.csv')

# Concatenate the two training DataFrames
train_df = pd.concat([train_df1, train_df2], ignore_index=True)

# Read validation and test datasets
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Fill NaN values with an empty string
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
test_df['processed_content'] = test_df['processed_content'].fillna('')

# Initialize the vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit on training data and transform it
X_train_tfidf = vectorizer.fit_transform(train_df['processed_content'])

# Transform the validation and test data
X_val_tfidf = vectorizer.transform(val_df['processed_content'])
X_test_tfidf = vectorizer.transform(test_df['processed_content'])

# Labels
y_train = train_df['category'].map({'reliable': 1, 'fake': 0}).values
y_val = val_df['category'].map({'reliable': 1, 'fake': 0}).values
y_test = test_df['category'].map({'reliable': 1, 'fake': 0}).values

# Train the model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Validation set predictions
val_predictions = model.predict(X_val_tfidf)
print(f"Validation Accuracy: {accuracy_score(y_val, val_predictions)}")
print(classification_report(y_val, val_predictions))

# Confusion matrix
cm = confusion_matrix(y_val, val_predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
