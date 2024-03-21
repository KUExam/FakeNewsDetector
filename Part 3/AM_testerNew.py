import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

# Load preprocessed data
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Concatenate all data for TF-IDF vectorization
all_data = pd.concat([train_df, val_df, test_df])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the processed content
tfidf_matrix = tfidf.fit_transform(all_data['processed_content'])

# Split the data back into train, validation, and test sets
X_train = tfidf_matrix[:len(train_df)]
y_train = train_df['category'].values
X_val = tfidf_matrix[len(train_df):len(train_df)+len(val_df)]
y_val = val_df['category'].values
X_test = tfidf_matrix[len(train_df)+len(val_df):]
y_test = test_df['category'].values


class ArticleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ArticleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Define hyperparameters
input_size = tfidf_matrix.shape[1]  # Number of features
hidden_size = 100
num_classes = 2  # Fake or Reliable

# Initialize model
model = ArticleClassifier(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_preds = torch.argmax(val_outputs, axis=1)
        val_acc = (val_preds == y_val_tensor).float().mean().item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Evaluate model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, axis=1)
    test_acc = (test_preds == y_test_tensor).float().mean().item()
    print('Test Accuracy:', test_acc)
