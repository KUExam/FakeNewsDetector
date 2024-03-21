import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("/Users/rasmuslyngsoe/Documents/GitHub/FakeNewsDetector")

from FakeNewsDetector.Part_1.Task_2 import tfidf_matrix, feature_names

# Read pre-split train, validation, and test sets
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Load pre-computed TF-IDF matrix and feature names
tfidf_matrix = ...  # Load TF-IDF matrix from the first code snippet
feature_names = ...  # Load feature names from the first code snippet

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(tfidf_matrix[train_df.index], dtype=torch.float32)
y_train_tensor = torch.tensor(train_df['type'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)
X_val_tensor = torch.tensor(tfidf_matrix[val_df.index], dtype=torch.float32)
y_val_tensor = torch.tensor(val_df['type'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)
X_test_tensor = torch.tensor(tfidf_matrix[test_df.index], dtype=torch.float32)
y_test_tensor = torch.tensor(test_df['type'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)

# Define neural network model
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

# Initialize model, loss function, and optimizer
input_size = X_train_tensor.shape[1]
hidden_size = 100
num_classes = 2
model = ArticleClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 10
batch_size = 50
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Validate model
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_preds = torch.argmax(val_outputs, axis=1)
        val_acc = accuracy_score(y_val_tensor, val_preds)
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

# Evaluate model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, axis=1)
    test_acc = accuracy_score(y_test_tensor, test_preds)
    print('Test Accuracy:', test_acc)
    print('Classification Report:')
    print(classification_report(y_test_tensor, test_preds))
