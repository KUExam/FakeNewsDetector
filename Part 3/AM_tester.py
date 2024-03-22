import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from tqdm import tqdm  # Import tqdm for progress bar

# Read data (assuming you have separate CSV files for train, validation, and test sets)
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')


# Fill NaN values with an empty string in the 'processed_content' column
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
test_df['processed_content'] = test_df['processed_content'].fillna('')

# Preprocess data
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = vectorizer.fit_transform(train_df['processed_content'])
y_train = train_df['type'].map({'fake': 0, 'reliable': 1})

X_val = vectorizer.transform(val_df['processed_content'])
y_val = val_df['type'].map({'fake': 0, 'reliable': 1})

X_test = vectorizer.transform(test_df['processed_content']) 
y_test = test_df['type'].map({'fake': 0, 'reliable': 1})

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

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
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    with tqdm(total=len(X_train_tensor), desc=f'Epoch {epoch+1}/{num_epochs}', unit=' samples') as pbar:
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i+batch_size]
            labels = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.update(len(inputs))  # Update progress bar

    # Validate model
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_preds = torch.argmax(val_outputs, axis=1)
        val_acc = accuracy_score(y_val_tensor, val_preds)
        print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

# Evaluate model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, axis=1)
    test_acc = accuracy_score(y_test_tensor, test_preds)
    print('Test Accuracy:', test_acc)
    print('Classification Report:')
    print(classification_report(y_test_tensor, test_preds))
