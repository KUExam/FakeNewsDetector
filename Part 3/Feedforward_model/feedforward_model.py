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
train_df = pd.read_csv('data/train_data.csv')
val_df = pd.read_csv('data/val_data.csv')
test_df = pd.read_csv('data/test_data.csv')

# Fill NaN values with an empty string in the 'processed_content' column
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
test_df['processed_content'] = test_df['processed_content'].fillna('')

# Preprocess data
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_train = vectorizer.fit_transform(train_df['processed_content'])
y_train = train_df['category'].map({'fake': 0, 'reliable': 1})

X_val = vectorizer.transform(val_df['processed_content'])
y_val = val_df['category'].map({'fake': 0, 'reliable': 1})

X_test = vectorizer.transform(test_df['processed_content']) 
y_test = test_df['category'].map({'fake': 0, 'reliable': 1})

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)  # Reshape to [batch_size, 1]
X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)  # Reshape to [batch_size, 1]
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)  # Reshape to [batch_size, 1]

# Define neural network model with dropout
class ArticleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ArticleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with probability 0.2
        self.fc2 = nn.Linear(hidden_size, 1)  # Output layer with one neuron
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  # Apply dropout
        out = self.fc2(out)
        return torch.sigmoid(out)  # Sigmoid activation for binary classification




# Initialize model, loss function, and optimizer
input_size = X_train_tensor.shape[1]
hidden_size = 100
model = ArticleClassifier(input_size, hidden_size)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Initialize variables for early stopping
best_val_loss = float('inf')  # Set initial best validation loss to infinity
patience = 3  # Number of epochs to wait for improvement
counter = 0  # Counter to track epochs without improvement

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
        val_preds = (val_outputs >= 0.5).float()  # Thresholding for prediction
        val_acc = accuracy_score(y_val_tensor, val_preds)
        print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0  # Reset counter if there's improvement
        else:
            counter += 1  # Increment counter if there's no improvement

        # Check early stopping condition
        if counter >= patience:
            print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss.')
            break  # Stop training loop

# Save the model
torch.save(model.state_dict(), 'FFN_model.pth')

# Test model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    test_preds = (test_outputs >= 0.5).float()  # Thresholding for prediction
    test_acc = accuracy_score(y_test_tensor, test_preds)
    print(f'Test Loss: {test_loss.item():.4f}, Test Accuracy: {test_acc:.4f}')
    print(classification_report(y_test_tensor, test_preds))