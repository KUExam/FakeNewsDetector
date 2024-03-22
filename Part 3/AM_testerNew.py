import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

# Load preprocessed data
train_df = pd.read_csv('Assigment2_clean.csv')
val_df = pd.read_csv('Assigment2_clean.csv')
test_df = pd.read_csv('Assigment2_clean.csv')

# Fill NaN values with an empty string in the 'processed_content' column
train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')
test_df['processed_content'] = test_df['processed_content'].fillna('')

# Concatenate all data for TF-IDF vectorization
all_data = pd.concat([train_df, val_df, test_df])

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=1000)

# Fit and transform the processed content
tfidf_matrix = tfidf.fit_transform(all_data['processed_content'])
feature_names = tfidf.get_feature_names_out()

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(tfidf_matrix[:len(train_df)], dtype=torch.float32)
y_train_tensor = torch.tensor(train_df['category'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)
X_val_tensor = torch.tensor(tfidf_matrix[len(train_df):len(train_df) + len(val_df)], dtype=torch.float32)
y_val_tensor = torch.tensor(val_df['category'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)
X_test_tensor = torch.tensor(tfidf_matrix[len(train_df) + len(val_df):], dtype=torch.float32)
y_test_tensor = torch.tensor(test_df['category'].map({'fake': 0, 'reliable': 1}).values, dtype=torch.long)

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
        val_acc = (val_preds == y_val_tensor).float().mean().item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

# Evaluate model on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_preds = torch.argmax(test_outputs, axis=1)
    test_acc = (test_preds == y_test_tensor).float().mean().item()
    print('Test Accuracy:', test_acc)
