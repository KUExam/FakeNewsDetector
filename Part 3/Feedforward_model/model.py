# model.py
import torch
import torch.nn as nn

class ArticleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ArticleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.dropout = nn.Dropout(p=0.2)  # Dropout layer with dropout probability 0.2
        self.fc2 = nn.Linear(hidden_size, 1)  # Second fully connected layer for binary classification
    
    def forward(self, x):
        out = self.fc1(x)  # Pass through the first fully connected layer
        out = self.relu(out)  # Apply ReLU activation
        out = self.dropout(out)  # Apply dropout
        out = self.fc2(out)  # Pass through the second fully connected layer
        return torch.sigmoid(out)  # Apply sigmoid activation for binary classification
