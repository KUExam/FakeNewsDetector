# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from train_test import train_model, test_model
from model import ArticleClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test the article classifier model.")
    parser.add_argument("--test_only", action="store_true", help="Test the model without training.")
    parser.add_argument("--train_only", action="store_true", help="Train the model without testing.")
    parser.add_argument("--new_data_file", type=str, default="data/test_data.csv", help="Path to the new dataset.")
    parser.add_argument("--model_file", type=str, default="FFN_model.pth", help="Path to the model file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if args.test_only and args.train_only:
        print("Error: Cannot specify both --test_only and --train_only.")
        exit()

    # Read data
    train_df = pd.read_csv('data/train_data.csv')
    val_df = pd.read_csv('data/val_data.csv')
    test_df = pd.read_csv(args.new_data_file)  # Use specified new data file

    # Fill NaN values with an empty string in the 'processed_content' column
    train_df['processed_content'] = train_df['processed_content'].fillna('')
    val_df['processed_content'] = val_df['processed_content'].fillna('')
    test_df['processed_content'] = test_df['processed_content'].fillna('')

    # Preprocess data
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train = vectorizer.fit_transform(train_df['processed_content'])
    y_train = train_df['category'].map({'fake': 0, 'reliable': 1})
    X_val = vectorizer.transform(val_df['processed_content'])
    y_val = val_df['category'].map({'fake': 0, 'reliable': 1})
    X_test = vectorizer.transform(test_df['processed_content']) 
    y_test = test_df['category'].map({'fake': 0, 'reliable': 1})

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Initialize model, loss function, and optimizer
    input_size = X_train_tensor.shape[1]
    hidden_size = 100
    model = ArticleClassifier(input_size, hidden_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    if args.test_only:
        # Load pre-trained model
        model.load_state_dict(torch.load(args.model_file))
        # Test model without further training
        test_model(model, criterion, X_test_tensor, y_test_tensor)
    elif args.train_only:
        # Train model
        train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
        # Save trained model
        torch.save(model.state_dict(), args.model_file)
        print("Trained model has been saved")
    else:
        # Train model
        train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
        # Save trained model
        torch.save(model.state_dict(), args.model_file)
        print("Trained model has been saved")
        # Test model
        test_model(model, criterion, X_test_tensor, y_test_tensor)
