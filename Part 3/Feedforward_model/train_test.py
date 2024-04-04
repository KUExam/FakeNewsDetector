# train_test.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm  

def train_model(model, criterion, optimizer, X_train, y_train, X_val, y_val, num_epochs=10, patience=3, batch_size=32):
    best_val_loss = float('inf')
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        with tqdm(total=len(X_train), desc=f'Epoch {epoch+1}/{num_epochs}', unit=' samples') as pbar:
            for i in range(0, len(X_train), batch_size):
                inputs = X_train[i:i+batch_size]
                labels = y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                pbar.update(len(inputs))  # Update progress bar
            
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_preds = (val_outputs >= 0.5).float()  # Thresholding for prediction
            val_acc = accuracy_score(y_val, val_preds)
            print(f'Validation Loss: {val_loss.item():.4f}, Validation Accuracy: {val_acc:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                print(f'Early stopping at epoch {epoch+1} due to no improvement in validation loss.')
                break

def test_model(model, criterion, X_test, y_test):
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_preds = (test_outputs >= 0.5).float()
        test_acc = accuracy_score(y_test, test_preds)
        print(f'Test Accuracy: {test_acc:.4f}')
        print(classification_report(y_test, test_preds))
