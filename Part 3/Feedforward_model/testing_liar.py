import argparse
import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from feedforward_model import ArticleClassifier

def test_on_new_dataset(new_data_file, model_file):
    # Read the new dataset
    new_df = pd.read_csv(new_data_file)

    # Fill NaN values with an empty string in the 'processed_content' column
    new_df['processed_content'] = new_df['processed_content'].fillna('')

    # Load the saved vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    vectorizer.fit(new_df['processed_content'])  # Assuming you have access to training data

    # Preprocess the new data using the fitted vectorizer
    X_new = vectorizer.transform(new_df['processed_content'])

    # Convert data to PyTorch tensor
    X_new_tensor = torch.tensor(X_new.toarray(), dtype=torch.float32)

    # Load the saved model
    input_size = X_new.shape[1]  # Input size should match the shape of the new data
    hidden_size = 100
    model = ArticleClassifier(input_size, hidden_size)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # Perform inference on the new dataset
    with torch.no_grad():
        new_outputs = model(X_new_tensor)
        new_preds = (new_outputs >= 0.5).float()  # Thresholding for prediction

    # True labels
    new_true_labels = new_df['category'].map({'fake': 0, 'reliable': 1})

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(new_true_labels, new_preds)
    classification_rep = classification_report(new_true_labels, new_preds)

    print(f'Accuracy on new dataset: {accuracy:.4f}')
    print('Classification Report on new dataset:')
    print(classification_rep)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test on new dataset.')
    parser.add_argument('--test_only', action='store_true', help='Run in test-only mode.')
    parser.add_argument('--new_data_file', type=str, default='data/train_liar_update.csv', help='Path to the new dataset.')
    parser.add_argument('--model_file', type=str, default='FFN_model.pth', help='Path to the model file.')
    args = parser.parse_args()

    if args.test_only:
        test_on_new_dataset(args.new_data_file, args.model_file)
    else:
        print("Training is not performed as --test_only flag is provided.")
