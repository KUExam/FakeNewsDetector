from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from tqdm.auto import tqdm
import os
from sklearn.metrics import classification_report, accuracy_score
from torch.cuda.amp import GradScaler, autocast

tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')

# Load our csv files
train_df = pd.read_csv('train_data.csv').fillna('')
val_df = pd.read_csv('val_data.csv').fillna('')

# Function to encode the data
def encode_data(df, tokenizer, batch_size=32, max_length=512, save_path='tokenized'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Initialize progress bar
    progress_bar = tqdm(total=len(df), desc="Tokenizing")

    # Prepare batches for tokenization to show progress
    batches = [df[i:i+batch_size] for i in range(0, df.shape[0], batch_size)]
    
    input_ids = []
    attention_masks = []
    labels = []
    
    for batch in batches:
        # Update progress bar
        progress_bar.update(len(batch))
        
        # Tokenize the batch
        inputs = tokenizer(list(batch['processed_content'].values), 
                           max_length=max_length, 
                           padding=True, 
                           truncation=True, 
                           return_tensors="pt")
        
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])
        labels.append(torch.tensor(batch['category'].map({'fake': 0, 'reliable': 1}).values))
    
    # Concatenate all batches
    input_ids = torch.cat(input_ids)
    attention_masks = torch.cat(attention_masks)
    labels = torch.cat(labels)
    
    # Close the progress bar
    progress_bar.close()
    
    # Create a DataLoader
    dataset = TensorDataset(input_ids, attention_masks, labels)

    # Save the tokenized data
    torch.save(input_ids, os.path.join(save_path, 'input_ids.pt'))
    torch.save(attention_masks, os.path.join(save_path, 'attention_masks.pt'))
    torch.save(labels, os.path.join(save_path, 'labels.pt'))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def load_tokenized_data(save_path):
    # Check if the tokenized data files exist
    input_ids_file = os.path.join(save_path, 'input_ids.pt')
    attention_masks_file = os.path.join(save_path, 'attention_masks.pt')
    labels_file = os.path.join(save_path, 'labels.pt')

    if os.path.exists(input_ids_file) and os.path.exists(attention_masks_file) and os.path.exists(labels_file):
        # Load the tokenized data
        input_ids = torch.load(input_ids_file)
        attention_masks = torch.load(attention_masks_file)
        labels = torch.load(labels_file)
        
        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(input_ids, attention_masks, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)  # Adjust batch_size if needed
        
        return dataloader
    else:
        raise FileNotFoundError("Tokenized data files not found in the specified path.")

train_df['processed_content'] = train_df['processed_content'].fillna('')
val_df['processed_content'] = val_df['processed_content'].fillna('')

train_df.dropna(subset=['processed_content'], inplace=True)
val_df.dropna(subset=['processed_content'], inplace=True)

assert train_df['processed_content'].isnull().sum() == 0, "train_df still contains NaN values"
assert val_df['processed_content'].isnull().sum() == 0, "val_df still contains NaN values"

train_dataloader = load_tokenized_data('tokenized_data/train')
validation_dataloader = load_tokenized_data('tokenized_data/validation')
# Load DistilBertForSequenceClassification
model = MobileBertForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=2)
model.cuda()

optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8) # Adjust learning rate if needed
epochs = 2 # Reduced epochs for speed
total_steps = len(train_dataloader) * epochs

# Scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

all_predictions, all_true_labels = [], []

def train_model(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    scaler = GradScaler()

    for epoch_i in range(epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            with autocast():
                outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.2f}")

        # Validation phase
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        all_predictions = []
        all_true_labels = []

        for batch in validation_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            with torch.no_grad():
                with autocast():
                    outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

                loss = outputs.loss
                logits = outputs.logits

            total_eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            total_eval_accuracy += accuracy_score(preds, label_ids)

            all_predictions.extend(preds)
            all_true_labels.extend(label_ids)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print(f"Validation Accuracy: {avg_val_accuracy:.2f}")

        if avg_val_accuracy > best_val_accuracy:
            print(f"Improved validation accuracy from {best_val_accuracy:.2f} to {avg_val_accuracy:.2f}. Saving model...")
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), 'best_model_state.bin')

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print(f"Validation Loss: {avg_val_loss:.2f}")

    print("Training complete!")

# After defining the train_model function, call it
train_model(model, train_dataloader, validation_dataloader, optimizer, scheduler, epochs=2)

print(classification_report(all_true_labels, all_predictions, target_names=['fake', 'reliable']))

# Convert lists to NumPy arrays for compatibility with Sklearn's metrics functions
all_predictions_np = np.array(all_predictions)
all_true_labels_np = np.array(all_true_labels)

# Print the classification report
print(classification_report(all_true_labels_np, all_predictions_np, target_names=['fake', 'reliable']))

# Calculate and print the final accuracy across all validation batches
final_val_accuracy = accuracy_score(all_true_labels_np, all_predictions_np)
print("Final Validation Accuracy: {:.2f}".format(final_val_accuracy))