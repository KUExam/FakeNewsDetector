# visualizations.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torchviz
from model import ArticleClassifier


def visualize_model(input_size, hidden_size):
    print("Visualizing model...")
    
    # Initialize model
    model = ArticleClassifier(input_size, hidden_size)
    # Print model summary or architecture
    print(model)
    # Visualize the model architecture directly in a window
    x = torch.randn(1, input_size)
    visualization = torchviz.make_dot(model(x), params=dict(model.named_parameters()))
    # Display the image in a window
    plt.figure(figsize=(20, 20))
    plt.imshow(visualization.view())
    plt.show()
    
    print("Model visualization displayed.")


def visualization(y_true, y_pred):
    # We make a confusion matrix to give a clearer picture of our model's performance
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
