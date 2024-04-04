# visualizations.py
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch
import torchviz
from model import ArticleClassifier

def visualize_model(input_size, hidden_size):
    # Initialize model
    model = ArticleClassifier(input_size, hidden_size)
    
    # Visualize the model architecture
    x = torch.randn(1, input_size)
    torchviz.make_dot(model(x), params=dict(model.named_parameters()))

def visualization(y_val, val_predictions):
    # We make a confusion matrix to give a clearer picture of our model's performance
    cm = confusion_matrix(y_val, val_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
