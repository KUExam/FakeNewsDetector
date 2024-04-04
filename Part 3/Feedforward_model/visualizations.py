import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def visualization(y_val, val_predictions):
    # We make a confusion matrix to give a clearer picture of our model's performance by showing the true positives, true negatives, false positives, and false negatives.
    cm = confusion_matrix(y_val, val_predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Fake', 'Reliable'], yticklabels=['Fake', 'Reliable'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
