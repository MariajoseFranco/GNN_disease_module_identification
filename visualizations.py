import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve


def plot_loss_and_metrics(losses, val_f1s, val_recalls, val_precisions, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Train Loss')
    plt.plot(val_f1s, label='Val F1-score')
    plt.plot(val_recalls, label='Val Recall')
    plt.plot(val_precisions, label='Val Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Training Progress')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true, y_scores, save_path=None):
    plt.figure(figsize=(6, 5))
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
