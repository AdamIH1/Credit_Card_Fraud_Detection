from sklearn.metrics import *
from mlxtend.plotting import plot_confusion_matrix 
from mlxtend.evaluate import confusion_matrix
import matplotlib.pyplot as plt

def classification_metrics(y_test, y_pred):

    desiered_metrics = [f1_score, precision_score,recall_score, accuracy_score]
    str_metrics = ['f1_score', 'precision_score','recall_score', 'accuracy_score']
    for i in range(len(desiered_metrics)): 
        act_metric = desiered_metrics[i](y_test,y_pred)
        print(f'{str_metrics[i]}: {act_metric:.4f}')

def performance_eval(y_test, y_pred, y_scores_proba):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    cm = confusion_matrix(y_target=y_test, y_predicted=y_pred)
    plot_confusion_matrix(conf_mat=cm, cmap=plt.cm.Greens, axis=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('Actual Label')

    fpr, tpr, threshold = roc_curve(y_test, y_scores_proba)
    roc_auc = auc(fpr, tpr)

    ax2.plot(fpr, tpr, 'b', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax2.plot([0, 1], [0, 1], 'r--')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([0, 1.02])
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_title(f'ROC Curve')
    ax2.legend(loc='lower right')

    precision, recall, _ = precision_recall_curve(y_test, y_scores_proba)
    pr_auc = auc(recall, precision)

    ax3.plot(recall, precision, 'b', label=f"PR Curve (AUC = {pr_auc:.2f})")
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([0, 1.02])
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title(f'Precision-Recall Curve')
    ax3.legend(loc='lower left')

    plt.tight_layout()
    plt.show()