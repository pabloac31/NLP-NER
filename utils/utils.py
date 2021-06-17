import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, f1_score

def evaluate_corpus(sequences, sequences_predictions):
    """Evaluate classification accuracy at corpus level, comparing with
    gold standard."""
    total = 0.0
    correct = 0.0
    for i, sequence in enumerate(sequences):
        pred = sequences_predictions[i]
        for j, y_hat in enumerate(pred.y):
            if sequence.y[j] != "O":
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
    return correct / total


def show_confusion_matrix(sequences, preds, normalize=False):
    y_true = []
    y_pred = []
    for seq, pred in zip(sequences, preds):
        y_true.extend(seq.y)
        y_pred.extend(pred.y.tolist())

    cm = confusion_matrix(y_true, y_pred)

    threshold = 24953
    cm_clipped = np.clip(cm, a_min=0, a_max=threshold)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm_clipped, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title("Confusion matrix")
    #plt.colorbar()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = threshold / 1.5 if normalize else threshold / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    positions = list(sp.state_labels.values())
    labels = list(sp.state_labels.keys())

    plt.xticks(positions, labels)
    plt.yticks(positions, labels)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.show()


def get_f1_score(sequences, preds):
    y_true = []
    y_pred = []
    for seq, pred in zip(sequences, preds):
        y_true.extend(seq.y)
        y_pred.extend(pred.y.tolist())

    return f1_score(y_true, y_pred, average='weighted')
