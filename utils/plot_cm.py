print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

log_eval = '../../exp/AID/20percent/log_eval/'
y_pred = open(log_eval + 'predictions.txt', 'r')
y_pred = np.asarray(y_pred.readlines(), dtype=int)

y_test = open(log_eval + 'labels.txt', 'r')
y_test = np.asarray(y_test.readlines(), dtype=int)

class_names = open('labels.txt', 'r')
class_names = np.asarray(class_names.readlines())
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          fontsize = 7.5,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # if not title:
    #     if normalize:
            # title = 'Normalised confusion matrix'
        # else:
        #     title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    y_true = np.reshape(y_true,(-1,1))
    y_pred = np.reshape(y_pred,(-1,1))
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalised confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots(figsize=(15,15))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
             rotation_mode="anchor", fontsize=15)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", fontsize=15)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    # thresh = 0.1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),fontsize=fontsize,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh or cm[i, j] < 0.03 else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, y_pred, classes=class_names,
#                       title='Confusion matrix, without normalization')
#
# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)
                      # title='Normalised Confusion Matrix')
# plt.savefig(log_eval+'rtn_nwpu_10percent.png', format='png', dpi=512)
plt.show()
