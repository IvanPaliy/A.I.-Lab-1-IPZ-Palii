import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
df.head()


# confusion_matrix
def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))


def palii_confusion_matrix(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


palii_confusion_matrix_rf = palii_confusion_matrix(df.actual_label.values, df.predicted_RF.values)

sklearn_confusion_matrix_rf = confusion_matrix(df.actual_label.values, df.predicted_RF.values)

assert np.array_equal(palii_confusion_matrix_rf,
                      sklearn_confusion_matrix_rf), 'Custom confusion matrix is not correct for RF'

custom_confusion_matrix_lr = palii_confusion_matrix(df.actual_label.values, df.predicted_LR.values)
sklearn_confusion_matrix_lr = confusion_matrix(df.actual_label.values, df.predicted_LR.values)
assert np.array_equal(custom_confusion_matrix_lr,
                      sklearn_confusion_matrix_lr), 'Custom confusion matrix is not correct for LR'


# accuracy score
def palii_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


custom_accuracy_rf = palii_accuracy_score(df.actual_label.values, df.predicted_RF.values)
sklearn_accuracy_rf = accuracy_score(df.actual_label.values, df.predicted_RF.values)
assert custom_accuracy_rf == sklearn_accuracy_rf, 'Custom accuracy score failed for RF'

custom_accuracy_lr = palii_accuracy_score(df.actual_label.values, df.predicted_LR.values)
sklearn_accuracy_lr = accuracy_score(df.actual_label.values, df.predicted_LR.values)
assert custom_accuracy_lr == sklearn_accuracy_lr, 'Custom accuracy score failed for LR'

print('Accuracy RF: %.3f' % custom_accuracy_rf)
print('Accuracy LR: %.3f' % custom_accuracy_lr)

# recall score
recall_rf_sklearn = recall_score(df.actual_label.values, df.predicted_RF.values)


def my_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    recall = TP / (TP + FN)
    return recall


custom_recall_rf = my_recall_score(df.actual_label.values, df.predicted_RF.values)
assert custom_recall_rf == recall_rf_sklearn, 'Custom recall score failed for RF'

recall_lr_sklearn = recall_score(df.actual_label.values, df.predicted_LR.values)

custom_recall_lr = my_recall_score(df.actual_label.values, df.predicted_LR.values)
assert custom_recall_lr == recall_lr_sklearn, 'Custom recall score failed for LR'

print('Recall RF: %.3f' % custom_recall_rf)
print('Recall LR: %.3f' % custom_recall_lr)

# precision score
precision_rf_sklearn = precision_score(df.actual_label.values, df.predicted_RF.values)


def my_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_TP(y_true, y_pred), find_FN(y_true, y_pred), find_FP(y_true, y_pred), find_TN(y_true, y_pred)
    precision = TP / (TP + FP)
    return precision


custom_precision_rf = my_precision_score(df.actual_label.values, df.predicted_RF.values)
assert custom_precision_rf == precision_rf_sklearn, 'Custom precision score failed for RF'

precision_lr_sklearn = precision_score(df.actual_label.values, df.predicted_LR.values)

custom_precision_lr = my_precision_score(df.actual_label.values, df.predicted_LR.values)
assert custom_precision_lr == precision_lr_sklearn, 'Custom precision score failed for LR'

print('Precision RF: %.3f' % custom_precision_rf)
print('Precision LR: %.3f' % custom_precision_lr)

# F1 score
f1_rf_sklearn = f1_score(df.actual_label.values, df.predicted_RF.values)


def my_f1_score(y_true, y_pred):
    recall = my_recall_score(y_true, y_pred)
    precision = my_precision_score(y_true, y_pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


custom_f1_rf = my_f1_score(df.actual_label.values, df.predicted_RF.values)
assert custom_f1_rf == f1_rf_sklearn, 'Custom F1 score failed for RF'

f1_lr_sklearn = f1_score(df.actual_label.values, df.predicted_LR.values)

custom_f1_lr = my_f1_score(df.actual_label.values, df.predicted_LR.values)
assert custom_f1_lr == f1_lr_sklearn, 'Custom F1 score failed for LR'

print('F1 RF: %.3f' % custom_f1_rf)
print('F1 LR: %.3f' % custom_f1_lr)

fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)
print('AUC RF:%.3f' % auc_RF)
print('AUC LR:%.3f' % auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
