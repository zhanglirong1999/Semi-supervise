from pytorch_metrics import BinaryAccuracy
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# metrics = BinaryAccuracy('cuda').to('cuda')
# metrics.update(torch.tensor([1, 1, 0, 0]), torch.tensor([0.98, 1, 0, 0.6]))
# print(metrics.compute())
f1 = f1_score(y_true=[1, 1, 0, 0], y_pred=[0.98, 1, 0, 0.6], average='macro')
print(f1)