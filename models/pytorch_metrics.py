import torch 
from torchmetrics import Metric

class BinaryAccuracy(Metric):
    """Calculates how often predictions matches labels.

    For example, if `y_true` is tensor([1, 1, 0, 0]) and `y_pred` is tensor([0.98, 1, 0, 0.6])
    then the binary accuracy is 3/4 or .75.  If the weights were specified as
    [1, 0, 0, 1] then the binary accuracy would be 1/2 or .5.

    Usage:

    ```python
    target = torch.tensor([1, 1, 0, 0])
    pred = torch.tensor([0.98, 1, 0, 0.6])
    bac = BinaryAccuracy()
    bac(pred, target)
    bac.compute() == 0.75
    ```
    """

    def __init__(self, device):
        super().__init__()
        self.total = torch.tensor(0, device=device)
        self.correct = torch.tensor(0, device=device)

    def update(self, preds, targets):
        batch_size = targets.size(0)
        threshold = torch.tensor(0.7)
        self.correct += torch.sum(
            torch.eq(
                torch.gt(preds, threshold), torch.gt(targets, threshold)))
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total