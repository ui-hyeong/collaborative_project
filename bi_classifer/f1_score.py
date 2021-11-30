import torch
from sklearn.metrics import f1_score


def f1score(y, y_hat) :
    y = y.cpu()
    y_hat = y_hat.cpu()
    f1 = f1_score(y, y_hat, average='micro')

    return f1

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc