import numpy as np
import torch
# import bottleneck as bn

def evaluate(X_pred, X_true, topk=10):
    batch_users = X_pred.shape[0]
    tp = np.reshape(1. / np.log2(np.arange(2, topk + 2)), [1, topk])

    # idx = bn.argpartition(-X_pred, topk, axis=1)
    idx = np.argsort(-X_pred, axis=-1)
    X_true_binary = (X_true > 0)
    results = X_true_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :topk]].astype(np.float32)

    Pu = np.sum(results, axis=1)
    # Tu = np.sum(X_true_binary, axis=1).astype(np.float32)
    tasks = np.asarray(X_true.sum(1)).flatten().nonzero()[0]
    Tu = np.asarray(X_true[tasks].sum(1)).flatten().astype(np.float32)

    HR = np.mean(np.sign(Pu))

    vP = np.divide(Pu, topk)
    PR = np.mean(vP)  # Precision

    vR = np.divide(Pu, np.minimum(Tu, topk)) 
    RC = np.mean(vR)  # Recall

    vA = vP + vR
    F1 = np.mean(2 * vP * vR / np.where(vA == 0, 1., vA)) # F1

    score = 1. / np.log2(np.arange(2., 2. + topk))
    DCG = np.sum(results * score[np.newaxis, :], 1)
    pDCG = np.cumsum(score)[np.minimum(Tu, topk).astype(np.int32) - 1]
    NDCG = np.mean(DCG / pDCG) #NDCG

    return HR, PR,  NDCG, RC, F1

if __name__ == '__main__':
    device = "cpu"
    te = torch.load("./log/batch_te.pth",map_location = device).detach().cpu().numpy()
    approx = torch.load("./approx.pth",map_location = device).detach().cpu().numpy()
    res = approx
    HR, PR,  NDCG, RC, F1 = evaluate(res, te, 10)
    print(HR, PR,  NDCG, RC, F1)
