import bottleneck as bn
import numpy as np
import sys
import math

def hit_rate(X_pred, X_true, k=10):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (X_true > 0)
    hits_num = np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)
    return np.mean(hits_num/k)

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=10):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    """
    batch_users = X_pred.shape[0]
    # x_pred_binary = (X_pred>0)*1
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    # 
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1.0 / np.log2(np.arange(2, k + 2))
    # 
    DCG = (np.int8(heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk]>0) * tp).sum(axis=1)
    # IDCG = np.array([(tp[: min(n, k)]).sum() for n in (heldout_batch!=0).sum(axis=1)])
    IDCG = tp[: k].sum()
    return np.mean(np.nan_to_num(DCG / IDCG))

def precision_recall_at_k(X_pred, X_true, k=10):
    num_users = len(X_pred)
    actual = [[] for _ in range(num_users)]
    where = np.where(X_true!=0)
    for idx in range(len(where[0])):
        actual[where[0][idx]].append(where[1][idx])
    # 
    rank = np.argsort(-X_pred)
    predicted = rank[:,:k]
    sum_recall = 0.0
    sum_precision = 0.0
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i])
        if len(act_set) != 0:
            sum_precision += len(act_set & pred_set) / float(k)
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_precision / true_users, sum_recall / true_users







