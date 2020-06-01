import numpy as np


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


import math
# Calculates the normalized discounted cumulative gain at k
# https://github.com/makgyver/pyros/blob/master/pyros/core/evaluation.py
def ndcg_k(actual, predicted, topk):
    # k = min(k, len(predicted))
    idcg = idcg_k(topk)
    res = 0
    for user_id in range(len(actual)):
        dcg_k = sum([int(predicted[user_id][j] in set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


# def hit_rate(actual, predicted, topk):
#     num_users = len(predicted)
#     hits = 0.0
#     for u in range(num_users):
#         hit = 1.0 if precision_at_k_per_sample(actual[u], predicted[u][:topk], topk)>0 else 0.0
#         hits +=hit
#     return hits/num_users


def hit_rate(actual, predicted, topk):
    hits = 0.0
    num_users = len(predicted)
    hit_rate  = []
    for i in range(num_users):
        hits = 0
        act_set = set(actual[i])
        pred_set = predicted[i][:topk]
        for item in pred_set:
            if item in act_set:
                hits += 1.0
        hit_rate.append(hits/topk)
    return float(sum(hit_rate))/num_users 

if __name__ == '__main__':
    actual = [[1, 2], [3, 4, 5]]
    predicted = [[10, 20, 1, 30, 40], [10, 3, 20, 4, 5]]
    print(ndcg_k(actual, predicted, 5))