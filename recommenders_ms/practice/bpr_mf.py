import sys
import os
print("adding: ", os.getcwd(), " to path")
sys.path.append(os.getcwd())
import cornac
import papermill as pm
import pandas as pd
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from reco_utils.recommender.cornac.cornac_utils import predict_ranking
from reco_utils.common.timer import Timer
from reco_utils.common.constants import SEED



# top k items to recommend
TOP_K = 10

# Model parameters
NUM_FACTORS = 200
NUM_EPOCHS = 100
MOVIELENS_DATA_SIZE = '100k'
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=["userID", "itemID", "rating"]
)

data.head()
train, test = python_random_split(data, 0.75)
train_set = cornac.data.Dataset.from_uir(train.itertuples(index=False), seed=SEED)


print('Number of users: {}'.format(train_set.num_users))
print('Number of items: {}'.format(train_set.num_items))


bpr = cornac.models.BPR(# cornac.models.bpr.recom_bpr.BPR(  #cornac.models.BPR(
    k=NUM_FACTORS,
    max_iter=NUM_EPOCHS,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=SEED
)
bpr.train_set = train_set


with Timer() as t:
    all_predictions = predict_ranking(bpr, train, usercol='userID', itemcol='itemID', remove_seen=True)
print("Took {} seconds for prediction.".format(t))

k = 10
eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=k)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=k)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')
for uid, user_idx in bpr.train_set.uid_map.items():
    break


# Record results with papermill for tests
pm.record("map", eval_map)
pm.record("ndcg", eval_ndcg)
pm.record("precision", eval_precision)
pm.record("recall", eval_recall)