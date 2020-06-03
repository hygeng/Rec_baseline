import sys
sys.path.append("../")
import time
import pandas as pd
import tensorflow as tf
from numpy import np

from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.dataset import movielens
from reco_utils.common.notebook_utils import is_jupyter
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)

print("System version: {}".format(sys.version))
print("Pandas version: {}".format(pd.__version__))
print("Tensorflow version: {}".format(tf.__version__))
# add gpu growth flags
# tf_config.gpu_options.allow_growth = True
# tf_config.gpu_options.per_process_gpu_memory_fraction = 0.1

# top k items to recommend
TOP_K = 10

# Model parameters
EPOCHS = 50
BATCH_SIZE = 128

SEED = 42
dataset = "ciao"
if dataset =="ciao":
    data_path = "/clust:er/home/it_stu110/data/ciao/ciao_with_rating_timestamp/rating_with_timestamp.mat"
    import scipy.io as scio
    data = scio.loadmat(data_path)
    #  userid, productid, categoryid, rating, helpfulness and  time point
    df = pd.DataFrame(data['rating'][:,[0,1,3,5]],  columns=["userID", "itemID", "rating", "timestamp"])
elif dataset =="yelp_ON":
    data_path = "/cluster/home/it_stu110/data/yelp/state/ON_reindex.csv"
    df = pd.read_csv(data_path,index_col = 0)[['user_id','business_id','stars','date']]
    df.columns = ["userID", "itemID", "rating", "timestamp"]
elif dataset == "movielens":
    MOVIELENS_DATA_SIZE = '100k'
    df = movielens.load_pandas_df(
        size=MOVIELENS_DATA_SIZE,
        header=["userID", "itemID", "rating", "timestamp"]
    )


# Select MovieLens data size: 100k, 1m, 10m, or 20m
# MOVIELENS_DATA_SIZE = '100k'
# df = movielens.load_pandas_df(
#     size=MOVIELENS_DATA_SIZE,
#     header=["userID", "itemID", "rating", "timestamp"]
# )


train, test = python_chrono_split(df, 0.75)
print("start getting data")
data = NCFDataset(train=train, test=test, seed=SEED)
print("start getting model")
model = NCF (
    n_users=data.n_users, 
    n_items=data.n_items,
    model_type="NeuMF",
    n_factors=4,
    layer_sizes=[16,8,4],
    n_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=1e-3,
    verbose=1,
    seed=SEED
)
print("start training data")
start_time = time.time()

model.fit(data)

train_time = time.time() - start_time

print("Took {} seconds for training.".format(train_time))


start_time = time.time()

users, items, preds = [], [], []
item = list(train.itemID.unique())
for user in train.userID.unique():
    user = [user] * len(item) 
    users.extend(user)
    items.extend(item)
    preds.extend(list(model.predict(user, item, is_list=True)))

all_predictions = pd.DataFrame(data={"userID": users, "itemID":items, "prediction":preds})

merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

test_time = time.time() - start_time
print("Took {} seconds for prediction.".format(test_time))

eval_map = map_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_ndcg = ndcg_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_precision = precision_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)
eval_recall = recall_at_k(test, all_predictions, col_prediction='prediction', k=TOP_K)

print("MAP:\t%f" % eval_map,
      "NDCG:\t%f" % eval_ndcg,
      "Precision@K:\t%f" % eval_precision,
      "Recall@K:\t%f" % eval_recall, sep='\n')

      