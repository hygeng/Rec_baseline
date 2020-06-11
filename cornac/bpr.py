import cornac
import numpy as np
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
# train, test = python_random_split(data, 0.75)
# ml_100k= cornac.data.Dataset.from_uir(data.itertuples(index=False), seed=SEED)

dataset = "yelp_ON"
if dataset =="ciao":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/data_process/ciao/ciao_simple.txt"
elif dataset =="yelp_ON":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/data_process/yelp/yelp_ON_simple.txt"

reader  = cornac.data.reader.Reader()
data  = reader.read(fpath = data_path, fmt='UIR', sep=' ', skip_lines=0, id_inline=False, parser=None)

# ml_100k = cornac.datasets.movielens.load_feedback(variant="100K")

rs = cornac.eval_methods.RatioSplit(data=data, test_size=0.2, rating_threshold=1.0, seed=123)


bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=10)
recall = cornac.metrics.Recall(k=10)
ndcg = cornac.metrics.NDCG(k=10)


experi = cornac.Experiment(
  eval_method=rs,
  models=[ bpr],
  metrics=[mae, rmse, prec, recall, ndcg],
  user_based=True
)

experi.run()
# keys: ['MAE', 'RMSE', 'NDCG@10', 'Precision@10', 'Recall@10']

results = experi.result[0].metric_user_results
hr = list(results['Precision@10'].values())
ndcg = list(results['NDCG@10'].values())
recall = list(results['Recall@10'].values())

print(#"MAP:\t%f" % eval_map,
    "Precision@K:\t%f" % np.mean(hr),
      "NDCG:\t%f" % np.mean(ndcg),
      "Recall@K:\t%f" % np.mean(recall),
      "F1@10:\t%f"% (2*(np.mean(recall)* np.mean(hr))/(np.mean(hr) + np.mean(recall))), sep='\n')


