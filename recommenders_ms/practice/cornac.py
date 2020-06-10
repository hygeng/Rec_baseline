import cornac

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

dataset = "ciao"
if dataset =="ciao":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/surprise/data/ciao/ciao_svdpp.csv"
elif dataset =="yelp_ON":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/surprise/data/yelp_ON/yelp_ON.csv"

data  = cornac.data.reader.read(data_path, fmt='UIR', sep='\t', skip_lines=0, id_inline=False, parser=None)
# ml_100k = cornac.datasets.movielens.load_feedback(variant="100K")

rs = cornac.eval_methods.RatioSplit(data=ml_100k, test_size=0.2, rating_threshold=4.0, seed=123)


bpr = cornac.models.BPR(k=10, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123)
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=10)
recall = cornac.metrics.Recall(k=10)
ndcg = cornac.metrics.NDCG(k=10)




cornac.Experiment(
  eval_method=rs,
  models=[ bpr],
  metrics=[mae, rmse, recall, ndcg],
  user_based=True
).run()
