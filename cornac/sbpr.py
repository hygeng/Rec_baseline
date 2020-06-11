import cornac
from cornac.data import Reader, GraphModality
from cornac.datasets import epinions
from cornac.eval_methods import RatioSplit
import numpy as np
dataset = "yelp_ON"
if dataset =="ciao":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/data_process/ciao/ciao_simple.txt"
    trust_path = "/cluster/home/it_stu110/proj/Rec_baseline/data_process/ciao/ciao_trust.txt"
elif dataset =="yelp_ON":
    data_path = "/cluster/home/it_stu110/proj/Rec_baseline/data_process/yelp/yelp_ON_simple.txt"
    trust_path = '/cluster/home/it_stu110/proj/Rec_baseline/data_process/yelp/yelp_ON_trust.txt'
# SBPR integrates user social network into Bayesian Personalized Ranking.
# The necessary data can be loaded as follows

reader = Reader()
trust = reader.read(trust_path, sep=' ')
feedback = reader.read(data_path, sep=' ')

# Instantiate a GraphModality, it makes it convenient to work with graph (network) auxiliary information
# For more details, please refer to the tutorial on how to work with auxiliary data
user_graph_modality = GraphModality(data=trust)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.1,
    rating_threshold=0.5,
    exclude_unknowns=True,
    verbose=True,
    user_graph=user_graph_modality,
)

# Instantiate SBPR model
sbpr = cornac.models.SBPR(
    k=10,
    max_iter=50,
    learning_rate=0.001,
    lambda_u=0.015,
    lambda_v=0.025,
    lambda_b=0.01,
    verbose=True,
)

# Use Recall@10 for evaluation
mae = cornac.metrics.MAE()
rmse = cornac.metrics.RMSE()
prec = cornac.metrics.Precision(k=10)
recall = cornac.metrics.Recall(k=10)
ndcg = cornac.metrics.NDCG(k=10)

# Put everything together into an experiment and run it
experi = cornac.Experiment(
    eval_method=ratio_split,  
    models=[ sbpr],
    metrics=[mae, rmse, prec, recall, ndcg],
    user_based=True
    )
experi.run()

results = experi.result[0].metric_user_results
MAE  = list(results['MAE'].values())
rmse = list(results['RMSE'].values())
hr = list(results['Precision@10'].values())
ndcg = list(results['NDCG@10'].values())
recall = list(results['Recall@10'].values())

print(#"MAP:\t%f" % eval_map,
    "Precision@K:\t%f" % np.mean(hr),
      "NDCG:\t%f" % np.mean(ndcg),
      "Recall@K:\t%f" % np.mean(recall),
      "F1@10:\t%f"% (2*(np.mean(recall)* np.mean(hr))/(np.mean(hr) + np.mean(recall))), sep='\n')