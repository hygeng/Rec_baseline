import surprise
import os
from collections import defaultdict
import numpy as np
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
fname = '/cluster/home/it_stu110/proj/Rec_baseline/surprise/data/ciao/ciao_svdpp.csv'
# fname  ='/cluster/home/it_stu110/proj/baseline/surprise/data/yelp_ON/yelp_ON.csv'
file_path = os.path.expanduser(fname)
reader = Reader(line_format='user item rating', sep=' ')
data = Dataset.load_from_file(file_path, reader=reader)

trainset, testset = train_test_split(data, test_size=.2)
# We'll use the famous SVD algorithm.
algo = surprise.prediction_algorithms.matrix_factorization.SVDpp()
# Train the algorithm on the trainset, and predict ratings for the testset
algo.fit(trainset)
predictions = algo.test(testset)

# algo.predict(uid, iid)[3]

def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.
    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])
        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls

testset = trainset.build_anti_testset()
# predictions = algo.test(testset)

top_n = get_top_n(predictions, n=10)
recon_dict = {}
for uid, user_ratings in top_n.items():
    recon_dict[uid] = [int(iid) for (iid, _) in user_ratings]

precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4.5)

# import metric
# csv_path = "/cluster/home/it_stu110/proj/hisr/data/ciao/ciao/test.csv"
# df_ciao = pd.read_csv(csv_path,index_col = 0)
# n_users = df_ciao['user_id'].unique().shape[0]
# n_items = df_ciao['business_id'].unique().shape[0]

# batch_size = 128
# item_list = list(range(n_items))

# num_batches = int(len(df_ciao) / batch_size)
# for batch_idx in range(num_batches):
#     batch_users = list(range(batch_idx * batch_size, ( batch_idx +1 ) * batch_size ))
#     batch_df  = df_ciao.loc[df_ciao['user_id'].isin(batch_users)]
#     df_users  = batch_df['user_id'].values 
#     # array_users  = df_users - batch_idx * batch_size

#     df_items  = batch_df['business_id'].values
#     df_stars  = batch_df['stars'].values

#     true_batch = np.zeros((batch_size, n_items))
#     true_batch[df_users, df_items] = df_stars

#     recon_batch = np.zeros((batch_size, n_items))

# for user in tqdm(df_users):
#     for item in item_list:
#         recon_batch[user - batch_idx * batch_size, item]  =algo.predict(user, item)[3]
#     # for user, item in batch_df[['user_id','business_id']].values:
#     #     recon_batch[user - batch_idx * batch_size, item]  =algo.predict(user, item)[3]
#     metric.hit_rate(recon_batch,true_batch,10)

# recommend(uids=df_users, verbose=True)
