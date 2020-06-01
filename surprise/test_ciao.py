import surprise
import os
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import accuracy
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
fname = '/cluster/home/it_stu110/proj/baseline/surprise/data/ciao/ciao_svdpp.csv'
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

algo.predict(uid, iid)[3]

import pandas as pd
csv_path = "/cluster/home/it_stu110/data/ciao/Ciao/done/ciao.csv"
df_ciao = pd.read_csv(csv_path,index_col = 0)
n_users = df_ciao['user_id'].unique().shape[0]
n_items = df_ciao['business_id'].unique().shape[0]

batch_size = 128

