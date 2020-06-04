import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
# choose dataset to process
dataset = 'yelp_ON'
raw_data_path = "/cluster/home/it_stu110/data/yelp/state/ON_reindex.csv"

# dataset = "ciao"
# raw_data_path = "/cluster/home/it_stu110/data/ciao/Ciao/done/ciao.csv"


processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
if not os.path.exists(processed_data_path):   
    os.mkdir(processed_data_path)

# names = ['user_id', 'business_id', 'stars', 'timestamp']
read = pd.read_csv(raw_data_path,  engine = 'python')



users = list(read['user_id'].unique())
user_id_index = dict((user_id, index) for user_id, index in zip(users, range(len(users))))
items = list(read['business_id'].unique())
business_id_index = dict((business_id, index) for business_id, index in zip(items, range(len(items))))
print("start reading")
# data = []
# with open(os.getcwd()+'/data/'+dataset+'/ratings.dat', 'r') as f:
#     lines = f.readlines()
#     count_user = 0
#     count_item = 0
#     for i in tqdm(range(len(lines))):
#         line = lines[i].strip().split('\t')
#         user_id = int(line[0])
#         business_id = int(line[1])
#         rating = float(line[2])

#         data.append([user_id_index[user_id], business_id_index[business_id], rating])



read['user_id'] = read['user_id'].map(user_id_index )
read['business_id'] = read['business_id'].map(user_id_index )

read = read.dropna(axis=0,how='any') #drop all rows that have any NaN values
read.reset_index(drop = True)

read['user_id'] = read['user_id'].astype('int')
read['business_id'] = read['business_id'].astype('int')
data = read[['user_id', 'business_id', 'stars']].values

# data = np.array(data)
np.random.shuffle(data)

pickle.dump(user_id_index, open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'wb'))
pickle.dump(business_id_index, open(os.path.join(processed_data_path, 'business_id_index.pkl'), 'wb'))
# np.savetxt(os.path.join(processed_data_path, 'data.txt'), data, fmt='%f')
np.save(os.path.join(processed_data_path, 'data.npy'), data)