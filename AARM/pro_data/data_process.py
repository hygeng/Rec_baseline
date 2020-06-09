import pandas as pd
from tqdm import tqdm
import os

dataset = "ciao"
if dataset =="ciao":
    data_path = "/cluster/home/it_stu110/data/ciao/Ciao/done/ciao.csv"
elif dataset =="yelp_ON":
    data_path = "/cluster/home/it_stu110/data/yelp/state/ON_reindex.csv"
save_dir = "/cluster/home/it_stu110/proj/Rec_baseline/AARM/data/{}/".format(dataset)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


df = pd.read_csv(data_path,index_col = 0)[['user_id','business_id','stars','date','text']]
df_users = sorted(list(df["user_id"].unique()))
df_items = sorted(list(df["business_id"].unique()))
# def split():
df = df.sample(frac = 1)
# df_users = list(df["user_id"].unique())
user_group = df.groupby("user_id")
df_train = pd.DataFrame(columns=['user_id','business_id','stars','date','text'])
df_valid = pd.DataFrame(columns=['user_id','business_id','stars','date','text'])
df_test  = pd.DataFrame(columns=['user_id','business_id','stars','date','text'])
for user, group in tqdm(user_group):
    df_train = pd.concat([df_train,  group.iloc[:int(0.7*len(group))-1,:]  ])
    df_test =  pd.concat([df_test,  group.iloc[int(0.7*len(group))-1:len(group)-1,:] ])
    df_valid = pd.concat([df_valid,  group.iloc[-1,:].to_frame().T ], axis =0)

# write users
with open(save_dir + "users.txt", "w") as f:
    for user in df_users:
        f.write(str(user))
        f.write("\n")

# write items
with open(save_dir + "product.txt", "w") as f:
    for item in df_items:
        f.write(str(item))
        f.write("\n")
# save datasets
df_train.to_csv(save_dir + "train.csv")
df_valid.to_csv(save_dir + "valid.csv")
df_test.to_csv(save_dir + "test.csv")

# save pairs
df_train[['user_id','business_id']].to_csv(save_dir + "train_pairs.txt",header = False, sep = ",",index = False)
df_valid[['user_id','business_id']].to_csv(save_dir + "valid_pairs.txt",header = False, sep = ",",index = False)
df_test[['user_id','business_id']].to_csv(save_dir + "test_pairs.txt",header = False, sep = ",",index = False)


