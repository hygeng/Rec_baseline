import pandas as pd
from tqdm import tqdm
fname = "/cluster/home/it_stu110/data/yelp/state/ON_reindex.csv"
df = pd.read_csv(fname,index_col = 0)

df['user_id'] = df['user_id'].apply(lambda x: x+1)
df['business_id'] = df['business_id'].apply(lambda x: x+1)

'''
df_all = df

len(df_all['user_id'].unique())
len(df_all['business_id'].unique())

users_per_item = 5
items_per_user = 5

# state = "ON"

prev_len = len(df_all)
x = df_all.groupby('business_id').filter(lambda x: len(x) >= users_per_item)
next_len = len(x)
while (next_len < prev_len):
    print(next_len, prev_len)
    prev_len = next_len
    x = x.groupby('user_id').filter(lambda x: len(x) >= items_per_user)
    x = x.groupby('business_id').filter(lambda x: len(x) >= users_per_item)
    next_len = len(x)

df_all = x
len(df_all['user_id'].unique())
len(df_all['business_id'].unique())
'''


# f = open('Beauty.txt', 'w')
# for user in User.keys():
#     for i in User[user]:
#         f.write('%d %d\n' % (user, i[1]))
# f.close()

uids = []
iids = []
data_grouped_by = df.groupby('user_id')
for uid, group in tqdm(data_grouped_by):
    user_time_rank = group.sort_values(by=['date'])
    iid = list(user_time_rank['business_id'].values)
    iids.extend(iid)
    uids.extend([uid for _ in range(len(iid))])

write_df = pd.DataFrame({"user_id":uids,"business_id":iids})
write_df.to_csv('data/yelp_ON.txt',sep = ' ',header=None, index=False)