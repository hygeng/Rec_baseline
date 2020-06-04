from __future__ import print_function
from evaluations import *
from pmf_model import *
import metric
from tqdm import tqdm
# print('PMF Recommendation Model Example')

# choose dataset name and load dataset, 'ml-1m', 'ml-10m'
dataset = 'yelp_ON'
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'))
item_id_index = pickle.load(open(os.path.join(processed_data_path, 'business_id_index.pkl'), 'rb'))
# data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)
data = np.load(os.path.join(processed_data_path, 'data.npy'))
# set split ratio
ratio = 0.6
train_data = data[:int(ratio*data.shape[0])]
vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]
test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]

NUM_USERS = max(user_id_index.values()) + 1
NUM_ITEMS = max(item_id_index.values()) + 1
print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))


R = np.zeros([NUM_USERS, NUM_ITEMS])
for ele in train_data:
    R[int(ele[0]), int(ele[1])] = float(ele[2])

# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 20
lr = 3e-5
iters = 500
model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(ratio, lambda_alpha, lambda_beta, latent_size,lr, iters))
U, V, train_loss_list, vali_rmse_list = model.train(train_data=train_data, vali_data=vali_data)

# print('testing model.......')
# preds = model.predict(data=test_data)
# print("pred:",preds.shape)

n_users = len(user_id_index)
n_items = len(item_id_index)
all_users = list(user_id_index.values())

hr = []
ndcg  = []
recall = []
batch_size = 128
# item_list = list(range(n_items))
np.set_printoptions(suppress=True)
num_batches = int(n_users/ batch_size)
for batch_idx in tqdm(range(num_batches)):
    start  = batch_idx * batch_size
    end = min(( batch_idx +1 ) * batch_size,len(test_data) )
    batch_users = all_users[ start : end ]
    batch_len = len(batch_users)
    # array_users  = df_users - batch_idx * batch_size
    batch_test_data = test_data[np.logical_or.reduce([test_data[:,0]==x for x in batch_users])]

    true_batch = np.zeros((batch_len, n_items))
    true_batch[batch_test_data[:,0] - start, batch_test_data[:,1]] = batch_test_data[:,2]

    batch_test_data_fake = np.zeros((batch_len,n_items, 3))
    batch_test_data_fake[:,:,1] = np.arange(n_items).repeat(batch_len).reshape(n_items, batch_len).transpose()
    batch_test_data_fake[:,:,0] = np.array(batch_users).repeat(n_items).reshape(batch_len, n_items)
    batch_test_data_fake[:,:,2]  = 1
    batch_test_data_fake = batch_test_data_fake.reshape(-1,3)

    preds = model.predict(data=batch_test_data_fake).reshape(batch_len,n_items)
    hr.append(metric.hit_rate(preds, true_batch))
    ndcg.append(metric.NDCG_binary_at_k_batch(preds, true_batch))
    recall.append(metric.precision_recall_at_k(preds, true_batch))
    # print("hit rate: {:6.6f} ndcg: {:6.6f} recall: {:6.6f}".format(np.mean(hr), np.mean(ndcg), np.mean(recall)))

# test_rmse = RMSE(preds, test_data[:, 2])
print("hit rate: {:6.6f} ndcg: {:6.6f} recall: {:6.6f}".format(np.mean(hr), np.mean(ndcg), np.mean(recall)))
# print('test rmse:{:f}'.format(test_rmse))
