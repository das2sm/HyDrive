import pickle

with open("data/infos/b2d_infos_train.pkl", 'rb') as f:
    data1 = pickle.load(f)
with open("data/infos/b2d_infos_val.pkl", 'rb') as f:
    data2 = pickle.load(f)

with open("data/infos/b2d_infos_trainval.pkl",'wb') as f:
    data = data1 + data2
    pickle.dump(data, f)
import ipdb; ipdb.set_trace()