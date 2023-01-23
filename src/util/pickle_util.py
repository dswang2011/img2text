import pickle

def output_to_pickle(my_dict,output_path):
    with open(output_path,'wb') as fw:
        pickle.dump(my_dict,fw)

def load_pickle(picke_path):
    with open(picke_path,'rb') as fr:
        res = pickle.load(fr)
    return res

