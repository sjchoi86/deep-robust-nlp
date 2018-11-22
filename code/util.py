import os,re
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz", 
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz", 
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                      "aclImdb", "test"))
    return train_df, test_df

def mixup(data, targets, alpha):
    n = data.shape[0]
    indices = np.random.permutation(n)
    data2 = data[indices,:]
    targets2 = targets[indices,:]
    lam = np.random.beta(alpha,alpha,size=(n,1)) # Sample from beta. 
    data = data * lam + data2 * (1 - lam)
    targets = targets * lam + targets2 * (1 - lam)
    return data, targets

def print_n_txt(_f,_chars,_addNewLine=True,_DO_PRINT=True):
    if _addNewLine: _f.write(_chars+'\n')
    else: _f.write(_chars)
    _f.flush();os.fsync(_f.fileno()) # Write to txt
    if _DO_PRINT:
        print (_chars)
        
def gpusession(): 
    config = tf.ConfigProto(); 
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    return sess

def random_flip_target(_t_train,_rate,_seed=0):
    _t_random = np.copy(_t_train)
    n_train = np.shape(_t_train)[0] # n-train
    r_idx = np.random.permutation(n_train)[:(int)(n_train*_rate)] # random indices
    _t_random[r_idx] = [1,1]-_t_train[r_idx] # flip
    return _t_random

def create_gradient_clipping(loss,optm,vars,clipVal=1.0):
    grads, vars = zip(*optm.compute_gradients(loss, var_list=vars))
    grads = [None if grad is None else tf.clip_by_value(grad,-clipVal,clipVal) for grad in grads]
    op = optm.apply_gradients(zip(grads, vars))
    train_op = tf.tuple([loss], control_inputs=[op])
    return train_op[0]