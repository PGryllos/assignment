# utilities for splitting the training dataset
# Those methods are useful only in the context of the notebook analysis
# I separated them in different modules to use from multiple notebooks
import numpy as np


def split_dataset(user_finances, month, holdout_users, threshold=0):
    """Splits the index and target var into train test split for both Income (in) and
    Expenses (out) classifier
    
    The method applies filtering, based on the provided threshold, on the target vars
    """
    train = user_finances[(user_finances.month != month) & ~user_finances.user_id.isin(holdout_users)]
    val = user_finances[(user_finances.month == month) & ~user_finances.user_id.isin(holdout_users)]
    test = user_finances[(user_finances.month == month) & user_finances.user_id.isin(holdout_users)]

    train_in, train_out = train[train['in'] > threshold], train[train['out'] > threshold]
    val_in, val_out = val[val['in'] > threshold], val[val['out'] > threshold]
    test_in, test_out = test[test['in'] > threshold], test[test['out'] > threshold]
    return train_in, train_out, val_in, val_out, test_in, test_out


def get_X_y(ds, feature_vectors, direction):
    X = []
    for m, u in zip(ds.month.values, ds.user_id.values):
        X.append(feature_vectors[m][u])
    return np.array(X), ds[direction].values