from collections import defaultdict

import numpy as np

# In this module I've implemented the feature engineering flow
# Note: this is not production ready code. The purpose of this module is
# to add modularity and facilitate the analysis on the notebooks


# below are the transaction types that occur in our dataset. One assumption I make here
# is that the August Holdout will contain a similar distribution of transaction types
transaction_types = ['PT', 'DT', 'CT', 'DD', 'DR', 'FT', 'BBU', 'BUB', 'TUB']


feature_names = [
    # transaction type features
    'PT_cnt', 'DT_cnt', 'CT_cnt', 'DD_cnt', 'DR_cnt', 'FT_cnt', 'BBU_cnt', 'BUB_cnt', 'TUB_cnt',
    # transaction frequency features
     'mean_tf', 'max_tf', '90th_tf',
    # dayofweek incoming transactions histogram
    'd0_in_freq', 'd1_in_freq', 'd2_in_freq', 'd3_in_freq', 'd4_in_freq', 'd5_in_freq', 'd6_in_freq',
    # dayofweek outgoing transactions  histogram
    'd0_out_freq', 'd1_out_freq', 'd2_out_freq', 'd3_out_freq', 'd4_out_freq', 'd5_out_freq', 'd6_out_freq',
    # quarter of month incoming transactions histogram
    'q0_in_freq', 'q1_in_freq', 'q2_in_freq', 'q3_in_freq',
    # quarter of month outgoing transactions histogram
    'q0_out_freq', 'q1_out_freq', 'q2_out_freq', 'q3_out_freq',
]


def _compute_transaction_type_features(ds, transaction_types=transaction_types):
    """Computes the amount of times each type was used by the user
    """
    feature_vectors = dict()
    groups = ds.groupby(['user_id', 'transaction_type']).transaction_date.count()
    cache = {u: {t: 0 for t in transaction_types} for u in ds.user_id.unique()}
    for (user, t), count in zip(groups.index, groups.values):
        cache[user][t] += count
        
    for user, counts in cache.items():
        feature_vectors[user] = [counts[t] for t in transaction_types]
    return feature_vectors


def _compute_transaction_freq_features(ds, daysinmonth):
    """Computes statistics like the mean transaction frequency per day
    In and Out transactions are not being treated differently
    
    Stats computed: mean, max, 90th percentile
    """
    # there is some duplication here, but we'd like those function to be atomic
    feature_vectors = dict()
    groups = ds.groupby(['user_id', 'transaction_date']).transaction_date.count()
    cache = defaultdict(list)
    for (user, date), count in zip(groups.index, groups.values):
        cache[user].append(count)
    
    for user in ds.user_id.unique():
        transactions_per_day = cache[user] + [0 for _ in range(daysinmonth - len(cache[user]))]
        feature_vectors[user] = [
            np.mean(transactions_per_day),
            np.max(transactions_per_day),
            np.percentile(transactions_per_day, 90),
        ]
    return feature_vectors


def _compute_dayofweek_histogram(ds):
    """Computes two histograms corresponding to Incoming/Outgoing amount per day of week
    Since a user can have only In or Out transaction, an vector of all zeros is a possible output
    """
    groups = ds.groupby(['user_id', 'transaction_date', 'transaction_direction']).amount_n26_currency.sum()
    week_in_dist, week_out_dist = dict(), dict()
    
    for (user, date, direction), amount in zip(groups.index, groups.values):
        dist = week_in_dist if direction == 'In' else week_out_dist
        if user not in dist:
            dist[user] = [0 for _ in range(7)]
        dist[user][date.dayofweek] += amount
    
    for user in ds.user_id.unique():
        for dist in [week_in_dist, week_out_dist]:
            if user not in dist:
                dist[user] = [0 for _ in range(7)]
            else:
                total = sum(dist[user])
                dist[user] = [v / total for v in dist[user]]
    return week_in_dist, week_out_dist

                
def _compute_quarter_histogram(ds):
    """Computes two histograms with 4 bins. Each bin corresponds to a quarter of the month.
    We naively split the month in four buckets i.e. [1-7, 8-14, 15-21, 21-end]. This creates
    an inconsistency with regards to the last bucket. A better al could be to split based
    on week on the month.
    """
    def quarter_in_month(day):
        if day <= 7:
            return 0
        elif day <= 14:
            return 1
        elif day <= 21:
            return 2
        else:
            return 3
    groups = ds.groupby(['user_id', 'transaction_date', 'transaction_direction']).amount_n26_currency.sum()
    quart_in_dist, quart_out_dist = dict(), dict()
    
    for (user, date, direction), amount in zip(groups.index, groups.values):
        dist = quart_in_dist if direction == 'In' else quart_out_dist
        if user not in dist:
            dist[user] = [0 for _ in range(4)]
        dist[user][quarter_in_month(date.day)] += amount
    
    for user in ds.user_id.unique():
        for dist in [quart_in_dist, quart_out_dist]:
            if user not in dist:
                dist[user] = [0 for _ in range(4)]
            else:
                total = sum(dist[user])
                dist[user] = [v / total for v in dist[user]]
    return quart_in_dist, quart_out_dist


def compute_feature_vectors(ds):
    """Expects the DataFrame that we created on the load steps
    
    Invokes all the feature calculation methods and creates a single feature
    vector for each (month, user) pair
    """
    feature_vectors = {m: defaultdict(list) for m in ds.month.unique()}
    occuring_transaction_types = ds.transaction_type.unique()

    for month in ds.month.unique():
        _ds = ds[ds.month == month]
        daysinmonth = _ds.transaction_date.iloc[0].daysinmonth
    
        # Features: counts of transactions types per user per month
        transaction_type_features = _compute_transaction_type_features(_ds)
        # Features: stats on transaction frequency (combined In/Out)
        transaction_freq_features = _compute_transaction_freq_features(_ds, daysinmonth)
        # Features: histograms for day of week of In/Out transactions
        dayofweek_histograms = _compute_dayofweek_histogram(_ds)
        # Features: histograms for quarter of the month In/Out transactions
        quarter_histograms = _compute_quarter_histogram(_ds)

        for user in _ds.user_id.unique():
            feature_vectors[month][user] += transaction_type_features[user]
            feature_vectors[month][user] += transaction_freq_features[user]
            feature_vectors[month][user] += dayofweek_histograms[0][user]
            feature_vectors[month][user] += dayofweek_histograms[1][user]
            feature_vectors[month][user] += quarter_histograms[0][user]
            feature_vectors[month][user] += quarter_histograms[1][user]
    return feature_vectors, feature_names