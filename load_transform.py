import pandas as pd


def load_transform_dataset(path, copy=False, shuffle=True):
    """Loads the dataset from a csv file into a pandas DataFrame and performs
    necessary transformations
    
    Returns the transformed dataset as well as a DataFrame with the target
    variables (income/expenses per user per month) 
    """
    if not isinstance(path, str) or not path.endswith('.csv'):
        raise RuntimeWarning('Please provide a path to the dataset')
    
    ds = pd.read_csv(path)
    if shuffle:
        ds = ds.sample(frac=1).reset_index()
    
    mcc_lookup = pd.read_csv('mcc_group_definition.csv').set_index('mcc_group').to_dict()
    transaction_type_lookup = pd.read_csv('transaction_types.csv').set_index('type').to_dict()

    ds.transaction_date = pd.to_datetime(ds.transaction_date)
    ds['transaction_direction'] = ds.transaction_type.apply(
        lambda t: transaction_type_lookup['direction'][t]
    )
    ds['transaction_agent'] = ds.transaction_type.apply(
        lambda t: transaction_type_lookup['agent'][t]
    )
    
    ds['day'] = ds.transaction_date.apply(lambda d: d.day_name())
    ds['day_of_year'] = ds.transaction_date.apply(lambda d: d.dayofyear)
    ds['day_of_week'] = ds.transaction_date.apply(lambda d: d.dayofweek)
    ds['month'] = ds.transaction_date.apply(lambda d: d.month_name())
    return ds, _create_income_expenses_df(ds)


def _create_income_expenses_df(ds):
    """Expects a DataFrame as constructed by `load_transform_dataset`
    
    Returns a DataFrame with the target variables
    """
    in_idx  = ds.transaction_direction == 'In'
    out_idx = ds.transaction_direction == 'Out'
    
    user_finances = pd.concat([
        ds[in_idx].groupby(['user_id', 'month'])['amount_n26_currency'].sum().rename('in'),
        ds[in_idx].groupby(['user_id', 'month'])['amount_n26_currency'].count().rename('count_in'),
        ds[out_idx].groupby(['user_id', 'month'])['amount_n26_currency'].sum().rename('out'),
        ds[out_idx].groupby(['user_id', 'month'])['amount_n26_currency'].count().rename('count_out')
    ], join='outer', axis=1).fillna(0).reset_index()
    user_finances['net'] = user_finances['in'] - user_finances['out']
    
    return user_finances
