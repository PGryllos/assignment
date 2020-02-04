import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


months = ['February', 'March', 'April', 'May', 'June', 'July']
days   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def plot_regressor_output(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2)
    
    plt.suptitle('Regression performance of {}'.format(model_name))
    axes[0].scatter(y_true, y_pred)
    axes[0].set_title('Predicted vs True')
    axes[0].set_ylabel('y_pred')
    axes[0].set_xlabel('y_true')
    
    axes[1].scatter(np.log1p(np.sort(y_true)), np.log1p(np.sort(y_pred)))
    axes[1].plot([0, 7], [0, 7], '--r')
    axes[1].set_title('Quantile-Quantile plot')
    axes[1].set_ylabel('y_pred')
    axes[1].set_xlabel('y_true')
    plt.show()


def plot_residual_distribution(classifiers_perf, direction):
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['figure.dpi'] = 80

    fig, axes = plt.subplots(2, 3)
    plt.suptitle('Distribution of residuals for {} classifier'.format('Income' if direction == 'in' else 'Expenses'))
    for i, month in enumerate(['February', 'March', 'April', 'May', 'June', 'July']):
        _axes = axes[i // 3, i % 3]
        for clf in classifiers_perf:
            sns.distplot(
                classifiers_perf[clf][month][direction]['holdout']['residuals'], hist_kws={'alpha': 0.4},
                hist=True, label=clf, kde=False, norm_hist=True, ax=_axes
            )
        _axes.set_title(month)
        _axes.legend()


def compare_model_performance(metric, classifiers_perf):
    plt.rcParams['figure.figsize'] = (12, 4)
    plt.rcParams['figure.dpi'] = 80

    fig, axes = plt.subplots(1, 2)

    mean_score = {k: {0: {}, 1: {}} for k in classifiers_perf}
    for clf in classifiers_perf:
        for i in range(2):
            for symbol, alpha, split in [('-^', .4, 'train'), ('-*', .5,'validation'), ('-o', .8,'holdout')]:
                score = []
                for month in months:
                    score.append(classifiers_perf[clf][month]['in' if i == 0 else 'out'][split][metric]) 
                mean_score[clf][i][split] = np.mean(score), np.std(score)
                if split == 'train':
                    continue
                
                axes[i].plot(months, score, symbol, label='{}_{}'.format(clf, split), alpha=alpha)
                axes[i].legend()
                axes[i].set_xlabel('Holdout Month')
                axes[i].set_ylabel(metric.upper())
                axes[i].set_title('{} predictor {}'.format('income' if i == 0 else 'expenses', metric.upper()))
    plt.show()

    for name, i in [('Income', 0), ('Expenses', 1)]:
        for clf in mean_score:
            print ('{} {} Predictor {} score'.format(clf.upper(), name, metric.upper()))
            print ('Train \t\tmean {} std {}\nValidation \tmean {} std {}\nHoldout \tmean {} std {}\n'.format(
                round(mean_score[clf][i]['train'][0], 2), round(mean_score[clf][i]['train'][1], 2),
                round(mean_score[clf][i]['validation'][0], 2), round(mean_score[clf][i]['validation'][1], 2),
                round(mean_score[clf][i]['holdout'][0], 2), round(mean_score[clf][i]['holdout'][1], 2)
            ))
            

def plot_monthly_finance_distributions(user_finances, idx_1, idx_2, label_1, label_2, log_transform=False):
    plt.rcParams['figure.figsize'] = (14, 6)
    plt.rcParams['figure.dpi'] = 50

    fig, axes = plt.subplots(2, 3)
    for i, month in enumerate(['February', 'March', 'April', 'May', 'June', 'July']):
        _axes = axes[i // 3, i % 3]

        for c, alpha, label, idx in [('g', .5, label_1, idx_1), ('r', .3, label_2, idx_2)]:
            if log_transform == True:
                target_var = np.log1p(user_finances[user_finances.month == month][idx])
            else:
                target_var = user_finances[user_finances.month == month][idx]

            sns.distplot(
                target_var, color=c, hist_kws={'alpha': alpha}, hist=True,
                label=label, kde=False, norm_hist=True, ax=_axes
            )
        _axes.set_xlabel('')
        _axes.set_title(month)
        _axes.legend()


def plot_week_over_week_transactions(ds, direction=None, transaction_type=None):
    assert direction in {'in', 'out'}
    in_idx  = ds.transaction_direction == 'In'
    out_idx = ds.transaction_direction == 'Out'
    week_over_week = {k: {'week_{}'.format(i): [] for i in range(5)} for k in months}

    fig, axes = plt.subplots(2, 3)

    for month in months:
        for day in range(7):
            idx = in_idx if direction == 'in' else out_idx
            idx = idx & (ds.transaction_type == transaction_type) if transaction_type else idx
            weekly_count = ds[
                idx & (ds.month == month) & (ds.day_of_week == day)
            ].groupby('day_of_year').transaction_date.count().values
            for i, c in enumerate(weekly_count):
                week_over_week[month]['week_{}'.format(i)].append(c)

    for i, month in enumerate(months):
        _axes = axes[i // 3, i % 3]
        for week, counts in week_over_week[month].items():
            _axes.plot([days[i] for i in range(len(counts))], counts, label=week)
            _axes.legend()
            _axes.set_title(month)

            
def plot_features_correlation(feature_vectors):
    features = []
    for month in feature_vectors:
        for user in feature_vectors[month]:
            features.append(feature_vectors[month][user])

    corr = pd.DataFrame(features).corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    plt.title('Feature Correlation Matrix')
    plt.show()


def visualize_finances(user_finances):
    fig, axes = plt.subplots(1, 3)

    for i, direction in enumerate(['in', 'out', 'net']):
        finances_stats = {}
        finances_stats['mean'] = user_finances.groupby('month')[direction].mean().to_dict()
        for k, v in [('q25', .25), ('q50', .5), ('q75', .75), ('q90', .90)]:
            finances_stats[k] = user_finances.groupby('month')[direction].quantile(v).to_dict()

        for stat, values in finances_stats.items():
            axes[i].plot(months, [values[m] for m in months], label=stat)

        axes[i].set_title(direction)
        axes[i].legend(loc='upper right')


def plot_transaction_type_dist(ds, amount=True):
    in_idx  = ds.transaction_direction == 'In'
    out_idx = ds.transaction_direction == 'Out'
    fig, axes = plt.subplots(2, 1)
    plt.suptitle('transaction type distribution (by {})'.format('amount' if amount else 'count'))
    
    if not amount:
        sns.countplot(ds[in_idx].transaction_type, ax=axes[0])
        sns.countplot(ds[out_idx].transaction_type, ax=axes[1])
    else:
        tt_by_amount = ds[in_idx].groupby('transaction_type').amount_n26_currency.sum()
        sns.barplot(tt_by_amount.index, tt_by_amount.values, ax=axes[0])
        tt_by_amount = ds[out_idx].groupby('transaction_type').amount_n26_currency.sum()
        sns.barplot(tt_by_amount.index, tt_by_amount.values, ax=axes[1])
    axes[0].set_ylabel('in')
    axes[1].set_ylabel('out')
    plt.show()
    