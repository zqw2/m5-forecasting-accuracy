import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from itertools import cycle
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.util.util import proj_root_dir


# 三指数平滑方法
def run_method():
    # config
    plt.style.use('bmh')
    sns.set_style("whitegrid")
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    warnings.filterwarnings("ignore")
    pd.set_option('max_colwidth', 100)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # 导入数据集：
    data = pd.read_csv(str(proj_root_dir / 'data/data_for_tsa.csv'))
    data['date'] = pd.to_datetime(data['date'])
    print(data.head())

    train = data[data['date'] <= '2016-03-27']
    test = data[(data['date'] > '2016-03-27') & (data['date'] <= '2016-04-24')]

    # plot data
    fig, ax = plt.subplots(figsize=(25, 5))
    train.plot(x='date', y='demand', label='Train', ax=ax)
    test.plot(x='date', y='demand', label='Test', ax=ax);

    predictions = pd.DataFrame()
    predictions['date'] = test['date']
    stats = pd.DataFrame(columns=['Model Name', 'Execution Time', 'RMSE'])

    # 开始调用具体方法
    def lags_windows(df):
        lags = [7]
        lag_cols = ["lag_{}".format(lag) for lag in lags]
        for lag, lag_col in zip(lags, lag_cols):
            df[lag_col] = df[["id", "demand"]].groupby("id")["demand"].shift(lag)

        wins = [7]
        for win in wins:
            for lag, lag_col in zip(lags, lag_cols):
                df["rmean_{}_{}".format(lag, win)] = df[["id", lag_col]].groupby("id")[lag_col].transform(
                    lambda x: x.rolling(win).mean())
        return df

    def per_timeframe_stats(df, col):
        # For each item compute its mean and other descriptive statistics for each month and dayofweek in the dataset
        months = df['month'].unique().tolist()
        for y in months:
            df.loc[df['month'] == y, col + '_month_mean'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(
                lambda x: x.mean()).astype("float32")
            df.loc[df['month'] == y, col + '_month_max'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(
                lambda x: x.max()).astype("float32")
            df.loc[df['month'] == y, col + '_month_min'] = df.loc[df['month'] == y].groupby(['id'])[col].transform(
                lambda x: x.min()).astype("float32")
            df[col + 'month_max_to_min_diff'] = (df[col + '_month_max'] - df[col + '_month_min']).astype("float32")

        dayofweek = df['dayofweek'].unique().tolist()
        for y in dayofweek:
            df.loc[df['dayofweek'] == y, col + '_dayofweek_mean'] = df.loc[df['dayofweek'] == y].groupby(['id'])[
                col].transform(lambda x: x.mean()).astype("float32")
            df.loc[df['dayofweek'] == y, col + '_dayofweek_median'] = df.loc[df['dayofweek'] == y].groupby(['id'])[
                col].transform(lambda x: x.median()).astype("float32")
            df.loc[df['dayofweek'] == y, col + '_dayofweek_max'] = df.loc[df['dayofweek'] == y].groupby(['id'])[
                col].transform(lambda x: x.max()).astype("float32")
        return df

    def feat_eng(df):
        df = lags_windows(df)
        df = per_timeframe_stats(df, 'demand')
        return df

    data = pd.read_csv(str(proj_root_dir / 'data/data_for_tsa.csv'))
    data['date'] = pd.to_datetime(data['date'])
    train = data[data['date'] <= '2016-03-27']
    test = data[(data['date'] > '2016-03-11') & (data['date'] <= '2016-04-24')]

    data_ml = feat_eng(train)
    data_ml = data_ml.dropna()

    useless_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'demand', 'date', 'demand_month_min']
    linreg_train_cols = ['sell_price', 'year', 'month', 'dayofweek', 'lag_7',
                         'rmean_7_7']  # use different columns for linear regression
    lgb_train_cols = data_ml.columns[~data_ml.columns.isin(useless_cols)]
    X_train = data_ml[lgb_train_cols].copy()
    y_train = data_ml["demand"]

    # Fit Light Gradient Boosting
    t0 = time.time()
    lgb_params = {
        "objective": "poisson",
        "metric": "rmse",
        "force_row_wise": True,
        "learning_rate": 0.075,
        "sub_row": 0.75,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        'verbosity': 1,
        'num_iterations': 2000,
        'num_leaves': 128,
        "min_data_in_leaf": 50,
    }
    np.random.seed(777)
    fake_valid_inds = np.random.choice(X_train.index.values, 365, replace=False)
    train_inds = np.setdiff1d(X_train.index.values, fake_valid_inds)
    train_data = lgb.Dataset(X_train.loc[train_inds], label=y_train.loc[train_inds], free_raw_data=False)
    fake_valid_data = lgb.Dataset(X_train.loc[fake_valid_inds], label=y_train.loc[fake_valid_inds], free_raw_data=False)

    m_lgb = lgb.train(lgb_params, train_data, valid_sets=[fake_valid_data], verbose_eval=0)
    t_lgb = time.time() - t0

    # Fit Linear Regression
    t0 = time.time()
    m_linreg = LinearRegression().fit(X_train[linreg_train_cols].loc[train_inds], y_train.loc[train_inds])
    t_linreg = time.time() - t0

    # Fit Random Forest
    t0 = time.time()
    m_rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=26, n_jobs=-1).fit(X_train.loc[train_inds],
                                                                                                y_train.loc[train_inds])
    t_rf = time.time() - t0

    fday = datetime(2016, 3, 28)
    max_lags = 15
    for tdelta in range(0, 28):
        day = fday + timedelta(days=tdelta)
        tst = test[(test.date >= day - timedelta(days=max_lags)) & (test.date <= day)].copy()
        tst = feat_eng(tst)
        tst_lgb = tst.loc[tst.date == day, lgb_train_cols].copy()
        test.loc[test.date == day, "preds_LightGB"] = m_lgb.predict(tst_lgb)
        tst_rf = tst.loc[tst.date == day, lgb_train_cols].copy()
        tst_rf = tst_rf.fillna(0)
        test.loc[test.date == day, "preds_RandomForest"] = m_rf.predict(tst_rf)

        tst_linreg = tst.loc[tst.date == day, linreg_train_cols].copy()
        tst_linreg = tst_linreg.fillna(0)
        test.loc[test.date == day, "preds_LinearReg"] = m_linreg.predict(tst_linreg)

    test_final = test.loc[test.date >= fday]

    model_name = 'RandomForest'
    predictions[model_name] = test_final["preds_" + model_name]

    # visualize
    fig, ax = plt.subplots(figsize=(25, 4))
    train[-28:].plot(x='date', y='demand', label='Train', ax=ax)
    test_final.plot(x='date', y='demand', label='Test', ax=ax);
    predictions.plot(x='date', y=model_name, label=model_name, ax=ax);
    # evaluate
    score = np.sqrt(mean_squared_error(predictions[model_name].values, test_final['demand']))
    print('RMSE for {}: {:.4f}'.format(model_name, score))

    stats = stats.append({'Model Name': model_name, 'Execution Time': t_lgb, 'RMSE': score}, ignore_index=True)

    print("stats: %s" % (stats,))
    plt.show()


if __name__ == '__main__':
    run_method()
